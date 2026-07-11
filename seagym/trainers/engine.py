from __future__ import annotations

"""Composable SEAGym engine primitives and artifact writer.

The ExecutionEngine is the new SEAGym component that sits above Harbor. It should
remain a thin orchestration and bookkeeping layer: users still write readable
Python loops, while Harbor handles actual task execution.

Inputs:
- `ExperimentContext`: config, task index, and split manifest.
- `BatchPlan`: already materialized train batches and views.
- `Env`: rollout environment boundary, currently deterministic or Harbor-backed.
- `agent_id`: Harbor or external agent identifier for this run.

Outputs:
- `inputs/experiment_config.json`, `inputs/split_manifest.json`,
  `inputs/batch_plan.json`.
- `records/evaluation_points.jsonl`: evaluation-point summaries and refs.
- `records/task_results.jsonl`: normalized per-task execution records.
- `records/verifier_results.jsonl`: reward / score / success projection.
- `records/metric_inputs.jsonl`: raw normalized records for offline metrics.

BDD expectations:
- Given an evaluation point and task ids, `run_tasks()` logs one normalized row
  per task to task logs, verifier logs, and metric inputs.
- Given repeated update-validation results, `assess_update()` labels the delta
  as beneficial, neutral, or harmful using config thresholds.
- Given a user-authored loop, ExecutionEngine methods should be usable independently;
  correctness must not depend on an opaque `evaluator.run()` wrapper.

Future work:
- Add final `A_0` vs `A_T` baseline orchestration.
- Add public/private process diagnostics and no-leakage checks.
- Add optional agent-state refs and checkpoint refs to evaluation points.
"""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from seagym.baselines import Baseline, BaselineState, Checkpoint, EvalBatch, ReplayBuffer, TrajectoryBatch, TrainBatch
from seagym.config import ExperimentContext
from seagym.data import BatchPlan
from seagym.envs import Env, TaskRunResult
from seagym.logging import ArtifactLayout, redact_sensitive
from seagym.rollout_agents import RolloutAgent
from seagym.scheduling import RuntimeScheduler
from seagym.trainers.assessment import assess_update_results
from seagym.trainers.checkpoint import (
    TrainerState,
    load_checkpoint_manifest,
    resolve_checkpoint,
    write_checkpoint_manifest,
    write_latest_checkpoint,
)
from seagym.utils import append_jsonl, write_json


@dataclass
class EvaluationPoint:
    evaluation_point_id: str
    point_type: str
    train_batch_index: int
    num_train_tasks_seen: int
    agent_id: str
    evaluations: dict[str, Any] = field(default_factory=dict)
    update_assessment: dict[str, Any] = field(default_factory=lambda: {"label": "not_applicable"})
    refs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_point_id": self.evaluation_point_id,
            "type": self.point_type,
            "step": {
                "train_batch_index": self.train_batch_index,
                "num_train_tasks_seen": self.num_train_tasks_seen,
            },
            "agent_state": {
                "agent_id": self.agent_id,
            },
            "evaluations": self.evaluations,
            "update_assessment": self.update_assessment,
            "refs": self.refs,
        }


class ExecutionEngine:
    def __init__(
        self,
        context: ExperimentContext,
        batch_plan: BatchPlan,
        env: Env,
        *,
        agent_id: str = "default",
        baseline: Baseline | None = None,
        baseline_state: BaselineState | None = None,
        rollout_agent: RolloutAgent | None = None,
        layout: ArtifactLayout | None = None,
    ):
        self.context = context
        self.batch_plan = batch_plan
        self.env = env
        self.agent_id = agent_id
        self.baseline = baseline
        self.baseline_state = baseline_state
        self.rollout_agent = rollout_agent
        self.replay_buffer = ReplayBuffer()
        self._last_train_trajectories: TrajectoryBatch | None = None
        self.layout = layout or ArtifactLayout.from_run_dir(context.config.run_dir)
        self.run_dir = self.layout.run_dir
        self.task_log_path = self.layout.task_results_path
        self.metric_input_path = self.layout.metric_inputs_path
        self.evaluation_points_path = self.layout.evaluation_points_path
        self.verifier_results_path = self.layout.verifier_results_path
        self.agent_updates_path = self.layout.agent_updates_path
        self.agent_checkpoints_path = self.layout.agent_checkpoints_path
        self.runtime_scheduler = (
            RuntimeScheduler.from_path(self.context.config.runtime_scheduling, self.layout.scheduling_history_path)
            if self.context.config.runtime_scheduling.enabled
            else None
        )
        self._bind_baseline_runtime()

    def write_run_inputs(self) -> None:
        write_json(self.layout.experiment_config_path, self.context.config.raw)
        write_json(self.layout.split_manifest_path, _split_to_dict(self.context))
        write_json(self.layout.batch_plan_path, self.batch_plan.to_dict())

    def make_train_batches(self) -> list[list[str]]:
        return self.batch_plan.train_batches

    def materialize_view(self, name: str) -> list[str] | dict[str, list[str]]:
        if name == "final":
            return self.batch_plan.views.get("final", {})
        return list(self.batch_plan.views.get(name, []))

    def final_views(self) -> dict[str, list[str]]:
        return dict(self.batch_plan.views.get("final", {}))

    def should_validate(self, train_batch_index: int) -> bool:
        num_epochs = self.context.config.schedule.num_epochs
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        num_batches = len(self.batch_plan.train_batches)
        if num_batches % num_epochs != 0:
            raise ValueError(
                "Cannot infer epoch boundaries: materialized train batch count "
                f"{num_batches} is not divisible by num_epochs {num_epochs}"
            )
        batches_per_epoch = num_batches // num_epochs
        return batches_per_epoch > 0 and train_batch_index % batches_per_epoch == 0

    def should_replay(self, train_batch_index: int) -> bool:
        replay = self.context.config.evaluation_strategy.get("replay", {})
        if not replay.get("enabled", False):
            return False
        freq = int(replay.get("frequency", 0))
        return freq > 0 and train_batch_index % freq == 0

    def record_evaluation_point(
        self,
        *,
        point_type: str,
        train_batch_index: int,
        num_train_tasks_seen: int,
    ) -> EvaluationPoint:
        if point_type == "initial":
            point_id = "E_0"
        elif point_type == "final":
            point_id = "E_T"
        else:
            point_id = f"E_{train_batch_index}"
        return EvaluationPoint(
            evaluation_point_id=point_id,
            point_type=point_type,
            train_batch_index=train_batch_index,
            num_train_tasks_seen=num_train_tasks_seen,
            agent_id=self.agent_id,
        )

    def write_evaluation_point(
        self,
        point: EvaluationPoint,
        *,
        evaluations: dict[str, Any],
        refs: dict[str, Any] | None = None,
        update_assessment: dict[str, Any] | None = None,
    ) -> EvaluationPoint:
        point.evaluations = evaluations
        if refs is not None:
            point.refs = refs
        if update_assessment is not None:
            point.update_assessment = update_assessment
        append_jsonl(self.evaluation_points_path, point.to_dict())
        return point

    def update_agent(
        self,
        batch_results: list[TaskRunResult],
        *,
        train_batch_index: int,
        num_train_tasks_seen: int,
        update_repeat_index: int = 1,
        num_updates_per_batch: int = 1,
        global_update_index: int | None = None,
    ) -> dict[str, Any]:
        """Run the post-train-batch baseline lifecycle update.

        This call is intentionally outside Harbor trial execution. Harbor runs
        rollout agents; SEAGym owns cross-batch state transitions and audit logs.
        """
        trajectories = self._last_train_trajectories or TrajectoryBatch.from_task_results(
            batch_results,
            task_ids=[result.task_id for result in batch_results],
            view_name="train",
            mode="train",
            batch_index=train_batch_index,
        )
        trajectories = _redact_trajectory_batch(trajectories)
        state = self._require_baseline_state()
        update_result = self._require_baseline().update(trajectories, state)
        summary = update_result.to_dict()
        summary.setdefault("num_records", len(trajectories.trajectories))
        row = {
            "run_id": self.batch_plan.run_id,
            "experiment_id": self.context.config.experiment_id,
            "agent_id": self.agent_id,
            "train_batch_index": train_batch_index,
            "update_repeat_index": update_repeat_index,
            "num_updates_per_batch": num_updates_per_batch,
            "global_update_index": global_update_index,
            "num_train_tasks_seen": num_train_tasks_seen,
            "summary": summary,
        }
        append_jsonl(self.agent_updates_path, row)
        append_jsonl(
            self.metric_input_path,
            {
                "run_id": self.batch_plan.run_id,
                "experiment_id": self.context.config.experiment_id,
                "split_id": self.context.split.split_id,
                "evaluation_point_id": None,
                "agent_id": self.agent_id,
                "agent_checkpoint_id": None,
                "baseline_role": None,
                "task_id": None,
                "attributes": {},
                "view_name": "agent_update",
                "mode": "update",
                "score": None,
                "success": None,
                "rewards": {},
                "cost": _extract_update_cost(summary),
                "error": summary.get("error"),
                "refs": {},
                "train_batch_index": train_batch_index,
                "update_repeat_index": update_repeat_index,
                "num_updates_per_batch": num_updates_per_batch,
                "global_update_index": global_update_index,
                "num_train_tasks_seen": num_train_tasks_seen,
                "update_summary": redact_sensitive(summary),
            },
        )
        return row

    def save_checkpoint(
        self,
        checkpoint_id: str,
        *,
        trainer_state: TrainerState,
        checkpoint_type: str = "epoch",
        metadata: dict[str, Any] | None = None,
        mark_latest: bool = True,
    ) -> dict[str, Any]:
        checkpoint_dir = self.layout.checkpoints_dir / checkpoint_id
        checkpoint = self._require_baseline().save_checkpoint(self._require_baseline_state(), checkpoint_dir)
        baseline_manifest = dict(checkpoint.metadata)
        baseline_manifest.setdefault("type", "baseline_checkpoint")
        baseline_manifest.setdefault("checkpoint_dir", str(checkpoint.checkpoint_dir))
        if checkpoint.state_ref is not None:
            baseline_manifest.setdefault("state_ref", checkpoint.state_ref)
        manifest = write_checkpoint_manifest(
            checkpoint_dir,
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            run_id=self.batch_plan.run_id,
            experiment_id=self.context.config.experiment_id,
            trainer_state=trainer_state,
            metadata=metadata,
            refs={
                "baseline_state": baseline_manifest.get("state_ref"),
                "batch_plan": str(self.layout.batch_plan_path),
                "config": str(self.layout.experiment_config_path),
            },
            baseline_manifest=baseline_manifest,
        )
        if mark_latest:
            write_latest_checkpoint(self.layout.run_dir, checkpoint_id)
        append_jsonl(
            self.agent_checkpoints_path,
            {
                "run_id": self.batch_plan.run_id,
                "experiment_id": self.context.config.experiment_id,
                "agent_id": self.agent_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_type,
                "metadata": metadata or {},
                "manifest": redact_sensitive(manifest),
            },
        )
        return manifest

    def alias_checkpoint(
        self,
        checkpoint_id: str,
        *,
        source_checkpoint_id: str,
        trainer_state: TrainerState,
        checkpoint_type: str = "epoch",
        metadata: dict[str, Any] | None = None,
        mark_latest: bool = True,
    ) -> dict[str, Any]:
        source_dir = self.layout.checkpoints_dir / source_checkpoint_id
        source_manifest_path = source_dir / "checkpoint.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Source checkpoint manifest not found: {source_manifest_path}")
        checkpoint_dir = self.layout.checkpoints_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = load_checkpoint_manifest(source_dir)
        baseline_manifest = dict(source_manifest.get("baseline") or {})
        baseline_manifest["type"] = "baseline_checkpoint_alias"
        baseline_manifest["alias_of"] = source_checkpoint_id
        baseline_manifest["source_checkpoint_id"] = source_checkpoint_id
        baseline_manifest["checkpoint_dir"] = str(checkpoint_dir)
        state_ref = baseline_manifest.get("state_ref")
        if isinstance(state_ref, str):
            source_path = Path(state_ref)
            baseline_manifest["state_ref"] = str(
                source_path if source_path.is_absolute() else Path("..") / source_checkpoint_id / source_path
            )
        manifest = write_checkpoint_manifest(
            checkpoint_dir,
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            run_id=self.batch_plan.run_id,
            experiment_id=self.context.config.experiment_id,
            trainer_state=trainer_state,
            metadata={**(metadata or {}), "alias_of": source_checkpoint_id},
            refs={
                "baseline_state": baseline_manifest.get("state_ref"),
                "batch_plan": str(self.layout.batch_plan_path),
                "config": str(self.layout.experiment_config_path),
            },
            baseline_manifest=baseline_manifest,
        )
        if mark_latest:
            write_latest_checkpoint(self.layout.run_dir, checkpoint_id)
        append_jsonl(
            self.agent_checkpoints_path,
            {
                "run_id": self.batch_plan.run_id,
                "experiment_id": self.context.config.experiment_id,
                "agent_id": self.agent_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_type,
                "metadata": metadata or {},
                "manifest": redact_sensitive(manifest),
            },
        )
        return manifest

    def load_checkpoint(self, checkpoint_id: str | Path) -> dict[str, Any]:
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        checkpoint = Checkpoint(checkpoint_dir=checkpoint_dir)
        self.baseline_state = self._require_baseline().load_checkpoint(checkpoint)
        return {
            "type": "baseline_checkpoint_load",
            "loaded": True,
            "checkpoint_dir": str(checkpoint_dir),
            "state": self.baseline_state.metadata,
        }

    def _checkpoint_dir(self, checkpoint_id: str | Path) -> Path:
        return resolve_checkpoint(self.layout.run_dir, checkpoint_id)

    def run_tasks(
        self,
        task_ids: list[str],
        *,
        view_name: str,
        mode: str,
        evaluation_point: EvaluationPoint | None = None,
        agent_checkpoint_id: str | None = None,
        baseline_role: str | None = None,
        train_batch_index: int | None = None,
        update_repeat_index: int | None = None,
        num_updates_per_batch: int | None = None,
        global_update_index: int | None = None,
        num_train_tasks_seen: int | None = None,
    ) -> list[TaskRunResult]:
        """Execute a materialized task list and write normalized artifacts.

        This is intentionally view-oriented rather than split-oriented. The
        caller decides whether the task ids are train, frozen update-validation,
        replay, final ID test, OOD test, or a negative-transfer probe.

        A single task is represented as a batch of size one at the environment
        boundary. ExecutionEngine always uses the batch-first env protocol.
        """
        batch_cls = TrainBatch if mode == "train" else EvalBatch
        batch_metadata = {
            "agent_checkpoint_id": agent_checkpoint_id,
            "baseline_role": baseline_role,
            "train_batch_index": train_batch_index,
            "update_repeat_index": update_repeat_index,
            "num_updates_per_batch": num_updates_per_batch,
            "global_update_index": global_update_index,
            "num_train_tasks_seen": num_train_tasks_seen,
        }
        original_task_ids = list(task_ids)
        decision = None
        if self.runtime_scheduler is not None and mode in self.context.config.runtime_scheduling.apply_to:
            decision = self.runtime_scheduler.plan(
                original_task_ids,
                mode=mode,
                workers=self._worker_count(),
            )
        execution_task_ids = original_task_ids if decision is None else decision.scheduled_task_ids
        batch = batch_cls(
            task_ids=execution_task_ids,
            view_name=view_name,
            mode=mode,
            batch_index=train_batch_index,
            metadata={key: value for key, value in batch_metadata.items() if value is not None},
        )
        trajectories = self._require_rollout_agent().rollout(
            batch,
            env=self.env,
            task_index=self.context.task_index,
            baseline_state=self._require_baseline_state(),
        )
        results = trajectories.to_task_results()
        if decision is not None:
            results = _restore_task_order(results, original_task_ids)
            trajectories = TrajectoryBatch.from_task_results(
                results,
                task_ids=original_task_ids,
                view_name=trajectories.view_name,
                mode=trajectories.mode,
                batch_index=trajectories.batch_index,
                epoch=trajectories.epoch,
                refs={**trajectories.refs, "scheduling_decision_index": decision.decision_index},
            )
        trajectories = _attach_trajectory_metadata(
            trajectories,
            train_batch_index=train_batch_index,
            update_repeat_index=update_repeat_index,
            num_updates_per_batch=num_updates_per_batch,
            global_update_index=global_update_index,
            num_train_tasks_seen=num_train_tasks_seen,
        )
        results = trajectories.to_task_results()
        if decision is not None and self.runtime_scheduler is not None:
            runtime_rows, diagnostics = self.runtime_scheduler.observe(decision, results)
            append_jsonl(
                self.layout.scheduling_decisions_path,
                {
                    **decision.to_dict(),
                    "train_batch_index": train_batch_index,
                    "diagnostics": diagnostics,
                },
            )
            for runtime_row in runtime_rows:
                append_jsonl(self.layout.task_runtimes_path, runtime_row)
            write_json(self.layout.scheduling_summary_path, self.runtime_scheduler.summary())
        if mode == "train":
            self.replay_buffer.add(trajectories)
            self._last_train_trajectories = trajectories

        tasks_by_id = {task.task_id: task for task in (self.context.task_index.require(task_id) for task_id in task_ids)}
        for result in results:
            task = tasks_by_id.get(result.task_id)
            if task is None:
                raise ValueError(
                    f"Rollout result task_id={result.task_id!r} was not requested in batch {list(task_ids)!r}"
                )
            row = self._result_row(
                result,
                task.attributes,
                evaluation_point,
                agent_checkpoint_id=agent_checkpoint_id,
                baseline_role=baseline_role,
                train_batch_index=train_batch_index,
                update_repeat_index=update_repeat_index,
                num_updates_per_batch=num_updates_per_batch,
                global_update_index=global_update_index,
                num_train_tasks_seen=num_train_tasks_seen,
            )
            append_jsonl(self.task_log_path, row)
            append_jsonl(self.metric_input_path, row)
            append_jsonl(
                self.verifier_results_path,
                {
                    "task_id": result.task_id,
                    "view_name": view_name,
                    "rewards": result.rewards,
                    "score": result.score,
                    "success": result.success,
                    "error": result.error,
                },
            )
        return results

    def _worker_count(self) -> int:
        return max(1, int(getattr(self.env, "n_concurrent", 1)))

    def _require_baseline_state(self) -> BaselineState:
        if self.baseline_state is None:
            raise RuntimeError("Baseline has not been initialized")
        return self.baseline_state

    def _require_baseline(self) -> Baseline:
        if self.baseline is None:
            raise RuntimeError("Baseline has not been configured")
        return self.baseline

    def _require_rollout_agent(self) -> RolloutAgent:
        if self.rollout_agent is None:
            raise RuntimeError("Rollout agent has not been configured")
        return self.rollout_agent

    def _bind_baseline_runtime(self) -> None:
        if self.baseline is None or self.rollout_agent is None:
            return
        bind_runtime = getattr(self.baseline, "bind_runtime", None)
        if bind_runtime is None:
            return
        bind_runtime(
            env=self.env,
            task_index=self.context.task_index,
            rollout_agent=self.rollout_agent,
            run_dir=self.run_dir,
            batch_plan=self.batch_plan,
        )

    def assess_update(
        self,
        current_results: list[TaskRunResult],
        previous_results: list[TaskRunResult],
    ) -> dict[str, Any]:
        return assess_update_results(current_results, previous_results, self.context.config.metrics)

    def _result_row(
        self,
        result: TaskRunResult,
        attributes: dict[str, Any],
        evaluation_point: EvaluationPoint | None,
        *,
        agent_checkpoint_id: str | None = None,
        baseline_role: str | None = None,
        train_batch_index: int | None = None,
        update_repeat_index: int | None = None,
        num_updates_per_batch: int | None = None,
        global_update_index: int | None = None,
        num_train_tasks_seen: int | None = None,
    ) -> dict[str, Any]:
        if train_batch_index is None and evaluation_point is not None:
            train_batch_index = evaluation_point.train_batch_index
        if num_train_tasks_seen is None and evaluation_point is not None:
            num_train_tasks_seen = evaluation_point.num_train_tasks_seen
        return {
            "run_id": self.batch_plan.run_id,
            "experiment_id": self.context.config.experiment_id,
            "split_id": self.context.split.split_id,
            "evaluation_point_id": None if evaluation_point is None else evaluation_point.evaluation_point_id,
            "agent_id": self.agent_id,
            "agent_checkpoint_id": agent_checkpoint_id,
            "baseline_role": baseline_role,
            "task_id": result.task_id,
            "attributes": attributes,
            "view_name": result.view_name,
            "mode": result.mode,
            "score": result.score,
            "success": result.success,
            "rewards": result.rewards,
            "cost": result.cost,
            "runtime_seconds": result.runtime_seconds,
            "error": result.error,
            "refs": redact_sensitive(result.refs),
            "train_batch_index": train_batch_index,
            "update_repeat_index": update_repeat_index,
            "num_updates_per_batch": num_updates_per_batch,
            "global_update_index": global_update_index,
            "num_train_tasks_seen": num_train_tasks_seen,
        }


def _split_to_dict(context: ExperimentContext) -> dict[str, Any]:
    split = context.split
    return {
        "split_id": split.split_id,
        "split_version": split.split_version,
        "seed": split.seed,
        "splits": {
            "train": split.train,
            "val": split.val,
            "test": split.test,
        },
    }


def _extract_update_cost(summary: dict[str, Any]) -> dict[str, float]:
    """Normalize baseline-update cost fields when an implementation exposes them."""
    source = summary.get("cost")
    if not isinstance(source, dict):
        logs = summary.get("logs")
        metrics = summary.get("metrics")
        if isinstance(logs, dict) and isinstance(logs.get("cost"), dict):
            source = logs["cost"]
        elif isinstance(metrics, dict) and isinstance(metrics.get("cost"), dict):
            source = metrics["cost"]
        else:
            source = summary.get("usage") if isinstance(summary.get("usage"), dict) else summary
    allowed = {
        "tokens",
        "total_tokens",
        "n_total_tokens",
        "input_tokens",
        "n_input_tokens",
        "cache_tokens",
        "n_cache_tokens",
        "output_tokens",
        "n_output_tokens",
        "cost_usd",
        "tool_calls",
        "wall_time",
    }
    return {
        str(key): float(value)
        for key, value in source.items()
        if key in allowed and isinstance(value, int | float)
    }


def _restore_task_order(results: list[TaskRunResult], task_ids: list[str]) -> list[TaskRunResult]:
    """Preserve the logical BatchPlan order for baseline consumers and records."""
    by_task: dict[str, list[TaskRunResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)
    ordered: list[TaskRunResult] = []
    for task_id in task_ids:
        ordered.extend(by_task.pop(task_id, []))
    for result in results:
        if result.task_id in by_task:
            ordered.extend(by_task.pop(result.task_id))
    return ordered


def _redact_trajectory_batch(batch: TrajectoryBatch) -> TrajectoryBatch:
    return replace(
        batch,
        trajectories=[
            replace(
                trajectory,
                refs=redact_sensitive(trajectory.refs),
                task_result=(
                    None
                    if trajectory.task_result is None
                    else replace(trajectory.task_result, refs=redact_sensitive(trajectory.task_result.refs))
                ),
            )
            for trajectory in batch.trajectories
        ],
        refs=redact_sensitive(batch.refs),
    )


def _attach_trajectory_metadata(
    batch: TrajectoryBatch,
    *,
    train_batch_index: int | None = None,
    update_repeat_index: int | None = None,
    num_updates_per_batch: int | None = None,
    global_update_index: int | None = None,
    num_train_tasks_seen: int | None = None,
) -> TrajectoryBatch:
    metadata = {
        "train_batch_index": train_batch_index,
        "update_repeat_index": update_repeat_index,
        "num_updates_per_batch": num_updates_per_batch,
        "global_update_index": global_update_index,
        "num_train_tasks_seen": num_train_tasks_seen,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    if not metadata:
        return batch
    return replace(
        batch,
        batch_index=train_batch_index if train_batch_index is not None else batch.batch_index,
        refs={**batch.refs, **metadata},
        trajectories=[
            replace(trajectory, refs={**trajectory.refs, **metadata})
            for trajectory in batch.trajectories
        ],
    )
