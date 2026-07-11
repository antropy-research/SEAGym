from __future__ import annotations

"""SEAGym trainer lifecycle."""

from dataclasses import replace
from pathlib import Path
from typing import Any
import warnings

from seagym.baselines import Baseline, StaticBaseline
from seagym.config import load_experiment_config, load_experiment_context
from seagym.data import SEAGymDataModule, BatchPlan
from seagym.envs import DeterministicEnv, Env
from seagym.logging import ArtifactLayout, write_run_reports
from seagym.metrics import MetricRegistry
from seagym.rollout_agents import RolloutAgent
from seagym.trainers.engine import ExecutionEngine
from seagym.trainers.loops import UpdateValidationLoop
from seagym.trainers.checkpoint import TrainerState, load_checkpoint_manifest, resolve_checkpoint
from seagym.trainers.run import RunOptions, make_run_dir
from seagym.runtime import inspect_experiment_config
from seagym.utils import read_jsonl, write_json


class SEAGymTrainer:
    """Compose data, loop, backend, baseline, artifacts, and metrics."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        agent_id: str = "default",
        run_dir: str | Path | None = None,
        overwrite: bool = True,
        baseline: Baseline | None = None,
        rollout_agent: RolloutAgent | None = None,
        env: Env | None = None,
        loop: UpdateValidationLoop | None = None,
        run_options: RunOptions | None = None,
    ):
        self.config_path = Path(config_path)
        self.agent_id = agent_id
        self.overwrite = overwrite
        self.baseline = baseline
        self.rollout_agent = rollout_agent
        self.env = env or DeterministicEnv()
        self.loop = loop or UpdateValidationLoop()
        context = load_experiment_context(self.config_path)
        if run_options is not None:
            run_dir = make_run_dir(
                experiment_id=context.config.experiment_id,
                output_dir=run_options.output_dir,
                run_dir=run_options.run_dir or run_dir,
                run_name=run_options.run_name,
            )
            self.overwrite = run_options.overwrite
        if run_dir is not None:
            context = replace(context, config=replace(context.config, run_dir=Path(run_dir).resolve()))
        self.context = context
        self.layout = ArtifactLayout.from_run_dir(self.context.config.run_dir)
        self._warn_for_runtime_risks()
        self.batch_plan: BatchPlan | None = None
        self.engine: ExecutionEngine | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        overwrite: bool = True,
        overrides=None,
        run_options: RunOptions | None = None,
    ) -> "SEAGymTrainer":
        from seagym.trainers.builder import build_trainer_components, trainer_overrides_from_values

        effective_run_options = run_options
        if run_options is not None:
            config = load_experiment_config(config_path)
            resolved_run_dir = make_run_dir(
                experiment_id=config.experiment_id,
                output_dir=run_options.output_dir,
                run_dir=run_options.run_dir,
                run_name=run_options.run_name,
            )
            effective_run_options = replace(run_options, run_dir=resolved_run_dir)
            overrides = trainer_overrides_from_values(run_dir=resolved_run_dir)
        components = build_trainer_components(config_path, overrides=overrides)
        return cls(
            config_path,
            agent_id=components.agent_id,
            run_dir=components.run_dir,
            overwrite=overwrite,
            baseline=components.baseline,
            rollout_agent=components.rollout_agent,
            env=components.env,
            run_options=effective_run_options,
        )

    def prepare(self, *, reset_run_dir: bool = True) -> ExecutionEngine:
        self.batch_plan = SEAGymDataModule(self.context).build()
        print(
            "seagym progress: trainer prepared batch plan "
            f"experiment={self.context.config.experiment_id} "
            f"train_batches={len(self.batch_plan.train_batches)} "
            f"updates_per_batch={self.context.config.schedule.num_updates_per_batch} "
            f"derived_num_updates={len(self.batch_plan.train_batches) * self.context.config.schedule.num_updates_per_batch} "
            f"run_dir={self.layout.run_dir}",
            flush=True,
        )
        if reset_run_dir:
            self.layout.prepare(overwrite=self.overwrite)
        else:
            for path in (
                self.layout.inputs_dir,
                self.layout.records_dir,
                self.layout.reports_dir,
                self.layout.checkpoints_dir,
                self.layout.harbor_jobs_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
        if self.baseline is None:
            self.baseline = StaticBaseline(
                baseline_id=self.agent_id,
                state_dir=self.layout.run_dir / "agent_state" / self.agent_id,
            )
        if self.rollout_agent is None:
            from seagym.rollout_agents.harbor import HarborRolloutAgent

            self.rollout_agent = HarborRolloutAgent(agent_id=self.agent_id)
        baseline_state = self.baseline.initialize(self.layout.run_dir)
        self.rollout_agent.initialize(self.layout.run_dir)
        self.engine = ExecutionEngine(
            self.context,
            self.batch_plan,
            self.env,
            agent_id=self.agent_id,
            baseline=self.baseline,
            baseline_state=baseline_state,
            rollout_agent=self.rollout_agent,
            layout=self.layout,
        )
        self.engine.write_run_inputs()
        return self.engine

    def fit(self, *, resume_from_checkpoint: str | Path | None = None) -> Path:
        engine = self.prepare(reset_run_dir=resume_from_checkpoint is None)
        resume_state = None
        if resume_from_checkpoint is not None:
            checkpoint_dir = resolve_checkpoint(self.layout.run_dir, resume_from_checkpoint)
            manifest = load_checkpoint_manifest(checkpoint_dir)
            trainer_state = manifest.get("trainer_state")
            if not isinstance(trainer_state, dict):
                raise ValueError(f"Checkpoint missing trainer_state: {checkpoint_dir}")
            resume_state = TrainerState.from_dict({**trainer_state, "checkpoint_id": str(checkpoint_dir)})
        print("seagym progress: trainer loop started", flush=True)
        self.loop.run(engine, resume=resume_state)
        print("seagym progress: trainer loop finished", flush=True)
        print("seagym progress: metrics computation started", flush=True)
        self.compute_metrics()
        print("seagym progress: metrics computation finished", flush=True)
        print("seagym progress: report writing started", flush=True)
        write_run_reports(self.layout.run_dir)
        print("seagym progress: report writing finished", flush=True)
        return self.layout.run_dir

    def compute_metrics(self) -> dict[str, Any]:
        records = read_jsonl(self.layout.metric_inputs_path)
        registry = MetricRegistry.from_config(self.context.config.metrics)
        metric_names = _metric_names_from_config(self.context.config.metrics)
        metrics = registry.compute(records, self.context.config.metrics, metric_names=metric_names)
        write_json(self.layout.metrics_path, metrics)
        return metrics

    def _warn_for_runtime_risks(self) -> None:
        backend = self.context.config.raw.get("backend") or {}
        if not isinstance(backend, dict) or backend.get("name") != "harbor":
            return
        for check in inspect_experiment_config(self.config_path, run_dir=self.context.config.run_dir):
            if check.status == "warn":
                warnings.warn(f"{check.name}: {check.detail}", RuntimeWarning, stacklevel=2)


def _metric_names_from_config(metrics_config: dict[str, Any]) -> list[str] | None:
    names: list[str] = []
    for name in metrics_config.get("primary") or []:
        if name not in names:
            names.append(str(name))
    cost_values = metrics_config["cost"] if "cost" in metrics_config else ["tokens"]
    for name in cost_values or []:
        if name not in names:
            names.append(str(name))
    return names or None
