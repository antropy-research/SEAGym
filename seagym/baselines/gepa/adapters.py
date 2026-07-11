from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any

from ..base import BaselineState
from ..data import EvalBatch


def _terminal_bench_task_name(record: dict[str, Any]) -> str:
    task_id = str(record.get("task_id", ""))
    if not task_id.startswith("terminal-bench/"):
        raise ValueError(f"GEPA terminal_bench native_adapter received non-Terminal-Bench task: {task_id!r}")
    return task_id.split("/", 1)[1]


def _task_id_from_record(record: dict[str, Any]) -> str:
    task_id = str(record.get("task_id", ""))
    if not task_id:
        raise ValueError("GEPA harbor native_adapter received trajectory without task_id")
    return task_id


class _SEAGymHarborGEPAAdapter:
    propose_new_texts = None

    def __init__(
        self,
        *,
        runtime: Any,
        baseline_id: str,
        update_dir: Path,
        candidate_component: str,
        candidate_filename: str,
        candidate_view_name: str,
        candidate_mode: str,
        max_reflective_records: int | None,
    ) -> None:
        self.runtime = runtime
        self.baseline_id = baseline_id
        self.update_dir = update_dir
        self.candidate_component = candidate_component
        self.candidate_filename = candidate_filename
        self.candidate_view_name = candidate_view_name
        self.candidate_mode = candidate_mode
        self.max_reflective_records = max_reflective_records
        self._evaluation_index = 0

    def evaluate(self, batch: list[str], candidate: dict[str, str], capture_traces: bool = False) -> Any:
        from gepa import EvaluationBatch  # type: ignore

        self._evaluation_index += 1
        eval_dir = self.update_dir / "candidate_evals" / f"candidate_{self._evaluation_index:04d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = eval_dir / self.candidate_filename
        candidate_path.write_text(str(candidate[self.candidate_component]), encoding="utf-8")
        baseline_state = BaselineState(
            state_dir=eval_dir,
            metadata={
                "baseline_id": self.baseline_id,
                "candidate_path": str(candidate_path),
                "prompt_template_path": str(candidate_path),
                "candidate_component": self.candidate_component,
                "gepa_candidate_eval": True,
            },
        )
        task_ids = [str(task_id) for task_id in batch]
        trajectories = self.runtime.rollout_agent.rollout(
            EvalBatch(task_ids=task_ids, view_name=self.candidate_view_name, mode=self.candidate_mode),
            env=self.runtime.env,
            task_index=self.runtime.task_index,
            baseline_state=baseline_state,
        )
        results = trajectories.to_task_results()
        outputs = [_harbor_rollout_output(result) for result in results]
        scores = [float(result.score) for result in results]
        reflective_trajectories = (
            [_harbor_reflective_trajectory(result, candidate[self.candidate_component]) for result in results]
            if capture_traces
            else None
        )
        (eval_dir / "evaluation_batch.json").write_text(
            json.dumps(
                {
                    "task_ids": task_ids,
                    "scores": scores,
                    "outputs": outputs,
                    "trajectories": reflective_trajectories,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=reflective_trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        dataset = {component: [] for component in components_to_update}
        for trajectory in _limited_reflective_trajectories(eval_batch.trajectories, self.max_reflective_records):
            feedback = "Successfully solved the task." if trajectory.get("success") else "Failed to solve the task."
            if trajectory.get("error"):
                feedback += f" Error: {trajectory['error']}"
            feedback += f" Score: {trajectory.get('score', 0.0)}."
            for component in components_to_update:
                dataset.setdefault(component, []).append(
                    {
                        "Task ID": trajectory.get("task_id"),
                        "Current Component": candidate.get(component, ""),
                        "Harbor Output": trajectory.get("output", ""),
                        "Rewards": trajectory.get("rewards", {}),
                        "Feedback": feedback,
                    }
                )
        return dataset


def _limited_reflective_trajectories(trajectories: Any, limit: int | None) -> list[Any]:
    if not trajectories:
        return []
    items = list(trajectories)
    if limit is None:
        return items
    return items[:limit]


def _harbor_rollout_output(result: Any) -> str:
    return (
        f"Harbor task {result.task_id}: success={result.success} "
        f"score={result.score} error={result.error or ''}".strip()
    )


def _harbor_reflective_trajectory(result: Any, component_text: str) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "success": result.success,
        "score": result.score,
        "rewards": dict(result.rewards),
        "error": result.error,
        "refs": _summarize_harbor_refs(result.refs),
        "output": _harbor_rollout_output(result),
        "candidate_component": component_text,
    }


def _summarize_harbor_refs(refs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("trial_name", "attempt_id", "job_dir", "result_path", "failure_mode"):
        if refs.get(key) not in (None, ""):
            summary[key] = refs[key]
    all_attempts = refs.get("all_attempts")
    if isinstance(all_attempts, list):
        summary["num_attempts"] = len(all_attempts)
    return summary


class _SEAGymTerminusAdapter:
    """Thin wrapper that exposes official run_agent_tb knobs not on TerminusAdapter."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        EvaluationBatch: Any,
        TerminusAdapter: Any,
        get_results: Any,
        run_agent_tb: Any,
        n_concurrent: int,
        instruction_prompt_path: str,
        dataset_name: str = "terminal-bench-core",
        dataset_version: str = "head",
        agent_import_path: str = "train_terminus:TerminusWrapper",
    ) -> None:
        self._EvaluationBatch = EvaluationBatch
        self._get_results = get_results
        self._run_agent_tb = run_agent_tb
        self._official_adapter = TerminusAdapter(
            n_concurrent=n_concurrent,
            instruction_prompt_path=instruction_prompt_path,
        )
        self.n_concurrent = n_concurrent
        self.instruction_prompt_path = instruction_prompt_path
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.agent_import_path = agent_import_path

    def evaluate(self, batch: list[Any], candidate: dict[str, str], capture_traces: bool = False) -> Any:
        del capture_traces
        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        run_id = "temp_gepa_run" + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = batch[0].model_name
        task_ids = [task.task_id for task in batch]

        self._run_agent_tb(
            task_ids,
            run_id,
            model_name,
            instruction_prompt=candidate["instruction_prompt"],
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            agent_import_path=self.agent_import_path,
            n_concurrent=self.n_concurrent,
            prompt_template_path=self.instruction_prompt_path,
        )

        for task in batch:
            try:
                success, score, failed_reason, messages = self._get_results(task.task_id, run_id)
            except Exception as exc:
                success = False
                score = 0
                failed_reason = str(exc)
                messages = []
            outputs.append(
                f"Terminal Bench outputs are omitted. Please see runs/{run_id}/{task.task_id}/ for detailed logging."
            )
            scores.append(score)
            trajectories.append(
                {
                    "messages": messages,
                    "instruction_prompt": candidate["instruction_prompt"],
                    "failed_reason": failed_reason,
                    "success": success,
                }
            )

        return self._EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Any:
        return self._official_adapter.make_reflective_dataset(candidate, eval_batch, components_to_update)


def _terminal_bench_adapter_kwargs(config: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "dataset_name": "dataset_name",
        "dataset_version": "dataset_version",
        "agent_import_path": "agent_import_path",
    }
    return {
        target: str(config[source])
        for source, target in mapping.items()
        if config.get(source) not in (None, "")
    }


def _materialize_terminal_bench_adapter_workspace(project_dir: Path | None, update_dir: Path) -> Path:
    if project_dir is None:
        raise RuntimeError("GEPABaseline requires config.project_dir")
    source_dir = project_dir / "src" / "gepa" / "examples" / "terminal-bench"
    if not source_dir.exists():
        raise FileNotFoundError(f"GEPA Terminal-Bench example directory not found: {source_dir}")
    workspace = update_dir / "terminal_bench_adapter"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "train_terminus.py", workspace / "train_terminus.py")
    prompt_templates = workspace / "prompt-templates"
    if prompt_templates.exists():
        shutil.rmtree(prompt_templates)
    shutil.copytree(source_dir / "prompt-templates", prompt_templates)
    return prompt_templates / "instruction_prompt.txt"


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
