from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any

from ..data import TrajectoryBatch


def _materialize_ahe_evidence(
    *,
    evolve: Any,
    trajectories: TrajectoryBatch,
    exp_dir: Path,
    workspace: Path,
    iteration: int,
    iteration_dir: Path,
) -> dict[str, Any]:
    """Adapt SEAGym trajectories into the AHE native update evidence shape."""

    exp_dir = exp_dir.resolve()
    workspace = workspace.resolve()
    iteration_dir = iteration_dir.resolve()
    input_dir = iteration_dir / "input"
    benchmark_dir = input_dir / "benchmark"
    job_dir = benchmark_dir / "seagym_train_batch"
    job_dir.mkdir(parents=True, exist_ok=True)
    workspace_snapshot = input_dir / "workspace"
    if not workspace_snapshot.exists() and workspace.exists():
        shutil.copytree(workspace, workspace_snapshot, ignore=shutil.ignore_patterns(".git", "run_root"))

    manifest = {
        "type": "ahe_native_evidence",
        "source": "seagym_trajectory_batch",
        "iteration": iteration,
        "job_dir": str(job_dir),
        "workspace_snapshot": str(workspace_snapshot),
        "task_count": len(trajectories.trajectories),
        "trials": [],
    }
    trial_index = 0
    for trajectory in trajectories.trajectories:
        for adapted in _expand_ahe_attempts(trajectory):
            trial_index += 1
            manifest["trials"].append(_materialize_ahe_trial(job_dir, adapted, index=trial_index))

    k = _infer_rollout_k(manifest["trials"])
    stats = evolve.compute_stats(job_dir, k=k)
    task_history = evolve.update_task_history(
        exp_dir,
        iteration,
        stats["task_results"],
        per_task_rollouts=stats.get("per_task_rollouts"),
    )
    stability = evolve.compute_task_stability(task_history)
    prev_task_results, prev_rollouts, prev_stats = _previous_ahe_iteration(task_history, iteration)
    diff = evolve.compute_iteration_diff(
        stats["task_results"],
        prev_task_results,
        current_rollouts=stats.get("per_task_rollouts"),
        prev_rollouts=prev_rollouts,
    )
    best_ever = evolve.update_best_ever(exp_dir, iteration, stats)
    scores_trend = _load_ahe_scores_trend(exp_dir)
    evolve.update_history_before(exp_dir, iteration, stats, job_dir, diff=diff)
    query = evolve.build_evolution_query(
        iteration=iteration,
        stats=stats,
        job_dir=job_dir,
        iteration_dir=iteration_dir,
        prev_stats=prev_stats,
        diff=diff,
        stability=stability,
        best_ever=best_ever,
        scores_trend=scores_trend,
    )
    manifest.update(
        {
            "k": k,
            "stats": _jsonable(stats),
            "diff": _jsonable(diff),
            "stability": _jsonable(stability),
            "best_ever": _jsonable(best_ever),
        }
    )
    manifest_path = input_dir / "ahe_native_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "job_dir": str(job_dir),
        "workspace_snapshot": str(workspace_snapshot),
        "k": k,
        "stats": _jsonable(stats),
        "diff": _jsonable(diff),
        "stability": _jsonable(stability),
        "query": query,
    }


def _expand_ahe_attempts(trajectory: Any) -> list[Any]:
    attempts = trajectory.refs.get("all_attempts") if isinstance(trajectory.refs, dict) else None
    if not isinstance(attempts, list) or not attempts:
        return [trajectory]
    expanded = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        refs = attempt.get("refs")
        if not isinstance(refs, dict):
            refs = {}
        reward = attempt.get("reward")
        if not isinstance(reward, int | float):
            rewards = attempt.get("rewards") if isinstance(attempt.get("rewards"), dict) else {}
            reward = max([value for value in rewards.values() if isinstance(value, int | float)], default=0.0)
        expanded.append(
            replace(
                trajectory,
                attempt_id=attempt.get("attempt_id") or refs.get("trial_name") or trajectory.attempt_id,
                success=bool(attempt.get("success", trajectory.success)),
                reward=float(reward),
                score=float(attempt.get("score", trajectory.score) or 0.0),
                rewards=dict(attempt.get("rewards") if isinstance(attempt.get("rewards"), dict) else trajectory.rewards),
                cost=dict(attempt.get("cost") if isinstance(attempt.get("cost"), dict) else trajectory.cost),
                error=None if attempt.get("error") is None else str(attempt.get("error")),
                refs={**trajectory.refs, **refs},
            )
        )
    return expanded or [trajectory]


def _materialize_ahe_trial(job_dir: Path, trajectory: Any, *, index: int) -> dict[str, Any]:
    task_name = _ahe_task_name(trajectory)
    trial_name = _ahe_trial_name(trajectory, task_name=task_name, index=index)
    trial_dir = job_dir / trial_name
    source_result = Path(str(trajectory.refs.get("result_path", ""))).resolve() if trajectory.refs.get("result_path") else None
    source_trial = source_result.parent if source_result and source_result.exists() else None
    if source_trial and source_trial.is_dir() and not trial_dir.exists():
        try:
            source_trial_resolved = source_trial.resolve()
            trial_dir.symlink_to(source_trial_resolved, target_is_directory=True)
            trace_status = _ahe_trial_trace_status(trial_dir)
            return {
                "task_name": task_name,
                "trial_name": trial_name,
                "trial_dir": str(trial_dir),
                "source_trial_dir": str(source_trial_resolved),
                "mode": "symlink",
                "success": trajectory.success,
                "score": trajectory.score,
                **trace_status,
            }
        except OSError:
            pass

    trial_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir = trial_dir / "verifier"
    agent_dir = trial_dir / "agent"
    verifier_dir.mkdir(exist_ok=True)
    agent_dir.mkdir(exist_ok=True)
    reward = float(trajectory.reward)
    (verifier_dir / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    if trajectory.error:
        (trial_dir / "exception.txt").write_text(str(trajectory.error), encoding="utf-8")
    result = {
        "trial_name": trial_name,
        "task_name": task_name,
        "verifier_result": {"rewards": trajectory.rewards},
        "agent_result": trajectory.cost,
        "exception_info": trajectory.error,
        "source": "seagym_trajectory_batch",
        "seagym_refs": trajectory.refs,
    }
    (trial_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_trial_resolved = source_trial.resolve() if source_trial is not None and source_trial.exists() else None
    _link_optional_trial_artifacts(source_trial_resolved, trial_dir)
    trace_status = _ahe_trial_trace_status(trial_dir)
    return {
        "task_name": task_name,
        "trial_name": trial_name,
        "trial_dir": str(trial_dir),
        "source_trial_dir": None if source_trial_resolved is None else str(source_trial_resolved),
        "mode": "synthetic",
        "success": trajectory.success,
        "score": trajectory.score,
        **trace_status,
    }


def _ahe_trial_trace_status(trial_dir: Path) -> dict[str, Any]:
    cleaned = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
    raw = trial_dir / "agent" / "nexau_in_memory_tracer.json"
    error = trial_dir / "agent" / "trace_export_error.json"
    return {
        "trace_present": cleaned.exists() or raw.exists(),
        "cleaned_trace_present": cleaned.exists(),
        "raw_trace_present": raw.exists(),
        "trace_export_error_present": error.exists(),
    }


def _link_optional_trial_artifacts(source_trial: Path | None, trial_dir: Path) -> None:
    if source_trial is None or not source_trial.exists():
        return
    for child_name in ("agent", "verifier"):
        source_child = (source_trial / child_name).resolve()
        dest_child = trial_dir / child_name
        if not source_child.exists() or dest_child.exists():
            continue
        try:
            dest_child.symlink_to(source_child, target_is_directory=source_child.is_dir())
        except OSError:
            if source_child.is_dir():
                shutil.copytree(source_child, dest_child, dirs_exist_ok=True)
            else:
                shutil.copy2(source_child, dest_child)


def _ahe_task_name(trajectory: Any) -> str:
    harbor_name = trajectory.refs.get("harbor_task_name")
    if harbor_name:
        return str(harbor_name)
    return str(trajectory.task_id).rstrip("/").rsplit("/", 1)[-1]


def _ahe_trial_name(trajectory: Any, *, task_name: str, index: int) -> str:
    raw = trajectory.refs.get("trial_name") or trajectory.attempt_id
    if raw:
        return str(raw).rstrip("/").rsplit("/", 1)[-1]
    return f"{task_name}__seagym{index:06d}"


def _infer_rollout_k(trials: list[dict[str, Any]]) -> int:
    counts: dict[str, int] = {}
    for trial in trials:
        task_name = str(trial["task_name"])
        counts[task_name] = counts.get(task_name, 0) + 1
    return max(counts.values(), default=1)


def _previous_ahe_iteration(task_history: dict[str, Any], iteration: int) -> tuple[dict[str, str] | None, dict | None, dict | None]:
    if iteration <= 1:
        return None, None, None
    prev_task_results: dict[str, str] = {}
    prev_rollouts: dict[str, Any] = {}
    for task_name, entries in task_history.items():
        for entry in entries:
            if entry[0] == iteration - 1:
                prev_task_results[task_name] = entry[1]
                if len(entry) >= 3:
                    prev_rollouts[task_name] = entry[2]
    prev_stats = None
    if prev_task_results:
        n_total = len(prev_task_results)
        n_pass = sum(1 for result in prev_task_results.values() if result == "pass")
        prev_stats = {"pass_rate": n_pass / n_total if n_total else 0.0}
    return prev_task_results or None, prev_rollouts or None, prev_stats


def _load_ahe_scores_trend(exp_dir: Path) -> list[dict[str, Any]] | None:
    scores_path = exp_dir / "iteration_scores.yaml"
    if not scores_path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(scores_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    scores = data.get("scores")
    return scores if isinstance(scores, list) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
