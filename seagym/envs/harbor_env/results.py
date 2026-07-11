from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from seagym.costs import extract_token_cost_from_json_file
from seagym.data.scoring import score_from_reward
from seagym.data.types import TaskRecord
from seagym.utils import read_json

from ..results import TaskRunResult


def normalize_harbor_trial_result(
    task: TaskRecord,
    result_path: str | Path,
    *,
    view_name: str,
    mode: str,
    agent_id: str,
    data: dict[str, Any] | None = None,
) -> TaskRunResult:
    data = data or read_json(result_path)
    verifier = data.get("verifier_result") or {}
    raw_rewards = verifier.get("rewards") or {}
    rewards = {
        str(key): float(value)
        for key, value in raw_rewards.items()
        if isinstance(value, int | float)
    }
    main_key = task.scoring.main_reward_key
    reward = float(rewards.get(main_key, 0.0))
    score = score_from_reward(reward, task.scoring)

    exception_info = data.get("exception_info")
    error = None if exception_info is None else str(exception_info)
    success = exception_info is None and reward >= task.scoring.success_threshold
    agent_result = data.get("agent_result") or {}
    cost = {
        key: float(agent_result[key])
        for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd", "total_tokens")
        if isinstance(agent_result.get(key), int | float)
    }
    refs = {
        "env": "harbor",
        "agent_id": agent_id,
        "result_path": str(result_path),
        "trial_name": data.get("trial_name"),
        "trial_uri": data.get("trial_uri"),
        "job_id": (data.get("config") or {}).get("job_id"),
        "harbor_source": data.get("source"),
        "harbor_task_name": data.get("task_name"),
        "task_checksum": data.get("task_checksum"),
    }
    for timestamp_key in ("started_at", "finished_at"):
        value = data.get(timestamp_key)
        if isinstance(value, str) and value:
            refs[timestamp_key] = value
    if not cost:
        trace_path = Path(result_path).parent / "agent" / "nexau_in_memory_tracer.cleaned.json"
        trace_cost = extract_token_cost_from_json_file(trace_path)
        if trace_cost:
            cost = trace_cost
            refs["cost_source"] = "nexau_cleaned_trace"
            refs["cost_path"] = str(trace_path)
    runtime_seconds = _trial_runtime_seconds(data)
    if runtime_seconds is not None:
        refs["runtime_source"] = "trial_elapsed"
    return TaskRunResult(
        task_id=task.task_id,
        view_name=view_name,
        mode=mode,
        rewards=rewards,
        score=score,
        success=success,
        cost=cost,
        runtime_seconds=runtime_seconds,
        error=error,
        refs=refs,
    )


def _trial_runtime_seconds(data: dict[str, Any]) -> float | None:
    """Return total sandbox-slot occupancy from Harbor's trial timestamps."""
    started_at = _parse_timestamp(data.get("started_at"))
    finished_at = _parse_timestamp(data.get("finished_at"))
    if started_at is None or finished_at is None:
        return None
    elapsed = (finished_at - started_at).total_seconds()
    return elapsed if elapsed >= 0 else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def select_representative_attempt(attempts: list[TaskRunResult]) -> TaskRunResult | None:
    if not attempts:
        return None
    return max(attempts, key=lambda result: (result.success, result.score))


def attempt_ref(result: TaskRunResult) -> dict[str, Any]:
    return {
        "attempt_id": result.refs.get("attempt_id") or result.refs.get("trial_name") or result.refs.get("trial_uri"),
        "success": result.success,
        "reward": max(result.rewards.values(), default=0.0),
        "score": result.score,
        "rewards": dict(result.rewards),
        "cost": dict(result.cost),
        "error": result.error,
        "refs": dict(result.refs),
    }


def failed_harbor_result(
    task: TaskRecord,
    view_name: str,
    mode: str,
    agent_id: str,
    error: str,
    refs: dict[str, Any] | None = None,
) -> TaskRunResult:
    return TaskRunResult(
        task_id=task.task_id,
        view_name=view_name,
        mode=mode,
        rewards={},
        score=0.0,
        success=False,
        error=error,
        refs=refs or {"env": "harbor", "agent_id": agent_id},
    )
