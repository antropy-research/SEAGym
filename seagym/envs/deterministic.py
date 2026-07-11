from __future__ import annotations

"""Deterministic environment for dry runs and tests."""

from seagym.data.scoring import score_from_reward
from seagym.data.types import TaskRecord
from .results import TaskRunResult


class DeterministicEnv:
    """Local deterministic environment for dry runs and tests."""

    def run_tasks(
        self,
        tasks: list[TaskRecord],
        *,
        view_name: str,
        mode: str,
        agent_id: str,
    ) -> list[TaskRunResult]:
        return [
            _deterministic_result(task, view_name=view_name, mode=mode, agent_id=agent_id)
            for task in tasks
        ]


def _deterministic_result(task: TaskRecord, *, view_name: str, mode: str, agent_id: str) -> TaskRunResult:
    fixtures = task.fixtures or {}
    scores = fixtures.get("scores", {})
    raw_score = scores.get(view_name, scores.get(mode, scores.get("default", 1.0)))
    reward_key = task.scoring.main_reward_key
    reward = float(raw_score)
    score = score_from_reward(reward, task.scoring)
    success = reward >= task.scoring.success_threshold
    return TaskRunResult(
        task_id=task.task_id,
        view_name=view_name,
        mode=mode,
        rewards={reward_key: reward},
        score=score,
        success=success,
        cost={
            "tokens": float(fixtures.get("tokens", 0)),
            "wall_time": float(fixtures.get("wall_time", 0)),
        },
        runtime_seconds=float(fixtures.get("wall_time", 0)),
        refs={
            "env": "deterministic",
            "agent_id": agent_id,
            "harbor_dataset": task.source.get("dataset"),
            "harbor_task_name": task.source.get("task_name"),
        },
    )
