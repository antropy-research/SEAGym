from __future__ import annotations

"""ML/RL-style data containers used by baseline lifecycles."""

from dataclasses import asdict, dataclass, field
from typing import Any

from seagym.envs import TaskRunResult


@dataclass(frozen=True)
class TaskBatch:
    task_ids: list[str]
    view_name: str
    mode: str
    batch_index: int | None = None
    epoch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainBatch(TaskBatch):
    view_name: str = "train"
    mode: str = "train"


@dataclass(frozen=True)
class EvalBatch(TaskBatch):
    pass


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    attempt_id: str | int | None
    view_name: str
    mode: str
    success: bool
    reward: float
    score: float
    rewards: dict[str, float]
    cost: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float | None = None
    error: str | None = None
    refs: dict[str, Any] = field(default_factory=dict)
    task_result: TaskRunResult | None = None

    @classmethod
    def from_task_result(cls, result: TaskRunResult) -> "Trajectory":
        attempt_id = result.refs.get("attempt_id") or result.refs.get("trial_name") or result.refs.get("trial_uri")
        reward = 0.0 if not result.rewards else max(result.rewards.values())
        return cls(
            task_id=result.task_id,
            attempt_id=attempt_id,
            view_name=result.view_name,
            mode=result.mode,
            success=result.success,
            reward=float(reward),
            score=result.score,
            rewards=dict(result.rewards),
            cost=dict(result.cost),
            runtime_seconds=result.runtime_seconds,
            error=result.error,
            refs=dict(result.refs),
            task_result=result,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("task_result", None)
        return data


@dataclass(frozen=True)
class TrajectoryBatch:
    trajectories: list[Trajectory]
    task_ids: list[str]
    view_name: str
    mode: str
    batch_index: int | None = None
    epoch: int | None = None
    refs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_results(
        cls,
        results: list[TaskRunResult],
        *,
        task_ids: list[str],
        view_name: str,
        mode: str,
        batch_index: int | None = None,
        epoch: int | None = None,
        refs: dict[str, Any] | None = None,
    ) -> "TrajectoryBatch":
        return cls(
            trajectories=[Trajectory.from_task_result(result) for result in results],
            task_ids=list(task_ids),
            view_name=view_name,
            mode=mode,
            batch_index=batch_index,
            epoch=epoch,
            refs=dict(refs or {}),
        )

    def to_task_results(self) -> list[TaskRunResult]:
        results: list[TaskRunResult] = []
        for trajectory in self.trajectories:
            if trajectory.task_result is not None:
                results.append(trajectory.task_result)
                continue
            results.append(
                TaskRunResult(
                    task_id=trajectory.task_id,
                    view_name=trajectory.view_name,
                    mode=trajectory.mode,
                    rewards=dict(trajectory.rewards),
                    score=trajectory.score,
                    success=trajectory.success,
                    cost=dict(trajectory.cost),
                    runtime_seconds=trajectory.runtime_seconds,
                    error=trajectory.error,
                    refs=dict(trajectory.refs),
                )
            )
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_ids": self.task_ids,
            "view_name": self.view_name,
            "mode": self.mode,
            "batch_index": self.batch_index,
            "epoch": self.epoch,
            "refs": self.refs,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }


@dataclass
class ReplayBuffer:
    """Minimal latest-batch buffer.

    The first implementation keeps on-policy semantics by returning only the
    latest train batch. Sampling strategies can be added later when an
    experiment explicitly wants replay or off-policy updates.
    """

    batches: list[TrajectoryBatch] = field(default_factory=list)

    def add(self, batch: TrajectoryBatch) -> None:
        self.batches.append(batch)

    def latest(self) -> TrajectoryBatch:
        if not self.batches:
            raise ValueError("ReplayBuffer is empty")
        return self.batches[-1]
