from __future__ import annotations

"""Pure SEAGym data types.

This module intentionally avoids filesystem scanning and experiment loading.
Loaders live under `seagym.data` and `seagym.config`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .scoring import ScoringRule


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    source: dict[str, Any]
    attributes: dict[str, Any]
    scoring: ScoringRule
    visibility: dict[str, Any] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        task_id = data.get("task_id")
        if not task_id:
            raise ValueError("Task record missing task_id")
        source = data.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"Task {task_id} missing source object")
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError(f"Task {task_id} missing attributes object")
        return cls(
            task_id=str(task_id),
            source=source,
            attributes=attributes,
            scoring=ScoringRule.from_dict(data.get("scoring")),
            visibility=data.get("visibility") or {},
            fixtures=data.get("fixtures") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "attributes": self.attributes,
            "scoring": self.scoring.to_dict(),
            "visibility": self.visibility,
            "fixtures": self.fixtures,
        }


@dataclass(frozen=True)
class TaskIndex:
    version: str
    tasks: dict[str, TaskRecord]
    path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> "TaskIndex":
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("Task index must contain tasks list")
        tasks = {task.task_id: task for task in (TaskRecord.from_dict(item) for item in raw_tasks)}
        if len(tasks) != len(raw_tasks):
            raise ValueError("Task index contains duplicate task_id values")
        return cls(version=str(data.get("version", "unknown")), tasks=tasks, path=path)

    def require(self, task_id: str) -> TaskRecord:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task_id {task_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tasks": [task.to_dict() for task in self.tasks.values()],
        }


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    split_version: str
    seed: int
    train: list[str]
    val: list[str]
    test: list[str]
    path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> "SplitManifest":
        splits = data.get("splits")
        if not isinstance(splits, dict):
            raise ValueError("Split manifest missing splits object")
        unexpected = sorted(set(splits) - {"train", "val", "test"})
        if unexpected:
            raise ValueError(f"Split manifest contains unsupported split keys: {unexpected}")
        for key in ("train", "val", "test"):
            if not isinstance(splits.get(key), list):
                raise ValueError(f"Split manifest missing splits.{key} list")
        return cls(
            split_id=str(data.get("split_id", "unknown")),
            split_version=str(data.get("split_version", "unknown")),
            seed=int(data.get("seed", 0)),
            train=[str(x) for x in splits["train"]],
            val=[str(x) for x in splits["val"]],
            test=[str(x) for x in splits["test"]],
            path=path,
        )
