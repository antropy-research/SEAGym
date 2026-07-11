from __future__ import annotations

"""Scan local Harbor-compatible task trees into SEAGym task indexes.

This is not a Harbor adapter and does not convert raw benchmarks into Harbor
format. It only reads existing Harbor `task.toml` files and records stable refs
needed by SEAGym.
"""

from pathlib import Path
import tomllib
from typing import Any

from seagym.data.scoring import ScoringRule
from seagym.data.types import TaskIndex, TaskRecord


def scan_harbor_task_tree(dataset_root: str | Path) -> TaskIndex:
    root = Path(dataset_root)
    task_files = sorted(root.rglob("task.toml"))
    if not task_files:
        raise ValueError(f"No Harbor task.toml files found under {root}")

    tasks = []
    for task_file in task_files:
        task_dir = task_file.parent
        task_data = tomllib.loads(task_file.read_text(encoding="utf-8"))
        task_section = task_data.get("task") or {}
        metadata = task_data.get("metadata") or {}
        registry_task_name = str(task_section.get("name") or task_dir.name)
        local_task_name = task_dir.name
        tasks.append(
            TaskRecord(
                task_id=registry_task_name,
                source={
                    "type": "harbor",
                    "dataset": root.name,
                    "dataset_path": str(root),
                    "dataset_version": str(task_data.get("version", "unknown")),
                    "task_name": local_task_name,
                    "registry_task_name": registry_task_name,
                    "local_path": str(task_dir),
                },
                attributes={
                    "domain": infer_domain_from_harbor_metadata(root.name, registry_task_name, metadata),
                    "task_type": str(metadata.get("category") or "unknown"),
                    "difficulty": str(metadata.get("difficulty") or "unknown"),
                    "source_benchmark": root.name,
                    "skills": _string_list(task_section.get("keywords")),
                    "tags": _string_list(metadata.get("tags")),
                },
                scoring=ScoringRule(),
                visibility={},
                fixtures={},
            )
        )

    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError(f"Duplicate Harbor task names found under {root}")
    return TaskIndex(
        version=f"harbor-dir:{root.name}",
        tasks=tasks_by_id,
        path=root,
    )


def infer_domain_from_harbor_metadata(dataset_name: str, task_name: str, metadata: dict[str, Any]) -> str:
    category = str(metadata.get("category") or "").lower()
    task_slug = task_name.rsplit("/", 1)[-1].lower()
    dataset = dataset_name.lower()
    if "swe-bench" in dataset or "debug" in category or "__" in task_name:
        return "code"
    if any(token in category for token in ("software-engineering", "programming", "coding")):
        return "code"
    if any(
        token in category
        for token in (
            "scientific-computing",
            "applied-statistics",
            "machine-learning",
            "simulation",
            "data",
            "optimization",
        )
    ):
        return "data_workflow"
    if any(token in task_slug for token in ("api", "server", "web", "terminal", "tool", "agent")):
        return "tool_use"
    return "code"


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
