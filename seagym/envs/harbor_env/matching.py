from __future__ import annotations

from pathlib import Path
from typing import Any

from seagym.data.types import TaskRecord


def find_trial_results(job_dir: Path) -> list[Path]:
    if not job_dir.exists():
        return []
    return sorted(path for path in job_dir.glob("*/result.json") if path.parent != job_dir)


def task_lookup(tasks: list[TaskRecord]) -> dict[str, TaskRecord]:
    lookup: dict[str, TaskRecord] = {}
    for task in tasks:
        for key in task_match_keys(task):
            lookup[key] = task
    return lookup


def task_match_keys(task: TaskRecord) -> set[str]:
    keys = {task.task_id}
    for source_key in ("task_name", "registry_task_name", "local_path"):
        value = task.source.get(source_key)
        if value:
            text = str(value)
            keys.add(text)
            keys.add(Path(text).name)
    return keys


def match_task_for_trial(
    data: dict[str, Any],
    result_path: Path,
    task_by_key: dict[str, TaskRecord],
) -> TaskRecord | None:
    candidate_values = [
        data.get("task_name"),
        data.get("source"),
        data.get("trial_name"),
        result_path.parent.name,
    ]
    config = data.get("config") or {}
    if isinstance(config, dict):
        candidate_values.extend([config.get("task_name"), config.get("task_id")])
    for value in candidate_values:
        if not value:
            continue
        text = str(value)
        if text in task_by_key:
            return task_by_key[text]
        slug = text.rsplit("/", 1)[-1]
        if slug in task_by_key:
            return task_by_key[slug]
        prefix = text.split("__", 1)[0]
        if prefix in task_by_key:
            return task_by_key[prefix]
    return None
