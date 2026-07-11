from __future__ import annotations

"""Task index loading.

This module decides whether a path is a saved SEAGym task index JSON file or
a local Harbor-compatible task tree. Harbor task scanning itself lives in
`seagym.data.harbor_scan`.
"""

from pathlib import Path

from seagym.data.types import TaskIndex
from seagym.paths import has_portable_anchor, resolve_portable_path
from seagym.utils import read_json


def load_task_index(path: str | Path) -> TaskIndex:
    source = Path(path)
    if source.is_dir():
        from .harbor_scan import scan_harbor_task_tree

        return scan_harbor_task_tree(source)
    return _load_task_index_json(source)


def _load_task_index_json(path: str | Path) -> TaskIndex:
    source = Path(path).resolve()
    return TaskIndex.from_dict(_normalize_task_index_paths(read_json(source), base_dir=source.parent), path=source)


def _normalize_task_index_paths(data: dict, *, base_dir: Path) -> dict:
    normalized = dict(data)
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return normalized
    normalized_tasks = []
    for item in tasks:
        if not isinstance(item, dict):
            normalized_tasks.append(item)
            continue
        task = dict(item)
        source = task.get("source")
        if isinstance(source, dict):
            task["source"] = _normalize_source_paths(source, base_dir=base_dir)
        normalized_tasks.append(task)
    normalized["tasks"] = normalized_tasks
    return normalized


def _normalize_source_paths(source: dict, *, base_dir: Path) -> dict:
    normalized = dict(source)
    for key in ("dataset_path", "local_path"):
        value = normalized.get(key)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if has_portable_anchor(value) or not path.is_absolute():
            normalized[key] = str(resolve_portable_path(value, base_dir=base_dir))
    return normalized
