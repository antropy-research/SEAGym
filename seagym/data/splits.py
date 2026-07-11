from __future__ import annotations

"""Split manifest loading and validation."""

from pathlib import Path

from seagym.data.types import SplitManifest, TaskIndex
from seagym.utils import read_json


def load_split_manifest(path: str | Path) -> SplitManifest:
    source = Path(path)
    return SplitManifest.from_dict(read_json(source), path=source)


def validate_split_manifest(task_index: TaskIndex, split: SplitManifest) -> None:
    _validate_split_task_ids(task_index, split)
    _validate_no_duplicates(split)
    _validate_disjoint_splits(split)


def _validate_split_task_ids(task_index: TaskIndex, split: SplitManifest) -> None:
    missing = [
        task_id
        for task_id in [*split.train, *split.val, *split.test]
        if task_id not in task_index.tasks
    ]
    if missing:
        raise ValueError(f"Split references unknown task ids: {missing}")


def _validate_no_duplicates(split: SplitManifest) -> None:
    for name, task_ids in (("train", split.train), ("val", split.val), ("test", split.test)):
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"Split {name} contains duplicate task ids: {duplicates}")


def _validate_disjoint_splits(split: SplitManifest) -> None:
    split_sets = {
        "train": set(split.train),
        "val": set(split.val),
        "test": set(split.test),
    }
    overlaps: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            overlaps.append(f"{left}/{right}: {overlap}")
    if overlaps:
        raise ValueError(f"Split train/val/test sets must be disjoint; overlaps: {overlaps}")
