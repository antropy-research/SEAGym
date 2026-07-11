from __future__ import annotations

"""Deterministic stratified split helpers for benchmark manifests."""

from collections import defaultdict
import hashlib
import random
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")


def stratified_subset(
    items: list[T],
    *,
    target_size: int,
    key: Callable[[T], Hashable],
    seed: int,
) -> list[T]:
    """Select a deterministic subset while preserving stratum proportions."""

    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if target_size > len(items):
        raise ValueError(f"target_size {target_size} exceeds item count {len(items)}")
    if target_size == len(items):
        return list(items)

    groups = _groups(items, key)
    counts = _allocate_counts(
        {stratum: len(group_items) for stratum, group_items in groups.items()},
        target_size,
    )
    selected: list[T] = []
    for stratum in sorted(groups, key=str):
        shuffled = list(groups[stratum])
        random.Random(_stratum_seed(seed, stratum)).shuffle(shuffled)
        selected.extend(shuffled[: counts[stratum]])
    return selected


def stratified_split(
    items: list[T],
    *,
    split_targets: dict[str, int],
    key: Callable[[T], Hashable],
    seed: int,
) -> dict[str, list[T]]:
    """Split items exactly by target sizes while balancing stratum proportions."""

    total_target = sum(split_targets.values())
    if total_target != len(items):
        raise ValueError(f"split target total {total_target} does not match item count {len(items)}")
    if any(count < 0 for count in split_targets.values()):
        raise ValueError("split targets must be non-negative")

    groups = _groups(items, key)
    stratum_sizes = {stratum: len(group_items) for stratum, group_items in groups.items()}
    allocations = _allocate_matrix(stratum_sizes, split_targets)
    result = {name: [] for name in split_targets}
    for stratum in sorted(groups, key=str):
        shuffled = list(groups[stratum])
        random.Random(_stratum_seed(seed, stratum)).shuffle(shuffled)
        offset = 0
        for split_name in split_targets:
            count = allocations[stratum][split_name]
            result[split_name].extend(shuffled[offset : offset + count])
            offset += count
    return result


def _groups(items: list[T], key: Callable[[T], Hashable]) -> dict[Hashable, list[T]]:
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def _allocate_counts(stratum_sizes: dict[Hashable, int], target_size: int) -> dict[Hashable, int]:
    total = sum(stratum_sizes.values())
    if total == 0:
        if target_size:
            raise ValueError("cannot allocate non-empty target from empty strata")
        return {stratum: 0 for stratum in stratum_sizes}
    raw = {
        stratum: (size * target_size / total)
        for stratum, size in stratum_sizes.items()
    }
    counts = {stratum: int(value) for stratum, value in raw.items()}
    remaining = target_size - sum(counts.values())
    order = sorted(
        stratum_sizes,
        key=lambda stratum: (raw[stratum] - counts[stratum], str(stratum)),
        reverse=True,
    )
    for stratum in order[:remaining]:
        counts[stratum] += 1
    return counts


def _allocate_matrix(
    stratum_sizes: dict[Hashable, int],
    split_targets: dict[str, int],
) -> dict[Hashable, dict[str, int]]:
    total = sum(stratum_sizes.values())
    if total == 0:
        return {stratum: {split: 0 for split in split_targets} for stratum in stratum_sizes}

    raw: dict[Hashable, dict[str, float]] = {}
    allocated: dict[Hashable, dict[str, int]] = {}
    row_remaining: dict[Hashable, int] = {}
    col_remaining = dict(split_targets)
    for stratum, size in stratum_sizes.items():
        raw[stratum] = {
            split: size * target / total
            for split, target in split_targets.items()
        }
        allocated[stratum] = {split: int(value) for split, value in raw[stratum].items()}
        row_remaining[stratum] = size - sum(allocated[stratum].values())
        for split, count in allocated[stratum].items():
            col_remaining[split] -= count

    cells = [
        (raw[stratum][split] - allocated[stratum][split], str(stratum), split, stratum)
        for stratum in stratum_sizes
        for split in split_targets
    ]
    cells.sort(reverse=True)
    while any(row_remaining.values()):
        progressed = False
        for _, _, split, stratum in cells:
            if row_remaining[stratum] <= 0 or col_remaining[split] <= 0:
                continue
            allocated[stratum][split] += 1
            row_remaining[stratum] -= 1
            col_remaining[split] -= 1
            progressed = True
            if not any(row_remaining.values()):
                break
        if not progressed:
            raise ValueError("could not allocate stratified split targets")

    if any(col_remaining.values()):
        raise ValueError(f"could not satisfy split target counts: {col_remaining}")
    return allocated


def _stratum_seed(seed: int, stratum: Hashable) -> int:
    digest = hashlib.sha256(f"{seed}:{repr(stratum)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
