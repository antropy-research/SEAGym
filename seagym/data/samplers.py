from __future__ import annotations

"""Sampling helpers for materialized train batches and evaluation views."""

import random
from typing import Any


def select_subset(candidates: list[str], sample_size: Any, seed: int) -> list[str]:
    if sample_size is None:
        return list(candidates)
    n = int(sample_size)
    if n >= len(candidates):
        return list(candidates)
    selected = list(candidates)
    random.Random(seed).shuffle(selected)
    return selected[:n]


def shuffled(items: list[str], seed: int) -> list[str]:
    selected = list(items)
    random.Random(seed).shuffle(selected)
    return selected
