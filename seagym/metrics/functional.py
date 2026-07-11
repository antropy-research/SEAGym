from __future__ import annotations

"""Functional metric helpers for normalized SEAGym records."""

from typing import Any


def success_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("success")) / len(rows)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
