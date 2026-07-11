from __future__ import annotations

"""Normalized task execution results."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    view_name: str
    mode: str
    rewards: dict[str, float]
    score: float
    success: bool
    cost: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float | None = None
    error: str | None = None
    refs: dict[str, Any] = field(default_factory=dict)
