from __future__ import annotations

"""Environment protocol for SEAGym task rollout."""

from typing import Protocol

from seagym.data.types import TaskRecord
from .results import TaskRunResult


class TaskEnv(Protocol):
    """Batch-first execution protocol.

    A single task is represented as a batch of size one. Environments expose
    only `run_tasks()` so engine code has one execution path.
    """

    def run_tasks(
        self,
        tasks: list[TaskRecord],
        *,
        view_name: str,
        mode: str,
        agent_id: str,
    ) -> list[TaskRunResult]:
        ...
