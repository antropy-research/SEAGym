from __future__ import annotations

"""Task rollout agent interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from seagym.baselines.base import BaselineState
from seagym.baselines.data import TaskBatch, TrajectoryBatch
from seagym.data.types import TaskIndex
from seagym.envs import TaskEnv


@dataclass
class RolloutAgentState:
    metadata: dict[str, Any] = field(default_factory=dict)


class RolloutAgent(Protocol):
    agent_id: str

    def initialize(self, run_dir: Path) -> RolloutAgentState:
        ...

    def rollout(
        self,
        batch: TaskBatch,
        *,
        env: TaskEnv,
        task_index: TaskIndex,
        baseline_state: BaselineState,
    ) -> TrajectoryBatch:
        ...
