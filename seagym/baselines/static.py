from __future__ import annotations

"""No-update baseline lifecycle."""

from dataclasses import dataclass

from .base import BaseBaseline, BaselineState, UpdateResult
from .data import TrajectoryBatch


@dataclass
class StaticBaseline(BaseBaseline):
    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        del state
        self.update_index += 1
        return UpdateResult(
            update_index=self.update_index,
            changed=False,
            status="static",
            metrics={"num_trajectories": len(trajectories.trajectories)},
        )
