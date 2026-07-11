from __future__ import annotations

"""Scoring rules shared by task records and execution normalization."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoringRule:
    main_reward_key: str = "reward"
    success_threshold: float = 1.0
    score_transform: str = "identity"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScoringRule":
        data = data or {}
        return cls(
            main_reward_key=str(data.get("main_reward_key", "reward")),
            success_threshold=float(data.get("success_threshold", 1.0)),
            score_transform=str(data.get("score_transform", "identity")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_reward_key": self.main_reward_key,
            "success_threshold": self.success_threshold,
            "score_transform": self.score_transform,
        }


def score_from_reward(reward: float, scoring: ScoringRule) -> float:
    if scoring.score_transform == "binary_threshold":
        return 1.0 if reward >= scoring.success_threshold else 0.0
    return reward
