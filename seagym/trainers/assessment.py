from __future__ import annotations

"""Update-validation assessment helpers."""

from typing import Any

from seagym.envs import TaskRunResult


def assess_update_results(
    current_results: list[TaskRunResult],
    previous_results: list[TaskRunResult],
    metrics_config: dict[str, Any],
) -> dict[str, Any]:
    current = _mean([result.score for result in current_results])
    previous = _mean([result.score for result in previous_results])
    delta = current - previous
    cfg = metrics_config.get("update_assessment", {})
    min_improvement = float(cfg.get("min_improvement", 0.02))
    harm_tolerance = float(cfg.get("harm_tolerance", 0.02))
    if delta >= min_improvement:
        label = "beneficial"
    elif delta <= -harm_tolerance:
        label = "harmful"
    else:
        label = "neutral"
    return {
        "label": label,
        "delta_prev": delta,
        "threshold_ref": cfg.get("threshold_ref"),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
