from __future__ import annotations

import json
from typing import Any


def _render_candidate(candidate: Any, *, component_key: str | None = None) -> str:
    if isinstance(candidate, str):
        return candidate
    if component_key is not None and isinstance(candidate, dict) and component_key in candidate:
        return str(candidate[component_key])
    return json.dumps(candidate, indent=2, sort_keys=True)


def _gepa_reflection_lm_cost(result: Any) -> dict[str, float]:
    candidates = [
        result,
        getattr(result, "reflection_lm", None),
        getattr(getattr(result, "config", None), "reflection_lm", None),
        getattr(getattr(getattr(result, "config", None), "reflection", None), "reflection_lm", None),
    ]
    for lm in candidates:
        tokens_in = getattr(lm, "total_tokens_in", None)
        tokens_out = getattr(lm, "total_tokens_out", None)
        if isinstance(tokens_in, int | float) or isinstance(tokens_out, int | float):
            input_tokens = float(tokens_in or 0.0)
            output_tokens = float(tokens_out or 0.0)
            total = input_tokens + output_tokens
            if total <= 0:
                continue
            cost = {"total_tokens": total}
            if input_tokens:
                cost["input_tokens"] = input_tokens
            if output_tokens:
                cost["output_tokens"] = output_tokens
            cost_usd = getattr(lm, "total_cost", None)
            if isinstance(cost_usd, int | float):
                cost["cost_usd"] = float(cost_usd)
            return cost
    return {}


def _best_score(result: Any) -> float | None:
    scores = getattr(result, "val_aggregate_scores", None)
    best_idx = getattr(result, "best_idx", None)
    if not isinstance(scores, list) or not isinstance(best_idx, int):
        return None
    if best_idx < 0 or best_idx >= len(scores):
        return None
    try:
        return float(scores[best_idx])
    except (TypeError, ValueError):
        return None
