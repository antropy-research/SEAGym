from __future__ import annotations

"""Metric registry and reference metric implementations.

Metrics operate on saved normalized records, not live Harbor jobs. This keeps
SEAGym reports reproducible and lets users add or recompute metrics offline.

Inputs:
- `records`: rows from `records/metric_inputs.jsonl`.
- `config`: the experiment metrics config.

Outputs:
- JSON-serializable metric dictionaries.

BDD expectations:
- Given records from multiple views, builtin metrics compute per-view values.
- Given domain attributes, domain macro success averages success rates across
  domains within each view.
- Given a Python `import_path`, the registry can load a user-defined metric.

Future work:
- Add reference metrics for Adaptation Gain, Transfer Gain, cost-normalized
  gain, and diagnostics.
- Add declarative metric table/report specs after the Python plugin path is
  stable.
"""

from dataclasses import dataclass
import importlib
from typing import Any, Protocol

from .functional import mean, success_rate

class Metric(Protocol):
    name: str

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SuccessRateMetric:
    name: str = "success_rate"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        by_view: dict[str, list[dict[str, Any]]] = {}
        for row in _task_metric_records(records):
            by_view.setdefault(_view_key(row), []).append(row)
        return {
            view: success_rate(rows)
            for view, rows in sorted(by_view.items())
        }


@dataclass(frozen=True)
class MeanScoreMetric:
    name: str = "mean_score"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        by_view: dict[str, list[dict[str, Any]]] = {}
        for row in _task_metric_records(records):
            by_view.setdefault(_view_key(row), []).append(row)
        return {
            view: mean([float(row["score"]) for row in rows])
            for view, rows in sorted(by_view.items())
        }


@dataclass(frozen=True)
class DomainMacroSuccessMetric:
    name: str = "domain_macro_success_rate"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in _task_metric_records(records):
            view = _view_key(row)
            domain = str((row.get("attributes") or {}).get("domain", "unknown"))
            grouped.setdefault(view, {}).setdefault(domain, []).append(row)
        result: dict[str, float] = {}
        for view, domains in sorted(grouped.items()):
            result[view] = mean([success_rate(rows) for rows in domains.values()])
        return result


@dataclass(frozen=True)
class UpdateValidationGainMetric:
    name: str = "update_validation_gain"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        scores = _mean_score_by_evaluation_point(records, view_name="update_validation")
        if not scores:
            return {"prev": {}, "base": {}}
        ordered = sorted(scores, key=_evaluation_point_sort_key)
        base_id = "E_0" if "E_0" in scores else ordered[0]
        prev: dict[str, float] = {}
        base: dict[str, float] = {}
        previous_id = base_id
        for point_id in ordered:
            if point_id == base_id:
                continue
            prev[point_id] = scores[point_id] - scores[previous_id]
            base[point_id] = scores[point_id] - scores[base_id]
            previous_id = point_id
        return {"prev": prev, "base": base}


@dataclass(frozen=True)
class ValidationSupportedUpdateRateMetric:
    name: str = "validation_supported_update_rate"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        gains = UpdateValidationGainMetric().compute(records, config)["prev"]
        threshold = float((config.get("update_assessment") or {}).get("min_improvement", 0.02))
        if not gains:
            return {"value": 0.0, "num_updates": 0, "threshold": threshold}
        supported = sum(1 for value in gains.values() if value >= threshold)
        return {
            "value": supported / len(gains),
            "num_updates": len(gains),
            "threshold": threshold,
        }


@dataclass(frozen=True)
class FinalGainMetric:
    name: str = "final_gain"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        scores = _mean_score_by_view_and_role(records)
        result: dict[str, float] = {}
        for view, by_role in sorted(scores.items()):
            if "A_T" in by_role and "A_0" in by_role:
                result[view] = by_role["A_T"] - by_role["A_0"]
        return result


@dataclass(frozen=True)
class ForgettingRateMetric:
    name: str = "forgetting_rate"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        scores = _mean_score_by_view_and_role(records)
        result: dict[str, float] = {}
        for view, by_role in sorted(scores.items()):
            if "A_T" in by_role and "A_0" in by_role:
                result[view] = max(0.0, by_role["A_0"] - by_role["A_T"])
        return result


@dataclass(frozen=True)
class TokenUsageMetric:
    name: str = "tokens"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        rollout_rows = [row for row in records if row.get("mode") != "update"]
        update_rows = [row for row in records if row.get("mode") == "update"]
        return {
            "rollout": _token_stats(rollout_rows),
            "update": _token_stats(update_rows),
            "overall": _token_stats(records),
            "by_view": {
                view: _token_stats(rows)
                for view, rows in sorted(_group_by_view(records).items())
            },
        }


@dataclass(frozen=True)
class CostFieldMetric:
    name: str
    field: str

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        rollout_rows = [row for row in records if row.get("mode") != "update"]
        update_rows = [row for row in records if row.get("mode") == "update"]
        return {
            "rollout": _cost_field_stats(rollout_rows, self.field),
            "update": _cost_field_stats(update_rows, self.field),
            "overall": _cost_field_stats(records, self.field),
            "by_view": {
                view: _cost_field_stats(rows, self.field)
                for view, rows in sorted(_group_by_view(records).items())
            },
        }


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric, *, name: str | None = None) -> None:
        self._metrics[name or metric.name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise KeyError(f"Unknown metric {name}") from exc

    def compute(
        self,
        records: list[dict[str, Any]],
        config: dict[str, Any],
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        names = metric_names or list(self._metrics)
        return {
            name: self.get(name).compute(records, config)
            for name in names
        }

    @classmethod
    def from_config(cls, metrics_config: dict[str, Any]) -> "MetricRegistry":
        registry = default_metric_registry()
        for item in metrics_config.get("registry", []):
            metric_name = item.get("name")
            if item.get("type") == "python":
                metric = _load_metric(str(item["import_path"]))
                registry.register(metric, name=str(metric_name) if metric_name else None)
            elif item.get("type") == "builtin":
                metric = registry.get(str(metric_name)) if metric_name else None
                if metric is None:
                    raise ValueError("builtin metric registry item requires name")
                registry.register(metric, name=str(metric_name))
        return registry


def default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(SuccessRateMetric())
    registry.register(MeanScoreMetric())
    registry.register(DomainMacroSuccessMetric())
    registry.register(UpdateValidationGainMetric())
    registry.register(ValidationSupportedUpdateRateMetric())
    registry.register(FinalGainMetric())
    registry.register(ForgettingRateMetric())
    token_metric = TokenUsageMetric()
    registry.register(token_metric)
    registry.register(token_metric, name="token_usage")
    registry.register(CostFieldMetric(name="tool_calls", field="tool_calls"))
    registry.register(CostFieldMetric(name="wall_time", field="wall_time"))
    registry.register(CostFieldMetric(name="cost_usd", field="cost_usd"))
    return registry


def _load_metric(import_path: str) -> Metric:
    module_name, _, attr = import_path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Invalid metric import_path: {import_path}")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    return obj() if isinstance(obj, type) else obj


def _view_key(row: dict[str, Any]) -> str:
    view = str(row["view_name"])
    role = row.get("baseline_role")
    if role:
        return f"{view}.{role}"
    return view


def _task_metric_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row.get("mode") != "update" and isinstance(row.get("score"), int | float)
    ]


def _mean_score_by_evaluation_point(records: list[dict[str, Any]], *, view_name: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in records:
        if row.get("view_name") != view_name:
            continue
        point_id = row.get("evaluation_point_id")
        if not point_id:
            continue
        grouped.setdefault(str(point_id), []).append(float(row.get("score", 0.0)))
    return {point_id: mean(scores) for point_id, scores in grouped.items()}


def _mean_score_by_view_and_role(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in records:
        role = row.get("baseline_role")
        if role not in ("A_0", "A_T"):
            continue
        view = str(row["view_name"])
        grouped.setdefault(view, {}).setdefault(str(role), []).append(float(row.get("score", 0.0)))
    return {
        view: {role: mean(scores) for role, scores in by_role.items()}
        for view, by_role in grouped.items()
    }


def _evaluation_point_sort_key(point_id: str) -> tuple[int, str]:
    if point_id == "E_0":
        return (0, point_id)
    if point_id == "E_T":
        return (10**9, point_id)
    prefix, _, suffix = point_id.partition("_")
    if prefix == "E" and suffix.isdigit():
        return (int(suffix), point_id)
    return (10**8, point_id)


def _group_by_view(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("view_name", "unknown")), []).append(row)
    return grouped


def _token_stats(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    totals = {
        "total_tokens": 0.0,
        "input_tokens": 0.0,
        "cache_tokens": 0.0,
        "output_tokens": 0.0,
        "cost_usd": 0.0,
    }
    observed = 0
    for row in records:
        cost = row.get("cost") or {}
        if not isinstance(cost, dict):
            continue
        total = _token_total(cost)
        if total is None:
            continue
        observed += 1
        totals["total_tokens"] += total
        totals["input_tokens"] += _cost_number(cost, "input_tokens", "n_input_tokens")
        totals["cache_tokens"] += _cost_number(cost, "cache_tokens", "n_cache_tokens")
        totals["output_tokens"] += _cost_number(cost, "output_tokens", "n_output_tokens")
        totals["cost_usd"] += _cost_number(cost, "cost_usd")
    return {
        "num_records": len(records),
        "num_records_with_tokens": observed,
        **totals,
        "mean_tokens": None if observed == 0 else totals["total_tokens"] / observed,
        "mean_cost_usd": None if observed == 0 else totals["cost_usd"] / observed,
    }


def _cost_field_stats(records: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for row in records:
        cost = row.get("cost") or {}
        if isinstance(cost, dict) and isinstance(cost.get(field), int | float):
            values.append(float(cost[field]))
    return {
        "num_records": len(records),
        "num_records_with_value": len(values),
        "total": sum(values),
        "mean": None if not values else mean(values),
    }


def _token_total(cost: dict[str, Any]) -> float | None:
    for key in ("tokens", "total_tokens", "n_total_tokens"):
        if isinstance(cost.get(key), int | float):
            return float(cost[key])
    parts = [
        _cost_number(cost, "input_tokens", "n_input_tokens"),
        _cost_number(cost, "cache_tokens", "n_cache_tokens"),
        _cost_number(cost, "output_tokens", "n_output_tokens"),
    ]
    if any(part != 0.0 for part in parts):
        return sum(parts)
    return None


def _cost_number(cost: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if isinstance(cost.get(key), int | float):
            return float(cost[key])
    return 0.0
