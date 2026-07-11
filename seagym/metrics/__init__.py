from __future__ import annotations

"""Metric registry and reference metrics."""

from .functional import mean, success_rate
from .registry import Metric, MetricRegistry, default_metric_registry

__all__ = ["Metric", "MetricRegistry", "default_metric_registry", "mean", "success_rate"]
