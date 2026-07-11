from __future__ import annotations

from typing import Any


class ConstantMetric:
    name = "constant_metric"

    def compute(self, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        return {"value": 7, "num_records": len(records)}

