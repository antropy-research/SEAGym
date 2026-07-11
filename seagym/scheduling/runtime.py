from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random
from pathlib import Path
from typing import Any

from seagym.config import RuntimeSchedulingConfig
from seagym.envs import TaskRunResult
from seagym.utils import read_json, write_json


@dataclass(frozen=True)
class SchedulingDecision:
    decision_index: int
    mode: str
    policy: str
    workers: int
    original_task_ids: list[str]
    scheduled_task_ids: list[str]
    predictions: dict[str, float | None]
    cold_start: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "mode": self.mode,
            "policy": self.policy,
            "workers": self.workers,
            "original_task_ids": self.original_task_ids,
            "scheduled_task_ids": self.scheduled_task_ids,
            "predictions": self.predictions,
            "cold_start": self.cold_start,
        }


@dataclass
class RuntimeScheduler:
    config: RuntimeSchedulingConfig
    history_path: Path
    runtime_history: dict[str, list[float]] = field(default_factory=dict)
    decisions_completed: int = 0

    @classmethod
    def from_path(cls, config: RuntimeSchedulingConfig, history_path: str | Path) -> "RuntimeScheduler":
        path = Path(history_path)
        if not path.exists():
            return cls(config=config, history_path=path)
        data = read_json(path)
        raw_history = data.get("runtime_history") or {}
        if not isinstance(raw_history, dict):
            raise ValueError(f"Invalid scheduling history at {path}")
        history = {
            str(task_id): [float(value) for value in values]
            for task_id, values in raw_history.items()
            if isinstance(values, list)
        }
        return cls(
            config=config,
            history_path=path,
            runtime_history=history,
            decisions_completed=int(data.get("decisions_completed", 0)),
        )

    def plan(self, task_ids: list[str], *, mode: str, workers: int) -> SchedulingDecision:
        predictions = {task_id: self._prediction(task_id) for task_id in task_ids}
        cold_start = any(value is None for value in predictions.values())
        policy = self.config.policy
        scheduled = list(task_ids)

        if policy == "random":
            rng = random.Random(self.config.random_seed + self.decisions_completed)
            rng.shuffle(scheduled)
        elif policy == "lpt" and not cold_start:
            scheduled = sorted(task_ids, key=lambda task_id: (-float(predictions[task_id]), task_id))

        return SchedulingDecision(
            decision_index=self.decisions_completed + 1,
            mode=mode,
            policy=policy,
            workers=max(1, workers),
            original_task_ids=list(task_ids),
            scheduled_task_ids=scheduled,
            predictions=predictions,
            cold_start=cold_start,
        )

    def observe(
        self,
        decision: SchedulingDecision,
        results: list[TaskRunResult],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        actual_by_task: dict[str, list[float]] = {}
        runtime_rows: list[dict[str, Any]] = []
        for result in results:
            runtime = result.runtime_seconds
            if runtime is None or runtime < 0:
                continue
            value = float(runtime)
            self.runtime_history.setdefault(result.task_id, []).append(value)
            actual_by_task.setdefault(result.task_id, []).append(value)
            runtime_rows.append(
                {
                    "decision_index": decision.decision_index,
                    "task_id": result.task_id,
                    "runtime_seconds": value,
                    "success": result.success,
                    "error": result.error,
                    "trial_name": result.refs.get("trial_name"),
                    "result_path": result.refs.get("result_path"),
                    "runtime_source": result.refs.get("runtime_source"),
                    "started_at": result.refs.get("started_at"),
                    "finished_at": result.refs.get("finished_at"),
                }
            )

        self.decisions_completed = decision.decision_index
        self._write_history()
        diagnostics = self._diagnostics(decision, actual_by_task)
        return runtime_rows, diagnostics

    def summary(self) -> dict[str, Any]:
        values = [value for history in self.runtime_history.values() for value in history]
        return {
            "policy": self.config.policy,
            "ema_k": self.config.ema_k,
            "cold_start": self.config.cold_start,
            "decisions_completed": self.decisions_completed,
            "tasks_with_history": len(self.runtime_history),
            "runtime_observations": len(values),
            "mean_runtime_seconds": None if not values else sum(values) / len(values),
        }

    def _prediction(self, task_id: str) -> float | None:
        values = self.runtime_history.get(task_id, [])
        if not values:
            return None
        alpha = 2.0 / (self.config.ema_k + 1.0)
        estimate = values[0]
        for value in values[1:]:
            estimate = alpha * value + (1.0 - alpha) * estimate
        return estimate

    def _write_history(self) -> None:
        write_json(
            self.history_path,
            {
                "version": 1,
                "decisions_completed": self.decisions_completed,
                "runtime_history": self.runtime_history,
            },
        )

    def _diagnostics(
        self,
        decision: SchedulingDecision,
        actual_by_task: dict[str, list[float]],
    ) -> dict[str, Any]:
        actual_times: dict[str, float] = {}
        for task_id in decision.scheduled_task_ids:
            values = actual_by_task.get(task_id)
            if not values:
                return {"available": False, "reason": "missing_runtime"}
            actual_times[task_id] = values[0]
        actual_makespan = _list_makespan(decision.scheduled_task_ids, actual_times, decision.workers)
        hindsight_order = sorted(actual_times, key=lambda task_id: (-actual_times[task_id], task_id))
        hindsight_lpt = _list_makespan(hindsight_order, actual_times, decision.workers)
        total = sum(actual_times.values())
        return {
            "available": True,
            "actual_makespan_seconds": actual_makespan,
            "lower_bound_seconds": max(max(actual_times.values()), total / decision.workers),
            "hindsight_lpt_seconds": hindsight_lpt,
        }


def _list_makespan(order: list[str], times: dict[str, float], workers: int) -> float:
    loads = [0.0] * max(1, workers)
    heapq.heapify(loads)
    for task_id in order:
        load = heapq.heappop(loads)
        heapq.heappush(loads, load + times[task_id])
    return max(loads)
