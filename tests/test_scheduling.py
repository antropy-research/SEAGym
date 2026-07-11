from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from seagym.config import RuntimeSchedulingConfig
from seagym.envs import TaskRunResult
from seagym.scheduling import RuntimeScheduler


class RuntimeSchedulerTest(unittest.TestCase):
    def test_lpt_waits_for_history_then_orders_by_ema_prediction(self) -> None:
        config = RuntimeSchedulingConfig(enabled=True, policy="lpt", ema_k=5.0, random_seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "scheduling_history.json"
            scheduler = RuntimeScheduler.from_path(config, history_path)

            first = scheduler.plan(["short", "long"], mode="train", workers=2)
            self.assertTrue(first.cold_start)
            self.assertEqual(first.scheduled_task_ids, ["short", "long"])
            runtime_rows, _ = scheduler.observe(
                first,
                [
                    _result("short", 2.0, finished_at="2026-07-11T00:00:02Z"),
                    _result("long", 10.0),
                ],
            )
            self.assertEqual(runtime_rows[0]["runtime_source"], "trial_elapsed")
            self.assertEqual(runtime_rows[0]["finished_at"], "2026-07-11T00:00:02Z")

            second = scheduler.plan(["short", "long"], mode="train", workers=2)
            self.assertFalse(second.cold_start)
            self.assertEqual(second.scheduled_task_ids, ["long", "short"])
            self.assertEqual(second.predictions, {"short": 2.0, "long": 10.0})

            restored = RuntimeScheduler.from_path(config, history_path)
            self.assertEqual(restored.runtime_history, {"short": [2.0], "long": [10.0]})
            self.assertEqual(restored.decisions_completed, 1)

    def test_random_policy_is_deterministic_per_decision_index(self) -> None:
        config = RuntimeSchedulingConfig(enabled=True, policy="random", random_seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            first = RuntimeScheduler.from_path(config, Path(tmp) / "first.json")
            second = RuntimeScheduler.from_path(config, Path(tmp) / "second.json")
            self.assertEqual(
                first.plan(["a", "b", "c", "d"], mode="train", workers=2).scheduled_task_ids,
                second.plan(["a", "b", "c", "d"], mode="train", workers=2).scheduled_task_ids,
            )


def _result(task_id: str, runtime_seconds: float, *, finished_at: str | None = None) -> TaskRunResult:
    return TaskRunResult(
        task_id=task_id,
        view_name="train",
        mode="train",
        rewards={"reward": 1.0},
        score=1.0,
        success=True,
        runtime_seconds=runtime_seconds,
        refs={"runtime_source": "trial_elapsed", "finished_at": finished_at} if finished_at else {},
    )
