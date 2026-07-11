from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from seagym import SEAGymDataModule, DeterministicEnv
from seagym.baselines import StaticBaseline, TrajectoryBatch, UpdateResult
from seagym.config import RuntimeSchedulingConfig, load_experiment_context
from seagym.envs import TaskRunResult
from seagym.rollout_agents.harbor import HarborRolloutAgent
from seagym.trainers import ExecutionEngine, TrainerState
from seagym.utils import read_jsonl


class ExecutionEngineTest(unittest.TestCase):
    def test_engine_writes_metric_inputs(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(patched, batch_plan, DeterministicEnv(), agent_id="test")
            engine.layout.prepare(overwrite=True)
            engine.write_run_inputs()
            point = engine.record_evaluation_point(
                point_type="initial",
                train_batch_index=0,
                num_train_tasks_seen=0,
            )
            results = engine.run_tasks(
                ["code_val_001"],
                view_name="update_validation",
                mode="validation",
                evaluation_point=point,
            )
            engine.write_evaluation_point(
                point,
                evaluations={
                    "update_validation": {
                        "view_ref": "V_update-val",
                        "subset_id": "code_val_001",
                        "score": results[0].score,
                        "num_tasks": 1,
                    }
                },
                refs={"metric_inputs": str(engine.metric_input_path)},
            )

            rows = read_jsonl(engine.metric_input_path)
            point_rows = read_jsonl(engine.evaluation_points_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_id"], "code_val_001")
            self.assertEqual(rows[0]["view_name"], "update_validation")
            self.assertIn("rewards", rows[0])
            self.assertEqual(rows[0]["runtime_seconds"], 0.0)
            self.assertEqual(point_rows[0]["evaluations"]["update_validation"]["score"], 1.0)
            self.assertEqual(point_rows[0]["refs"]["metric_inputs"], str(engine.metric_input_path))

    def test_engine_uses_env_batch_api(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            env = _BatchEnv()
            engine = _engine(patched, batch_plan, env, agent_id="test")
            engine.layout.prepare(overwrite=True)
            engine.write_run_inputs()

            results = engine.run_tasks(
                ["code_val_001", "tool_val_001"],
                view_name="update_validation",
                mode="validation",
            )

            rows = read_jsonl(engine.metric_input_path)
            self.assertEqual(env.batch_calls, 1)
            self.assertEqual([result.task_id for result in results], ["code_val_001", "tool_val_001"])
            self.assertEqual([row["task_id"] for row in rows], ["code_val_001", "tool_val_001"])

    def test_engine_schedules_lpt_after_runtime_warmup(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        raw = dict(ctx.config.raw)
        raw["runtime_scheduling"] = {
            "enabled": True,
            "apply_to": ["train"],
            "policy": "lpt",
            "estimator": {"kind": "ema", "k": 5, "cold_start": "none"},
        }
        config = replace(
            ctx.config,
            raw=raw,
            runtime_scheduling=RuntimeSchedulingConfig.from_dict(raw["runtime_scheduling"], default_seed=ctx.config.seed),
        )
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(replace(ctx, config=config), Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            env = _TimedEnv({"code_001": 1.0, "tool_001": 10.0})
            engine = _engine(patched, batch_plan, env, agent_id="test")
            engine.layout.prepare(overwrite=True)

            task_ids = ["code_001", "tool_001"]
            first = engine.run_tasks(task_ids, view_name="train", mode="train")
            second = engine.run_tasks(task_ids, view_name="train", mode="train")

            decisions = read_jsonl(engine.layout.scheduling_decisions_path)
            self.assertEqual(env.task_orders, [task_ids, ["tool_001", "code_001"]])
            self.assertEqual([result.task_id for result in first], task_ids)
            self.assertEqual([result.task_id for result in second], task_ids)
            self.assertTrue(decisions[0]["cold_start"])
            self.assertFalse(decisions[1]["cold_start"])
            self.assertEqual(decisions[1]["scheduled_task_ids"], ["tool_001", "code_001"])
            self.assertTrue(engine.layout.scheduling_history_path.exists())

    def test_engine_records_multiple_attempt_results_for_one_task(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            result_0 = TaskRunResult(
                task_id="code_001",
                view_name="train",
                mode="train",
                rewards={"reward": 0.0},
                score=0.0,
                success=False,
                refs={"attempt_index": 0},
            )
            result_1 = TaskRunResult(
                task_id="code_001",
                view_name="train",
                mode="train",
                rewards={"reward": 1.0},
                score=1.0,
                success=True,
                refs={"attempt_index": 1},
            )
            engine = _engine(
                patched,
                batch_plan,
                _FixedResultsEnv([result_0, result_1]),
                agent_id="test",
            )
            engine.layout.prepare(overwrite=True)

            results = engine.run_tasks(["code_001"], view_name="train", mode="train")

            rows = read_jsonl(engine.task_log_path)
            self.assertEqual(len(results), 2)
            self.assertEqual([row["task_id"] for row in rows], ["code_001", "code_001"])
            self.assertEqual([row["refs"]["attempt_index"] for row in rows], [0, 1])

    def test_engine_calls_baseline_update_and_checkpoint(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(patched, batch_plan, DeterministicEnv(), agent_id="static")
            engine.layout.prepare(overwrite=True)
            engine.write_run_inputs()

            results = engine.run_tasks(["code_001"], view_name="train", mode="train")
            update = engine.update_agent(
                results,
                train_batch_index=1,
                num_train_tasks_seen=1,
            )
            checkpoint = engine.save_checkpoint(
                "E_1",
                trainer_state=TrainerState(
                    epoch=0,
                    train_batch_index=1,
                    global_step=1,
                    updates_completed=1,
                    num_train_tasks_seen=1,
                    checkpoint_id="E_1",
                ),
            )

            update_rows = read_jsonl(engine.agent_updates_path)
            metric_rows = read_jsonl(engine.metric_input_path)
            update_metric_rows = [row for row in metric_rows if row["mode"] == "update"]
            self.assertEqual(update["summary"]["type"], "baseline_update")
            self.assertFalse(update["summary"]["changed"])
            self.assertEqual(update_rows[0]["summary"]["num_records"], 1)
            self.assertEqual(update_metric_rows[0]["view_name"], "agent_update")
            self.assertEqual(checkpoint["checkpoint_id"], "E_1")
            self.assertTrue((engine.layout.checkpoints_dir / "E_1" / "checkpoint.json").exists())

    def test_should_validate_uses_epoch_boundary(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(patched, batch_plan, DeterministicEnv(), agent_id="static")

            self.assertFalse(engine.should_validate(1))
            self.assertTrue(engine.should_validate(2))

    def test_engine_loads_checkpoint_by_id_or_path(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(patched, batch_plan, DeterministicEnv(), agent_id="static")
            engine.layout.prepare(overwrite=True)
            engine.baseline_state = engine.baseline.initialize(engine.layout.run_dir)
            engine.save_checkpoint(
                "E_1",
                trainer_state=TrainerState(
                    epoch=0,
                    train_batch_index=0,
                    global_step=0,
                    updates_completed=0,
                    num_train_tasks_seen=0,
                    checkpoint_id="E_1",
                ),
            )

            by_id = engine.load_checkpoint("E_1")
            by_path = engine.load_checkpoint(engine.layout.checkpoints_dir / "E_1")

            self.assertTrue(by_id["loaded"])
            self.assertTrue(by_path["loaded"])
            self.assertEqual(by_path["checkpoint_dir"], str(engine.layout.checkpoints_dir / "E_1"))

    def test_engine_records_baseline_update_cost_for_metrics(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(patched, batch_plan, DeterministicEnv(), agent_id="cost-baseline", baseline=_CostBaseline())
            engine.layout.prepare(overwrite=True)

            results = engine.run_tasks(["code_001"], view_name="train", mode="train")
            engine.update_agent(results, train_batch_index=1, num_train_tasks_seen=1)

            update_rows = [row for row in read_jsonl(engine.metric_input_path) if row["mode"] == "update"]
            self.assertEqual(update_rows[0]["cost"]["input_tokens"], 11.0)
            self.assertEqual(update_rows[0]["cost"]["output_tokens"], 7.0)
            self.assertEqual(update_rows[0]["cost"]["cost_usd"], 0.4)

    def test_engine_records_baseline_update_cost_from_logs_for_metrics(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            engine = _engine(
                patched,
                batch_plan,
                DeterministicEnv(),
                agent_id="cost-baseline",
                baseline=_LogCostBaseline(),
            )
            engine.layout.prepare(overwrite=True)

            results = engine.run_tasks(["code_001"], view_name="train", mode="train")
            engine.update_agent(results, train_batch_index=1, num_train_tasks_seen=1)

            update_rows = [row for row in read_jsonl(engine.metric_input_path) if row["mode"] == "update"]
            self.assertEqual(update_rows[0]["cost"]["total_tokens"], 42.0)

    def test_engine_redacts_sensitive_refs(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            patched = _patch_run_dir(ctx, Path(tmp))
            batch_plan = SEAGymDataModule(patched).build()
            result = TaskRunResult(
                task_id="code_001",
                view_name="train",
                mode="train",
                rewards={"reward": 1.0},
                score=1.0,
                success=True,
                refs={
                    "command": [
                        "--agent-env",
                        "LLM_API_KEY=secret",
                        "--agent-env",
                        "HTTP_PROXY=http://user:pass@127.0.0.1:7890",
                    ]
                },
            )
            env = _FixedResultEnv(result)
            baseline = _CaptureBaseline()
            engine = _engine(patched, batch_plan, env, agent_id="test", baseline=baseline)
            engine.layout.prepare(overwrite=True)

            results = engine.run_tasks(["code_001"], view_name="train", mode="train")
            engine.update_agent(results, train_batch_index=1, num_train_tasks_seen=1)

            row = read_jsonl(engine.task_log_path)[0]
            self.assertIn("LLM_API_KEY=<redacted>", row["refs"]["command"])
            self.assertIn("HTTP_PROXY=http://***@127.0.0.1:7890", row["refs"]["command"])
            self.assertIn("LLM_API_KEY=<redacted>", str(baseline.trajectories.to_dict()))


def _patch_run_dir(ctx, run_dir: Path):
    from dataclasses import replace

    return replace(ctx, config=replace(ctx.config, run_dir=run_dir))


def _engine(ctx, batch_plan, env, *, agent_id: str, baseline: StaticBaseline | None = None) -> ExecutionEngine:
    baseline = baseline or StaticBaseline(
        baseline_id=agent_id,
        state_dir=ctx.config.run_dir / "agent_state" / agent_id,
    )
    state = baseline.initialize(ctx.config.run_dir)
    rollout_agent = HarborRolloutAgent(agent_id=agent_id)
    return ExecutionEngine(
        ctx,
        batch_plan,
        env,
        agent_id=agent_id,
        baseline=baseline,
        baseline_state=state,
        rollout_agent=rollout_agent,
    )


class _BatchEnv:
    def __init__(self) -> None:
        self.batch_calls = 0

    def run_tasks(self, tasks, *, view_name: str, mode: str, agent_id: str):
        self.batch_calls += 1
        return [
            TaskRunResult(
                task_id=task.task_id,
                view_name=view_name,
                mode=mode,
                rewards={"reward": 1.0},
                score=1.0,
                success=True,
                refs={"env": "batch-test", "agent_id": agent_id},
            )
            for task in tasks
        ]


class _TimedEnv:
    def __init__(self, runtimes: dict[str, float]) -> None:
        self.runtimes = runtimes
        self.task_orders: list[list[str]] = []

    def run_tasks(self, tasks, *, view_name: str, mode: str, agent_id: str):
        self.task_orders.append([task.task_id for task in tasks])
        return [
            TaskRunResult(
                task_id=task.task_id,
                view_name=view_name,
                mode=mode,
                rewards={"reward": 1.0},
                score=1.0,
                success=True,
                runtime_seconds=self.runtimes[task.task_id],
            )
            for task in tasks
        ]


class _FixedResultEnv:
    def __init__(self, result: TaskRunResult) -> None:
        self.result = result

    def run_tasks(self, tasks, *, view_name: str, mode: str, agent_id: str):
        return [self.result]


class _FixedResultsEnv:
    def __init__(self, results: list[TaskRunResult]) -> None:
        self.results = list(results)

    def run_tasks(self, tasks, *, view_name: str, mode: str, agent_id: str):
        return list(self.results)


class _CaptureBaseline(StaticBaseline):
    def __init__(self) -> None:
        super().__init__(baseline_id="capture", state_dir=Path("/tmp/capture"))
        self.trajectories = None

    def update(self, trajectories: TrajectoryBatch, state):
        self.trajectories = trajectories
        return UpdateResult(
            update_index=1,
            changed=False,
            status="captured",
            logs={
                "type": "capture_update",
                "num_records": len(trajectories.trajectories),
            },
        )


class _CostBaseline(StaticBaseline):
    def __init__(self) -> None:
        super().__init__(baseline_id="cost-baseline", state_dir=Path("/tmp/cost-baseline"))

    def update(self, trajectories: TrajectoryBatch, state):
        return UpdateResult(
            update_index=1,
            changed=True,
            status="updated",
            logs={
                "type": "cost_update",
                "num_records": len(trajectories.trajectories),
                "cost": {"input_tokens": 11, "output_tokens": 7, "cost_usd": 0.4},
            },
        )


class _LogCostBaseline(StaticBaseline):
    def __init__(self) -> None:
        super().__init__(baseline_id="cost-baseline", state_dir=Path("/tmp/cost-baseline"))

    def update(self, trajectories: TrajectoryBatch, state):
        return UpdateResult(
            update_index=1,
            changed=True,
            status="updated",
            logs={"cost": {"total_tokens": 42}},
        )


if __name__ == "__main__":
    unittest.main()
