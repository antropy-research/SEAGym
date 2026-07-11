from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from seagym import SEAGymTrainer
from seagym.baselines import Checkpoint, PromptRefineBaseline, StaticBaseline
from seagym.envs.harbor_env import HarborEnv
from seagym.logging import ArtifactLayout, write_run_reports
from seagym.rollout_agents.harbor import HarborRolloutAgent
from seagym.trainers import TrainerState, UpdateValidationLoop
from seagym.trainers.builder import TrainerOverrides, build_trainer_components
from seagym.trainers.builder import build_env
from seagym.utils import read_jsonl


class _ResumeLoopEngine:
    def __init__(self) -> None:
        self.context = SimpleNamespace(
            config=SimpleNamespace(
                schedule=SimpleNamespace(num_updates_per_batch=1, num_epochs=1),
            )
        )
        self.metric_input_path = Path("metric_inputs.jsonl")
        self.loaded_checkpoints: list[str] = []
        self.train_rollouts: list[list[str]] = []
        self.written_points: list[SimpleNamespace] = []

    def materialize_view(self, name: str):
        if name == "update_validation":
            return []
        if name == "replay":
            return []
        raise AssertionError(f"unexpected view: {name}")

    def make_train_batches(self) -> list[list[str]]:
        return [["task-1", "task-2"], ["task-3", "task-4"]]

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, object]:
        self.loaded_checkpoints.append(checkpoint_id)
        return {"loaded": True, "checkpoint_id": checkpoint_id}

    def record_evaluation_point(
        self,
        *,
        point_type: str,
        train_batch_index: int,
        num_train_tasks_seen: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            evaluation_point_id="E_T" if point_type == "final" else f"E_{train_batch_index}",
            point_type=point_type,
            train_batch_index=train_batch_index,
            num_train_tasks_seen=num_train_tasks_seen,
        )

    def save_checkpoint(self, checkpoint_id: str, **kwargs) -> dict[str, object]:
        return {"checkpoint_id": checkpoint_id, "metadata": kwargs.get("metadata") or {}}

    def alias_checkpoint(self, checkpoint_id: str, **kwargs) -> dict[str, object]:
        return {"checkpoint_id": checkpoint_id, "metadata": kwargs.get("metadata") or {}}

    def run_tasks(self, task_ids: list[str], *, view_name: str, mode: str, **kwargs):
        if mode == "train":
            self.train_rollouts.append(list(task_ids))
        return []

    def should_replay(self, train_batch_index: int) -> bool:
        return False

    def final_views(self) -> dict[str, list[str]]:
        return {}

    def write_evaluation_point(self, point: SimpleNamespace, **kwargs) -> SimpleNamespace:
        self.written_points.append(point)
        return point


class TrainerReportsTest(unittest.TestCase):
    def test_reports_surface_multi_attempt_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "multi_attempt_report"
            layout = ArtifactLayout.from_run_dir(run_dir)
            layout.prepare()
            layout.metrics_path.write_text("{}\n", encoding="utf-8")
            layout.evaluation_points_path.write_text("", encoding="utf-8")
            row = {
                "run_id": "run-1",
                "experiment_id": "exp-1",
                "split_id": "split-1",
                "agent_id": "agent",
                "task_id": "task-1",
                "attributes": {"domain": "code"},
                "view_name": "train",
                "mode": "train",
                "score": 1.0,
                "success": True,
                "error": None,
                "refs": {
                    "job_dir": "job",
                    "result_path": "job/task-1__b/result.json",
                    "all_attempts": [
                        {"attempt_id": "task-1__a", "success": False, "score": 0.0},
                        {"attempt_id": "task-1__b", "success": True, "score": 1.0},
                    ],
                },
            }
            layout.task_results_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            write_run_reports(run_dir)

            summary = layout.summary_path.read_text(encoding="utf-8")
            tasks_csv = layout.tasks_csv_path.read_text(encoding="utf-8")
            self.assertIn("- Task attempts: `2`", summary)
            self.assertIn("- Attempt successes: `1`", summary)
            self.assertIn("attempt_count,attempt_successes,attempt_best_score", tasks_csv)
            self.assertIn("2,1,1.0", tasks_csv)

    def test_artifact_layout_resolves_relative_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                layout = ArtifactLayout.from_run_dir("relative_run")
            finally:
                os.chdir(old_cwd)

            self.assertTrue(layout.run_dir.is_absolute())
            self.assertEqual(layout.run_dir, (Path(tmp) / "relative_run").resolve())
            self.assertEqual(layout.harbor_jobs_dir, layout.run_dir / "harbor" / "jobs")

    def test_trainer_writes_audit_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "trainer_run"

            result_dir = SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id="deterministic",
            ).fit()

            self.assertEqual(result_dir, run_dir.resolve())
            self.assertTrue((run_dir / "inputs" / "experiment_config.json").exists())
            self.assertTrue((run_dir / "records" / "metric_inputs.jsonl").exists())
            self.assertTrue((run_dir / "records" / "task_results.jsonl").exists())
            self.assertTrue((run_dir / "reports" / "summary.md").exists())
            self.assertTrue((run_dir / "reports" / "tasks.csv").exists())
            self.assertTrue((run_dir / "harbor" / "jobs").is_dir())

    def test_trainer_overwrite_preserves_runtime_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "trainer_run"
            runtime_dir = run_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "runtime_check.json").write_text('{"ok": true}\n', encoding="utf-8")

            SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id="deterministic",
                overwrite=True,
            ).fit()

            self.assertTrue((run_dir / "runtime" / "runtime_check.json").exists())

    def test_trainer_records_final_a0_at_and_replay_point_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "trainer_protocol_run"

            SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id="deterministic",
            ).fit()

            rows = read_jsonl(run_dir / "records" / "metric_inputs.jsonl")
            final_rows = [row for row in rows if row["mode"].startswith("final")]
            roles = sorted({row.get("baseline_role") for row in final_rows})
            self.assertEqual(roles, ["A_0", "A_T"])
            self.assertTrue(all(row.get("evaluation_point_id") == "E_T" for row in final_rows))
            replay_rows = [row for row in rows if row["view_name"] == "replay"]
            self.assertTrue(replay_rows)
            self.assertTrue(all(row.get("evaluation_point_id") for row in replay_rows))

            points = read_jsonl(run_dir / "records" / "evaluation_points.jsonl")
            final_point = next(point for point in points if point["evaluation_point_id"] == "E_T")
            self.assertIn("baseline_score", final_point["evaluations"]["id_test"])
            self.assertIn("gain_vs_A_0", final_point["evaluations"]["id_test"])

            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("final_gain", metrics)
            self.assertIn("validation_supported_update_rate", metrics)
            self.assertIn("tokens", metrics)
            self.assertGreater(metrics["tokens"]["rollout"]["num_records_with_tokens"], 0)
            self.assertGreater(metrics["tokens"]["update"]["num_records"], 0)

    def test_update_validation_runs_only_at_epoch_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "epoch_end_validation_run"

            SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id="deterministic",
            ).fit()

            points = read_jsonl(run_dir / "records" / "evaluation_points.jsonl")
            validation_points = [point for point in points if point["type"] == "update_validation"]
            self.assertEqual([point["step"]["train_batch_index"] for point in validation_points], [2])
            self.assertEqual(validation_points[0]["evaluation_point_id"], "E_2")
            self.assertIn("update_validation", validation_points[0]["evaluations"])

    def test_update_validation_loop_can_resume_after_completed_batch(self) -> None:
        engine = _ResumeLoopEngine()

        UpdateValidationLoop().run(
            engine,
            resume=TrainerState(
                checkpoint_id="epoch_0001",
                epoch=1,
                train_batch_index=2,
                num_train_tasks_seen=4,
                updates_completed=2,
                global_step=2,
                previous_update_validation_results=[],
            ),
        )

        self.assertEqual(engine.loaded_checkpoints, ["epoch_0001"])
        self.assertEqual(engine.train_rollouts, [])
        self.assertEqual([point.point_type for point in engine.written_points], ["final"])

    def test_update_validation_loop_rejects_invalid_resume_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume.train_batch_index"):
            UpdateValidationLoop().run(
                _ResumeLoopEngine(),
                resume=TrainerState(
                    checkpoint_id="epoch_0002",
                    epoch=1,
                    train_batch_index=3,
                    num_train_tasks_seen=6,
                    updates_completed=3,
                    global_step=3,
                    previous_update_validation_results=[],
                ),
            )

    def test_static_baseline_writes_lifecycle_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "static_baseline_run"
            baseline = StaticBaseline(
                baseline_id="static-baseline",
                state_dir=run_dir / "agent_state" / "static-baseline",
            )

            SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id="static-baseline",
                baseline=baseline,
            ).fit()

            updates = read_jsonl(run_dir / "records" / "agent_updates.jsonl")
            points = read_jsonl(run_dir / "records" / "evaluation_points.jsonl")
            self.assertGreater(len(updates), 0)
            self.assertTrue((run_dir / "checkpoints" / "initial" / "checkpoint.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "epoch_0001" / "checkpoint.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "final" / "checkpoint.json").exists())
            self.assertIn("agent_checkpoint", points[0]["refs"])
            checkpoint_rows = read_jsonl(run_dir / "records" / "agent_checkpoints.jsonl")
            epoch_rows = [row for row in checkpoint_rows if row["checkpoint_id"] == "epoch_0001"]
            self.assertEqual(len(epoch_rows), 1)
            self.assertEqual(epoch_rows[0]["metadata"]["epoch_index"], 1)
            self.assertEqual(epoch_rows[0]["metadata"]["kind"], "epoch")
            e2_rows = [row for row in checkpoint_rows if row["checkpoint_id"] == "E_2"]
            self.assertEqual(len(e2_rows), 1)
            self.assertEqual(e2_rows[0]["metadata"]["alias_of"], "epoch_0001")
            self.assertEqual(e2_rows[0]["manifest"]["baseline"]["type"], "baseline_checkpoint_alias")
            final_rows = [row for row in checkpoint_rows if row["checkpoint_id"] == "final"]
            self.assertEqual(len(final_rows), 1)
            self.assertEqual(final_rows[0]["metadata"]["alias_of"], "epoch_0001")
            self.assertEqual(final_rows[0]["manifest"]["baseline"]["type"], "baseline_checkpoint_alias")

            alias_manifest = json.loads((run_dir / "checkpoints" / "final" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(alias_manifest["baseline"]["state_ref"], "../epoch_0001/baseline_state")

            restored = StaticBaseline(
                baseline_id="static-baseline",
                state_dir=run_dir / "restored_state" / "static-baseline",
            )
            restored.load_checkpoint(Checkpoint(run_dir / "checkpoints" / "final"))
            self.assertTrue((run_dir / "restored_state" / "static-baseline" / "baseline_state.json").exists())

            relocated = Path(tmp) / "relocated"
            shutil.copytree(run_dir / "checkpoints" / "final", relocated / "checkpoints" / "final")
            shutil.copytree(run_dir / "checkpoints" / "epoch_0001", relocated / "checkpoints" / "epoch_0001")
            relocated_baseline = StaticBaseline(
                baseline_id="static-baseline",
                state_dir=relocated / "restored_state" / "static-baseline",
            )
            relocated_baseline.load_checkpoint(Checkpoint(relocated / "checkpoints" / "final"))
            self.assertTrue((relocated / "restored_state" / "static-baseline" / "baseline_state.json").exists())

    def test_trainer_supports_multiple_updates_per_train_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "multi_update_run"
            config_path = Path(tmp) / "multi_update_config.json"
            config = json.loads(Path("tests/fixtures/pilot/configs/pilot.json").read_text(encoding="utf-8"))
            config["task_dataset"]["path"] = str(Path("tests/fixtures/pilot/tasks/task_index.json").resolve())
            config["split_manifest"]["path"] = str(Path("tests/fixtures/pilot/splits/pilot_seed42.json").resolve())
            config["schedule"]["num_updates_per_batch"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")

            SEAGymTrainer(config_path, run_dir=run_dir, agent_id="static-baseline").fit()

            updates = read_jsonl(run_dir / "records" / "agent_updates.jsonl")
            train_rows = [row for row in read_jsonl(run_dir / "records" / "metric_inputs.jsonl") if row["mode"] == "train"]
            self.assertEqual(len(updates), 4)
            self.assertEqual(len(train_rows), 8)
            self.assertEqual([row["update_repeat_index"] for row in updates], [1, 2, 1, 2])
            self.assertEqual([row["global_update_index"] for row in updates], [1, 2, 3, 4])
            self.assertEqual([row["num_train_tasks_seen"] for row in updates], [2, 4, 6, 8])
            self.assertEqual(sorted({row["update_repeat_index"] for row in train_rows}), [1, 2])
            self.assertEqual(sorted({row["global_update_index"] for row in train_rows}), [1, 2, 3, 4])
            self.assertEqual(sorted({row["num_updates_per_batch"] for row in train_rows}), [2])
            self.assertTrue((run_dir / "checkpoints" / "epoch_0001" / "checkpoint.json").exists())

    def test_from_config_builds_baseline_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "from_config_static"
            config_path = Path(tmp) / "static_config.json"
            config = json.loads(Path("tests/fixtures/pilot/configs/pilot.json").read_text(encoding="utf-8"))
            config["task_dataset"]["path"] = str(Path("tests/fixtures/pilot/tasks/task_index.json").resolve())
            config["split_manifest"]["path"] = str(Path("tests/fixtures/pilot/splits/pilot_seed42.json").resolve())
            config["baseline"] = {
                "name": "static-baseline",
                "class_path": "seagym.baselines.static:StaticBaseline",
                "config": {},
            }
            config["rollout_agent"] = {
                "name": "static-baseline",
                "class_path": "seagym.rollout_agents.harbor:HarborRolloutAgent",
                "config": {"agent": "static-baseline"},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            trainer = SEAGymTrainer.from_config(
                config_path,
                overrides=TrainerOverrides(run_dir=run_dir),
            )

            self.assertEqual(trainer.agent_id, "static-baseline")
            self.assertIsInstance(trainer.baseline, StaticBaseline)
            self.assertEqual(trainer.layout.run_dir, run_dir.resolve())

            result_dir = trainer.fit()
            self.assertEqual(result_dir, run_dir.resolve())
            self.assertTrue((run_dir / "records" / "agent_updates.jsonl").exists())

    def test_from_config_warns_but_does_not_block_nonstandard_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "nonstandard_run_dir"

            config_path = Path(tmp) / "harbor_config.json"
            config = json.loads(Path("runs/local_harbor/configs/oracle_no_update.json").read_text(encoding="utf-8"))
            config["task_dataset"]["path"] = str(Path("runs/local_harbor/tasks/task_index.json").resolve())
            config["split_manifest"]["path"] = str(Path("runs/local_harbor/splits/local_harbor_seed42.json").resolve())
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch.dict(os.environ, {"SEAGYM_DATA_ROOT": str(Path(tmp) / "datasets")}):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    trainer = SEAGymTrainer.from_config(
                        config_path,
                        overrides=TrainerOverrides(run_dir=run_dir),
                    )

            self.assertEqual(trainer.layout.run_dir, run_dir.resolve())
            self.assertTrue(any("config:run_dir" in str(item.message) for item in caught))

    def test_prompt_refine_baseline_writes_llm_update_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "prompt_refine_baseline"
            baseline = PromptRefineBaseline(
                baseline_id="prompt-refine",
                state_dir=run_dir / "agent_state" / "prompt_refine",
            )
            baseline._refine_prompt = lambda prompt_input: "Refined prompt for future tasks."

            SEAGymTrainer(
                "tests/fixtures/pilot/configs/pilot.json",
                run_dir=run_dir,
                agent_id=baseline.baseline_id,
                baseline=baseline,
            ).fit()

            updates = read_jsonl(run_dir / "records" / "agent_updates.jsonl")
            self.assertEqual(updates[0]["summary"]["type"], "baseline_update")
            self.assertEqual(updates[0]["summary"]["logs"]["type"], "llm_prompt_refine_update")
            self.assertTrue(updates[0]["summary"]["changed"])
            self.assertTrue((run_dir / "agent_state" / "prompt_refine" / "prompt_template.md").exists())
            self.assertTrue((run_dir / "checkpoints" / "final" / "checkpoint.json").exists())

    def test_prompt_refine_baseline_can_configure_harbor_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "prompt_refine_harbor"
            baseline = PromptRefineBaseline(
                baseline_id="prompt-refine",
                state_dir=run_dir / "agent_state" / "prompt_refine",
            )
            state = baseline.initialize(run_dir)
            rollout_agent = HarborRolloutAgent(agent_id="codex", agent_kwargs={"reasoning_effort": "low"})

            env = HarborEnv(
                run_dir / "harbor" / "jobs",
                agent_spec=rollout_agent.harbor_agent_spec(state),
                model_name="openai/test-model",
            )

            self.assertIsInstance(env, HarborEnv)
            self.assertEqual(env.agent_id, "codex")
            self.assertEqual(env.model_name, "openai/test-model")
            self.assertIn("prompt_template_path", env.agent_kwargs)
            self.assertEqual(env.agent_kwargs["reasoning_effort"], "low")

    def test_local_harbor_baseline_configs_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "codex_static"
            with patch.dict(
                os.environ,
                {
                    "HTTP_PROXY": "http://127.0.0.1:7890",
                    "HTTPS_PROXY": "http://127.0.0.1:7890",
                    "ALL_PROXY": "socks5://127.0.0.1:7891",
                    "NO_PROXY": "localhost,127.0.0.1",
                    "SEAGYM_CONTAINER_HTTP_PROXY": "http://192.168.5.2:7890",
                    "SEAGYM_CONTAINER_HTTPS_PROXY": "http://192.168.5.2:7890",
                    "SEAGYM_CONTAINER_ALL_PROXY": "socks5://192.168.5.2:7891",
                    "SEAGYM_CONTAINER_NO_PROXY": "localhost,127.0.0.1",
                    "SEAGYM_DATA_ROOT": str(Path(tmp) / "datasets"),
                },
            ):
                codex = build_trainer_components(
                    "runs/local_harbor/configs/codex_static.json",
                    overrides=TrainerOverrides(run_dir=run_dir),
                )
                prompt_refine = build_trainer_components(
                    "runs/local_harbor/configs/prompt_refine_deepseek.json",
                    overrides=TrainerOverrides(run_dir=Path(tmp) / "prompt_refine"),
                )
                oracle = build_trainer_components(
                    "runs/local_harbor/configs/oracle_no_update.json",
                    overrides=TrainerOverrides(run_dir=Path(tmp) / "oracle"),
                )
                opencode = build_trainer_components(
                    "runs/local_harbor/configs/opencode_static.json",
                    overrides=TrainerOverrides(run_dir=Path(tmp) / "opencode"),
                )

            self.assertEqual(oracle.agent_id, "oracle")
            self.assertIsInstance(oracle.env, HarborEnv)
            self.assertEqual(codex.agent_id, "codex")
            self.assertIsInstance(codex.env, HarborEnv)
            self.assertEqual(codex.rollout_agent.agent_id, "codex")
            self.assertNotIn("HTTP_PROXY", codex.env.agent_env)
            self.assertNotIn("HTTP_PROXY", codex.env.verifier_env)
            self.assertEqual(codex.env.model_name, "gpt-5.3-codex")
            self.assertEqual(prompt_refine.agent_id, "codex")
            self.assertIsInstance(prompt_refine.env, HarborEnv)
            self.assertEqual(prompt_refine.rollout_agent.agent_id, "codex")
            self.assertNotIn("ALL_PROXY", prompt_refine.env.agent_env)
            self.assertEqual(opencode.agent_id, "opencode")
            self.assertIsInstance(opencode.env, HarborEnv)
            self.assertEqual(opencode.rollout_agent.agent_id, "opencode")
            self.assertEqual(opencode.env.model_name, "deepseek/deepseek-chat")

    def test_harbor_backend_env_config_can_select_cloud_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "daytona"
            rollout_agent = HarborRolloutAgent(agent_id="oracle")
            env = build_env(
                {
                    "backend": {
                        "name": "harbor",
                        "env": "daytona",
                        "environment_kwargs": {"workspace_size": 4},
                    }
                },
                backend_name="harbor",
                run_dir=run_dir,
                rollout_agent=rollout_agent,
                rollout_model=None,
            )

            self.assertIsInstance(env, HarborEnv)
            assert isinstance(env, HarborEnv)
            self.assertEqual(env.env, "daytona")
            self.assertEqual(env.environment_kwargs, {"workspace_size": 4})


if __name__ == "__main__":
    unittest.main()
