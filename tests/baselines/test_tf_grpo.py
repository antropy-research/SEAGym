from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from seagym.baselines import build_baseline
from seagym.baselines.data import Trajectory, TrajectoryBatch
from seagym.baselines.tf_grpo import (
    TFGRPOBaseline,
    _apply_update_prompt_profile,
    _canonical_json,
    _filter_update_rollouts,
    _to_tf_grpo_rollout,
)


class TFGRPOBaselineTest(unittest.TestCase):
    def test_tf_grpo_runtime_config_is_loaded_from_baseline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {
                            "project_dir": "reference/tf-grpo",
                            "python_bin": "/opt/tf-grpo/bin/python",
                            "setup_commands": ["echo setup"],
                            "runtime_env": {"PYTHONPATH": "reference/tf-grpo"},
                        },
                        "models": {
                            "update_model": {
                                "provider": "deepseek",
                                "model": "deepseek/deepseek-v4-flash",
                                "api_key_env": "DEEPSEEK_API_KEY",
                            }
                        },
                    }
                },
                run_dir=Path(tmp) / "run",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, TFGRPOBaseline)
            assert isinstance(baseline, TFGRPOBaseline)
            self.assertTrue(baseline.runtime.enabled)
            self.assertEqual(baseline.runtime.python_bin, "/opt/tf-grpo/bin/python")
            self.assertEqual(baseline.runtime.env["PYTHONPATH"], "reference/tf-grpo")
            self.assertEqual(baseline.runtime.env["UTU_LLM_TYPE"], "chat.completions")
            self.assertEqual(baseline.runtime.env["UTU_LLM_MODEL"], "deepseek-v4-flash")
            self.assertEqual(baseline.runtime.env["UTU_LLM_BASE_URL"], "https://api.deepseek.com/v1")
            self.assertEqual(baseline.runtime.env["UTU_LLM_API_KEY"], "${DEEPSEEK_API_KEY}")

    def test_tf_grpo_rollout_uses_harbor_instruction_and_agent_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            trial_dir = root / "job" / "trials" / "trial-1"
            agent_dir = trial_dir / "agent"
            task_dir.mkdir(parents=True)
            agent_dir.mkdir(parents=True)
            (task_dir / "instruction.md").write_text("Inspect the logs and fix the date filter.\n", encoding="utf-8")
            (trial_dir / "result.json").write_text(
                json.dumps({"config": {"task": {"path": str(task_dir)}}}) + "\n",
                encoding="utf-8",
            )
            (agent_dir / "trajectory.json").write_text(
                json.dumps(
                    {
                        "agent": {"name": "opencode", "model_name": "deepseek-chat"},
                        "steps": [
                            {
                                "step_id": 1,
                                "source": "assistant",
                                "message": "(tool use)",
                                "tool_calls": [{"function_name": "bash", "arguments": {"cmd": "pytest -q"}}],
                                "observation": {"stdout": "1 failed"},
                            }
                        ],
                        "final_metrics": {"duration": 3.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (agent_dir / "response.txt").write_text("Done after updating the parser.\n", encoding="utf-8")
            record = {
                "task_id": "terminal-bench/sample",
                "reward": 1.0,
                "refs": {"result_path": str(trial_dir / "result.json")},
            }

            rollout = _to_tf_grpo_rollout(record)

            self.assertIn("Inspect the logs", rollout["problem"])
            trajectory = rollout["trajectories"][0]["trajectory"]
            self.assertIn("Tool call: bash", trajectory)
            self.assertIn("pytest -q", trajectory)
            self.assertIn("Final response file", trajectory)
            self.assertTrue(rollout["seagym_artifacts"]["has_harbor_trajectory"])
            self.assertFalse(rollout["seagym_artifacts"]["used_metadata_fallback"])

    def test_tf_grpo_rollout_falls_back_to_normalized_metadata(self) -> None:
        rollout = _to_tf_grpo_rollout({"task_id": "task-a", "reward": 0.0, "refs": {"result_path": "/missing/result.json"}})

        self.assertEqual(rollout["problem"], "task-a")
        self.assertEqual(rollout["reward"], 0)
        self.assertIn("SEAGym normalized result metadata", rollout["trajectories"][0]["trajectory"])
        self.assertTrue(rollout["seagym_artifacts"]["used_metadata_fallback"])

    def test_tf_grpo_update_filter_requires_harbor_trajectory_by_default(self) -> None:
        real = {"seagym_artifacts": {"has_harbor_trajectory": True, "has_response": False, "used_metadata_fallback": False}}
        response_only = {"seagym_artifacts": {"has_harbor_trajectory": False, "has_response": True, "used_metadata_fallback": False}}
        fallback = {"seagym_artifacts": {"has_harbor_trajectory": False, "has_response": False, "used_metadata_fallback": True}}

        self.assertEqual(_filter_update_rollouts([real, response_only, fallback], skip_metadata_fallback=True), [real])
        self.assertEqual(_filter_update_rollouts([real, response_only, fallback], skip_metadata_fallback=False), [real, response_only, fallback])

    def test_tf_grpo_update_prompt_profile_restores_native_prompts(self) -> None:
        package_name = "tests_tmp_tf_grpo_prompts"
        prompts_name = package_name + ".prompts"
        experience_name = package_name + ".experience"
        package = types.ModuleType(package_name)
        prompts = types.ModuleType(prompts_name)
        experience = types.ModuleType(experience_name)
        prompts.SINGLE_ROLLOUT_SUMMARY_TEMPLATE = "native-summary"
        prompts.SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = "native-summary-no-gt"
        prompts.SINGLE_QUERY_CRITIQUE_TEMPLATE = "native-critique"
        prompts.SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = "native-critique-no-gt"
        prompts.BATCH_EXPERIENCE_UPDATE_TEMPLATE = "native-batch"
        experience.SINGLE_ROLLOUT_SUMMARY_TEMPLATE = "native-summary"
        experience.SINGLE_QUERY_CRITIQUE_TEMPLATE = "native-critique"
        old_modules = {name: sys.modules.get(name) for name in (package_name, prompts_name, experience_name)}
        sys.modules[package_name] = package
        sys.modules[prompts_name] = prompts
        sys.modules[experience_name] = experience
        try:
            _apply_update_prompt_profile(experience, "meta")
            self.assertNotEqual(prompts.SINGLE_ROLLOUT_SUMMARY_TEMPLATE, "native-summary")
            self.assertNotEqual(experience.SINGLE_QUERY_CRITIQUE_TEMPLATE, "native-critique")

            _apply_update_prompt_profile(experience, "native")

            self.assertEqual(prompts.SINGLE_ROLLOUT_SUMMARY_TEMPLATE, "native-summary")
            self.assertEqual(prompts.SINGLE_QUERY_CRITIQUE_TEMPLATE, "native-critique")
            self.assertEqual(experience.SINGLE_ROLLOUT_SUMMARY_TEMPLATE, "native-summary")
            self.assertEqual(experience.SINGLE_QUERY_CRITIQUE_TEMPLATE, "native-critique")
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_tf_grpo_update_skips_batches_without_real_harbor_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "tf_grpo_project"
            package = project / "training_free_grpo" / "math"
            package.mkdir(parents=True)
            (project / "training_free_grpo" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "experience.py").write_text(
                textwrap.dedent(
                    """
                    class ExperienceUpdater:
                        def run(self, rollouts, experiences, save_dir, max_workers=4, given_ground_truth=False, only_partial_correct=False):
                            raise AssertionError("updater should not run without real trajectories")
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "training_free_grpo" or name.startswith("training_free_grpo."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {"project_dir": str(project)},
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            Path(state.metadata["experience_path"]).write_text('{"G0": "keep this"}\n', encoding="utf-8")
            batch = TrajectoryBatch(
                task_ids=["task"],
                view_name="train",
                mode="train",
                trajectories=[
                    Trajectory(
                        task_id="task",
                        attempt_id="attempt",
                        view_name="train",
                        mode="train",
                        success=True,
                        reward=1.0,
                        score=1.0,
                        rewards={"reward": 1.0},
                    )
                ],
            )

            result = built.baseline.update(batch, state)

            self.assertFalse(result.changed)
            self.assertEqual(result.status, "unchanged")
            self.assertEqual(result.metrics["metadata_fallback_records"], 1)
            self.assertEqual(result.metrics["skipped_metadata_fallback_records"], 1)
            self.assertEqual(result.metrics["skipped_non_trajectory_records"], 0)
            self.assertEqual(result.logs["skipped_reason"], "no TF-GRPO rollouts with real agent trajectories")
            self.assertIn("[G0] keep this", Path(state.metadata["prompt_template_path"]).read_text(encoding="utf-8"))

    def test_tf_grpo_runtime_update_uses_absolute_script_and_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "reference" / "tf-grpo"
            project.mkdir(parents=True)
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {
                            "project_dir": str(project),
                            "python_bin": ".venv/bin/python",
                            "update_prompt_profile": "meta",
                        },
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")
            captured: dict[str, object] = {}

            def fake_run_python_script(runtime, *, script_path, args, cwd):
                del runtime
                captured["script_path"] = script_path
                captured["args"] = args
                captured["cwd"] = cwd
                Path(args[-1]).write_text(
                    json.dumps(
                        {
                            "changed": False,
                            "status": "unchanged",
                            "metrics": {"num_experiences": 0},
                            "artifacts": {"update_dir": args[2]},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            with patch("seagym.baselines.tf_grpo.baseline.run_setup_commands", return_value={"ok": True}), patch(
                "seagym.baselines.tf_grpo.baseline.run_python_script", side_effect=fake_run_python_script
            ):
                result = built.baseline.update(empty_batch, state)

            self.assertEqual(result.status, "unchanged")
            self.assertTrue(Path(captured["script_path"]).is_absolute())
            args = captured["args"]
            assert isinstance(args, list)
            for index in (0, 1, 2, 3, 12):
                self.assertTrue(Path(args[index]).is_absolute(), msg=f"arg {index}: {args[index]}")
            self.assertEqual(args[10], "meta")
            self.assertEqual(args[11], "1")
            self.assertEqual(captured["cwd"], project)

    def test_tf_grpo_update_model_config_maps_openai_and_glm_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {
                            "project_dir": "reference/tf-grpo",
                            "update_model_ref": "update_model",
                        },
                        "models": {
                            "update_model": {
                                "provider": "openai_compatible",
                                "model": "glm-4-plus",
                                "api_base": "https://api.z.ai/api/coding/paas/v4",
                                "api_key_env": "GLM_API_KEY",
                            }
                        },
                    }
                },
                run_dir=Path(tmp) / "run",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, TFGRPOBaseline)
            assert isinstance(baseline, TFGRPOBaseline)
            self.assertEqual(baseline.runtime.env["UTU_LLM_MODEL"], "glm-4-plus")
            self.assertEqual(baseline.runtime.env["UTU_LLM_BASE_URL"], "https://api.z.ai/api/coding/paas/v4")
            self.assertEqual(baseline.runtime.env["UTU_LLM_API_KEY"], "${GLM_API_KEY}")

    def test_tf_grpo_update_reports_unchanged_when_native_experiences_do_not_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "tf_grpo_project"
            package = project / "training_free_grpo" / "math"
            package.mkdir(parents=True)
            (project / "training_free_grpo" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "experience.py").write_text(
                textwrap.dedent(
                    """
                    class ExperienceUpdater:
                        def run(self, rollouts, experiences, save_dir, max_workers=4, given_ground_truth=False, only_partial_correct=False):
                            return experiences
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "training_free_grpo" or name.startswith("training_free_grpo."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {
                            "project_dir": str(project),
                            "given_ground_truth": False,
                            "only_partial_correct": False,
                            "skip_metadata_fallback": False,
                        },
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            Path(state.metadata["experience_path"]).write_text('{"G0": "keep"}\n', encoding="utf-8")
            empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

            result = built.baseline.update(empty_batch, state)

            self.assertFalse(result.changed)
            self.assertEqual(result.status, "unchanged")
            self.assertEqual(result.metrics["num_experiences"], 1)

    def test_tf_grpo_update_captures_usage_only_from_native_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "tf_grpo_project"
            package = project / "training_free_grpo" / "math"
            package.mkdir(parents=True)
            (project / "training_free_grpo" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "experience.py").write_text(
                textwrap.dedent(
                    """
                    class ExperienceUpdater:
                        def run(self, rollouts, experiences, save_dir, max_workers=4, given_ground_truth=False, only_partial_correct=False):
                            return {"G1": {"text": "learned", "usage": {"total_tokens": 33}}}
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "training_free_grpo" or name.startswith("training_free_grpo."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {"project_dir": str(project), "skip_metadata_fallback": False},
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            batch = TrajectoryBatch(
                task_ids=["task"],
                view_name="train",
                mode="train",
                trajectories=[
                    Trajectory(
                        task_id="task",
                        attempt_id="attempt",
                        view_name="train",
                        mode="train",
                        success=True,
                        reward=1.0,
                        score=1.0,
                        rewards={"reward": 1.0},
                        cost={"total_tokens": 999},
                    )
                ],
            )

            result = built.baseline.update(batch, state)

            self.assertEqual(result.logs["cost"], {"total_tokens": 33.0})
            self.assertEqual(result.logs["cost_source"], "tf_grpo_update_outputs")

    def test_tf_grpo_update_meters_native_llm_openai_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "tf_grpo_project"
            package = project / "training_free_grpo" / "math"
            package.mkdir(parents=True)
            (project / "training_free_grpo" / "__init__.py").write_text("", encoding="utf-8")
            (project / "training_free_grpo" / "llm.py").write_text(
                textwrap.dedent(
                    """
                    class Usage:
                        prompt_tokens = 21
                        completion_tokens = 9
                        total_tokens = 30

                    class Message:
                        content = '[{"option": "add", "experience": "learned"}]'

                    class Choice:
                        message = Message()

                    class Response:
                        choices = [Choice()]
                        usage = Usage()

                    class Completions:
                        def create(self, **kwargs):
                            return Response()

                    class Chat:
                        completions = Completions()

                    class Client:
                        chat = Chat()

                    class LLM:
                        def __init__(self):
                            self.model_name = "fake-model"
                            self.client = Client()
                    """
                ),
                encoding="utf-8",
            )
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "experience.py").write_text(
                textwrap.dedent(
                    """
                    from training_free_grpo.llm import LLM

                    class ExperienceUpdater:
                        def __init__(self):
                            self.llm = LLM()

                        def run(self, rollouts, experiences, save_dir, max_workers=4, given_ground_truth=False, only_partial_correct=False):
                            self.llm.chat("summarize")
                            self.llm.chat("update")
                            return {"G1": "learned"}
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "training_free_grpo" or name.startswith("training_free_grpo."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "tf-grpo",
                        "class_path": "seagym.baselines.tf_grpo:TFGRPOBaseline",
                        "config": {"project_dir": str(project), "skip_metadata_fallback": False},
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

            result = built.baseline.update(empty_batch, state)

            self.assertEqual(
                result.logs["cost"],
                {"total_tokens": 60.0, "input_tokens": 42.0, "output_tokens": 18.0},
            )
            self.assertEqual(result.logs["cost_source"], "tf_grpo_llm_usage")

    def test_tf_grpo_canonical_json_is_order_insensitive_for_change_detection(self) -> None:
        self.assertEqual(_canonical_json({"b": 2, "a": 1}), _canonical_json({"a": 1, "b": 2}))



if __name__ == "__main__":
    unittest.main()
