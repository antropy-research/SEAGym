from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from seagym.baselines import build_baseline
from seagym.baselines.ace import ACEBaseline, _materialize_ace_traces
from seagym.baselines.ace.runner import _ACE_RUNTIME_RUNNER
from seagym.baselines.data import Trajectory, TrajectoryBatch
from seagym.envs import TaskRunResult
from seagym.rollout_agents import build_rollout_agent
from seagym.rollout_agents.harbor import HarborRolloutAgent


class ACEBaselineTest(unittest.TestCase):
    def test_ace_runtime_runner_marks_all_failed_samples_as_error(self) -> None:
        self.assertIn("all_results_failed", _ACE_RUNTIME_RUNNER)
        self.assertIn("ACEAllSamplesFailed", _ACE_RUNTIME_RUNNER)

    def test_ace_update_model_config_maps_deepseek_to_native_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
                            "update_model_ref": "update_model",
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
                run_dir=Path(tmp) / "ace",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, ACEBaseline)
            assert isinstance(baseline, ACEBaseline)
            self.assertEqual(baseline.model, "deepseek:deepseek-v4-flash")
            self.assertEqual(baseline.runtime.env["DEEPSEEK_API_KEY"], "${DEEPSEEK_API_KEY}")

    def test_ace_update_model_config_maps_openai_to_chat_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
                            "update_model_ref": "update_model",
                        },
                        "models": {
                            "update_model": {
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "api_key_env": "OPENAI_API_KEY",
                            }
                        },
                    }
                },
                run_dir=Path(tmp) / "ace",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, ACEBaseline)
            assert isinstance(baseline, ACEBaseline)
            self.assertEqual(baseline.model, "openai-chat:gpt-4o-mini")
            self.assertEqual(baseline.runtime.env["OPENAI_API_KEY"], "${OPENAI_API_KEY}")

    def test_ace_update_model_config_maps_glm_openai_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
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
                run_dir=Path(tmp) / "ace",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, ACEBaseline)
            assert isinstance(baseline, ACEBaseline)
            self.assertEqual(baseline.model, "openai-chat:glm-4-plus")
            self.assertEqual(baseline.runtime.env["OPENAI_API_KEY"], "${GLM_API_KEY}")
            self.assertEqual(baseline.runtime.env["OPENAI_BASE_URL"], "https://api.z.ai/api/coding/paas/v4")

    def test_ace_legacy_litellm_deepseek_model_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
                            "model": "litellm:deepseek/deepseek-chat",
                        },
                    }
                },
                run_dir=Path(tmp) / "ace",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, ACEBaseline)
            assert isinstance(baseline, ACEBaseline)
            self.assertEqual(baseline.model, "deepseek:deepseek-chat")
            self.assertEqual(baseline.runtime.env["DEEPSEEK_API_KEY"], "${DEEPSEEK_API_KEY}")

    def test_ace_run_config_uses_deepseek_rollout_and_run_local_state(self) -> None:
        config = {
            "baseline": {
                "name": "ace-trace-learning",
                "class_path": "seagym.baselines.ace:ACEBaseline",
                "config": {
                    "project_dir": "reference/ace",
                    "model": "deepseek:deepseek-chat",
                },
            },
            "rollout_agent": {
                "name": "opencode",
                "class_path": "seagym.rollout_agents.harbor:HarborRolloutAgent",
                "config": {
                    "agent": "opencode",
                    "import_path": "seagym.rollout_agents.opencode_preinstalled:PreinstalledOpenCode",
                    "model_ref": "rollout_model",
                },
                "models": {
                    "rollout_model": {
                        "model": "deepseek/deepseek-v4-flash",
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            baseline_build = build_baseline(config, run_dir=run_dir, base_dir=Path.cwd())
            rollout_build = build_rollout_agent(config, run_dir=run_dir, base_dir=Path.cwd())

            self.assertIsInstance(baseline_build.baseline, ACEBaseline)
            self.assertEqual(baseline_build.baseline.state_dir, (run_dir / "agent_state" / "ace-trace-learning").resolve())
            self.assertEqual(rollout_build.agent_id, "opencode")
            self.assertIsInstance(rollout_build.rollout_agent, HarborRolloutAgent)
            agent = rollout_build.rollout_agent
            assert isinstance(agent, HarborRolloutAgent)
            self.assertEqual(
                agent.agent_import_path,
                "seagym.rollout_agents.opencode_preinstalled:PreinstalledOpenCode",
            )
            self.assertEqual(rollout_build.rollout_model, "deepseek/deepseek-v4-flash")

    def test_ace_materializes_standard_trace_from_harbor_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("Solve the terminal task.", encoding="utf-8")
            trial_dir = root / "trial"
            (trial_dir / "agent").mkdir(parents=True)
            (trial_dir / "verifier").mkdir()
            (trial_dir / "agent" / "trajectory.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "reasoning_content": "I should inspect the instructions first.",
                                "message": "(tool use)",
                                "observation": {
                                    "results": [
                                        {
                                            "content": "<path>/app/instruction.md</path>\n<content>\nSolve the terminal task.\n</content>"
                                        }
                                    ]
                                },
                            },
                            {
                                "reasoning_content": "The implementation is complete.",
                                "message": "Done.",
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "source": "terminal-bench-2",
                        "config": {"task": {"path": str(task_dir)}},
                        "verifier_result": {
                            "rewards": {"reward": 0.0},
                            "correct_answer": "SECRET_GT",
                            "reasoning": "SECRET_JUDGE_RATIONALE",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            trajectory = Trajectory(
                task_id="task-a",
                attempt_id="attempt-1",
                view_name="train",
                mode="train",
                success=False,
                reward=0.0,
                score=0.0,
                rewards={"reward": 0.0},
                refs={"result_path": str(trial_dir / "result.json")},
            )
            batch = TrajectoryBatch(trajectories=[trajectory], task_ids=["task-a"], view_name="train", mode="train")

            traces = _materialize_ace_traces(batch)

            self.assertEqual(len(traces), 1)
            trace = traces[0]
            self.assertEqual(trace["question"], "Solve the terminal task.")
            self.assertIn("I should inspect", trace["reasoning"])
            self.assertEqual(trace["answer"], "Done.")
            self.assertIn("success=False", trace["feedback"])
            self.assertIsNone(trace["ground_truth"])
            serialized = json.dumps(trace)
            self.assertNotIn("SECRET_GT", serialized)
            self.assertNotIn("SECRET_JUDGE_RATIONALE", serialized)

    def test_ace_update_passes_standard_traces_to_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ace_project"
            package = project / "ace"
            package.mkdir(parents=True)
            capture_path = root / "captured_records.json"
            (package / "__init__.py").write_text(
                textwrap.dedent(
                    f"""
                    import json

                    class ACELiteLLM:
                        def __init__(self, model, **kwargs):
                            self.model = model

                        def learn_from_traces(self, records, epochs=1, wait=True):
                            with open({str(capture_path)!r}, "w", encoding="utf-8") as handle:
                                json.dump(records, handle)
                            return []

                        def save(self, path):
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(json.dumps({{"skills": {{"s1": {{}}}}}}))

                        def get_strategies(self):
                            return "strategy"
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "ace" or name.startswith("ace."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {"project_dir": str(project)},
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            trajectory = Trajectory(
                task_id="train-a",
                attempt_id="attempt-1",
                view_name="train",
                mode="train",
                success=True,
                reward=1.0,
                score=1.0,
                rewards={"reward": 1.0},
            )
            batch = TrajectoryBatch(trajectories=[trajectory], task_ids=["train-a"], view_name="train", mode="train")

            result = built.baseline.update(batch, state)

            self.assertTrue(result.changed)
            records = json.loads(capture_path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["question"], "train-a")
            self.assertIn("feedback", records[0])
            self.assertIn("ground_truth", records[0])
            self.assertNotIn("refs", records[0])
            update_dir = Path(result.artifacts["update_dir"])
            self.assertTrue((update_dir / "seagym_trajectories.json").exists())
            self.assertTrue((update_dir / "ace_traces.json").exists())

    def test_ace_update_can_inject_custom_update_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ace_project"
            package = project / "ace"
            implementations = package / "implementations"
            implementations.mkdir(parents=True)
            capture_path = root / "captured_kwargs.json"
            reflector_prompt_path = root / "reflector.md"
            skill_manager_prompt_path = root / "skill_manager.md"
            skill_manager_system_prompt_path = root / "skill_manager_system.md"
            reflector_prompt_path.write_text("custom reflector {feedback}", encoding="utf-8")
            skill_manager_prompt_path.write_text("custom skill manager {reflections}", encoding="utf-8")
            skill_manager_system_prompt_path.write_text("custom skill manager system", encoding="utf-8")
            (package / "__init__.py").write_text(
                textwrap.dedent(
                    f"""
                    import json

                    class ACELiteLLM:
                        def __init__(self, model, **kwargs):
                            payload = {{
                                "model": model,
                                "reflector_prompt": getattr(kwargs.get("reflector"), "prompt_template", None),
                                "skill_manager_prompt": getattr(kwargs.get("skill_manager"), "prompt_template", None),
                                "skill_manager_system_prompt": getattr(kwargs.get("skill_manager"), "system_prompt", None),
                            }}
                            with open({str(capture_path)!r}, "w", encoding="utf-8") as handle:
                                json.dump(payload, handle)

                        def learn_from_traces(self, records, epochs=1, wait=True):
                            return []

                        def save(self, path):
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(json.dumps({{"skills": {{"s1": {{}}}}}}))

                        def get_strategies(self):
                            return "strategy"
                    """
                ),
                encoding="utf-8",
            )
            (implementations / "__init__.py").write_text(
                textwrap.dedent(
                    """
                    class Reflector:
                        def __init__(self, model, *, prompt_template, **kwargs):
                            self.model = model
                            self.prompt_template = prompt_template

                    class SkillManager:
                        def __init__(self, model, *, prompt_template=None, system_prompt=None, **kwargs):
                            self.model = model
                            self.prompt_template = prompt_template
                            self.system_prompt = system_prompt
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "ace" or name.startswith("ace."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": str(project),
                            "update_prompt_variant": "custom_test",
                            "reflector_prompt_path": str(reflector_prompt_path),
                            "skill_manager_prompt_path": str(skill_manager_prompt_path),
                            "skill_manager_system_prompt_path": str(skill_manager_system_prompt_path),
                        },
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

            result = built.baseline.update(empty_batch, state)

            self.assertTrue(result.changed)
            self.assertEqual(state.metadata["update_prompt_config"]["variant"], "custom_test")
            captured = json.loads(capture_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["reflector_prompt"], "custom reflector {feedback}")
            self.assertEqual(captured["skill_manager_prompt"], "custom skill manager {reflections}")
            self.assertEqual(captured["skill_manager_system_prompt"], "custom skill manager system")

    def test_ace_custom_update_prompt_variant_requires_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "custom prompt path"):
                build_baseline(
                    {
                        "baseline": {
                            "name": "ace",
                            "class_path": "seagym.baselines.ace:ACEBaseline",
                            "config": {
                                "project_dir": "reference/ace",
                                "update_prompt_variant": "custom_without_paths",
                            },
                        }
                    },
                    run_dir=Path(tmp) / "run",
                    base_dir=Path.cwd(),
                )

    def test_ace_batch_reflect_then_update_calls_skill_manager_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ace_project"
            package = project / "ace"
            package.mkdir(parents=True)
            capture_path = root / "batch_update_capture.json"
            (package / "__init__.py").write_text(
                textwrap.dedent(
                    f"""
                    import json

                    class FakeReflection:
                        def __init__(self, index):
                            self.reasoning = f"reasoning-{{index}}"
                            self.key_insight = f"insight-{{index}}"

                    class FakeReflector:
                        def __init__(self):
                            self.calls = []

                        def reflect(self, **kwargs):
                            self.calls.append(kwargs["question"])
                            return FakeReflection(len(self.calls))

                    class FakeSkillManager:
                        def update_skills(self, **kwargs):
                            payload = {{
                                "num_reflections": len(kwargs["reflections"]),
                                "questions": kwargs["question_context"],
                                "progress": kwargs["progress"],
                            }}
                            with open({str(capture_path)!r}, "w", encoding="utf-8") as handle:
                                json.dump(payload, handle)
                            return {{"operations": [{{"type": "add"}}]}}

                    class FakeSkillbook:
                        def __init__(self):
                            self.skills = {{}}

                    class ACELiteLLM:
                        def __init__(self, model, **kwargs):
                            self.model = model
                            self.reflector = FakeReflector()
                            self.skill_manager = FakeSkillManager()
                            self.skillbook = FakeSkillbook()

                        def save(self, path):
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(json.dumps({{"skills": {{"s1": {{}}}}}}))

                        def get_strategies(self):
                            return "batch strategy"
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "ace" or name.startswith("ace."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {"project_dir": str(project), "update_mode": "batch_reflect_then_update"},
                    }
                },
                run_dir=root / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(root / "run")
            batch = TrajectoryBatch(
                trajectories=[
                    Trajectory(
                        task_id="train-a",
                        attempt_id="attempt-a",
                        view_name="train",
                        mode="train",
                        success=False,
                        reward=0.0,
                        score=0.0,
                        rewards={"reward": 0.0},
                    ),
                    Trajectory(
                        task_id="train-b",
                        attempt_id="attempt-b",
                        view_name="train",
                        mode="train",
                        success=True,
                        reward=1.0,
                        score=1.0,
                        rewards={"reward": 1.0},
                    ),
                ],
                task_ids=["train-a", "train-b"],
                view_name="train",
                mode="train",
            )

            result = built.baseline.update(batch, state)

            self.assertTrue(result.changed)
            captured = json.loads(capture_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["num_reflections"], 2)
            self.assertIn("batch_size 2", captured["progress"])
            self.assertIn("train-a", captured["questions"])
            self.assertIn("train-b", captured["questions"])

    def test_ace_direct_update_captures_result_usage_before_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ace_project"
            package = project / "ace"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                textwrap.dedent(
                    """
                    class FakeUsage:
                        input_tokens = 5
                        output_tokens = 7
                        total_tokens = 12

                    class FakeResult:
                        def usage(self):
                            return FakeUsage()

                    class ACELiteLLM:
                        def __init__(self, model, **kwargs):
                            self.model = model

                        def learn_from_traces(self, records, epochs=1, wait=True):
                            return [FakeResult()]

                        def save(self, path):
                            import json
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(json.dumps({"skills": {"s1": {}}}))

                        def get_strategies(self):
                            return "strategy"
                    """
                ),
                encoding="utf-8",
            )
            for name in list(sys.modules):
                if name == "ace" or name.startswith("ace."):
                    del sys.modules[name]
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {"project_dir": str(project)},
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
                {"total_tokens": 12.0, "input_tokens": 5.0, "output_tokens": 7.0},
            )
            self.assertEqual(result.logs["cost_source"], "ace_results_usage")

    def test_ace_runtime_setup_failure_is_recorded_as_update_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
                            "setup_commands": ["exit 7"],
                            "setup_timeout_sec": 5,
                        },
                    }
                },
                run_dir=Path(tmp) / "run",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(Path(tmp) / "run")
            empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

            result = built.baseline.update(empty_batch, state)

            self.assertEqual(result.status, "error")
            self.assertFalse(result.changed)
            self.assertTrue(Path(result.artifacts["error_summary"]).exists())
            self.assertIn("setup", result.logs["logs"])

    def test_ace_runtime_update_uses_absolute_script_and_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "reference" / "ace"
            project.mkdir(parents=True)
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": str(project),
                            "python_bin": ".venv/bin/python",
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
                            "metrics": {"num_trajectories": 0},
                            "artifacts": {"update_dir": args[2]},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            with patch("seagym.baselines.ace.baseline.run_setup_commands", return_value={"ok": True}), patch(
                "seagym.baselines.ace.baseline.run_python_script", side_effect=fake_run_python_script
            ):
                result = built.baseline.update(empty_batch, state)

            self.assertEqual(result.status, "unchanged")
            self.assertTrue(Path(captured["script_path"]).is_absolute())
            args = captured["args"]
            assert isinstance(args, list)
            for index in (0, 1, 2, 3, 14):
                self.assertTrue(Path(args[index]).is_absolute(), msg=f"arg {index}: {args[index]}")
            self.assertEqual(args[9], "ace_default")
            self.assertEqual(args[10], "")
            self.assertEqual(args[11], "")
            self.assertEqual(args[12], "")
            self.assertEqual(args[13], "native_trace_analyser")
            self.assertEqual(captured["cwd"], project)


if __name__ == "__main__":
    unittest.main()
