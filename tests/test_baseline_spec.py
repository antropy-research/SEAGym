from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seagym.baselines import BaseBaseline, PromptRefineBaseline, StaticBaseline, build_baseline
from seagym.baselines.ace import ACEBaseline
from seagym.baselines.gepa import GEPABaseline
from seagym.baselines.tf_grpo import TFGRPOBaseline
from seagym.rollout_agents import build_rollout_agent
from seagym.rollout_agents.ahe_nexau import AHENexAURolloutAgent
from seagym.rollout_agents.harbor import HarborRolloutAgent


def _has_path(value: object, key_path: tuple[str, ...]) -> bool:
    current = value
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


class BaselineConfigTest(unittest.TestCase):
    def test_run_configs_use_baseline_class_path(self) -> None:
        forbidden_keys = {
            ("baseline", "type"),
            ("baseline", "project"),
            ("baseline", "rollout"),
            ("baseline", "con" + "troller"),
            ("baseline", "adapter"),
        }
        for path in Path("runs").glob("**/configs/*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            baseline = config.get("baseline") or {}
            rollout_agent = config.get("rollout_agent") or {}
            self.assertIsInstance(baseline, dict, msg=str(path))
            self.assertIsInstance(rollout_agent, dict, msg=str(path))
            self.assertIn("class_path", baseline, msg=str(path))
            self.assertIn("class_path", rollout_agent, msg=str(path))
            for key_path in forbidden_keys:
                self.assertFalse(_has_path(config, key_path), msg=f"{path}: {'.'.join(key_path)}")

    def test_fixture_config_uses_baseline_class_path(self) -> None:
        config = json.loads(Path("tests/fixtures/pilot/configs/pilot.json").read_text(encoding="utf-8"))
        self.assertEqual(config["baseline"]["class_path"], "seagym.baselines.static:StaticBaseline")
        self.assertEqual(config["rollout_agent"]["class_path"], "seagym.rollout_agents.harbor:HarborRolloutAgent")
        self.assertNotIn("con" + "troller", config["baseline"])
        self.assertNotIn("rollout", config["baseline"])

    def test_builds_static_baseline_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "baseline": {
                    "name": "static",
                    "class_path": "seagym.baselines.static:StaticBaseline",
                    "config": {},
                }
            }
            built = build_baseline(config, run_dir=Path(tmp) / "run", base_dir=Path(tmp))

            self.assertEqual(built.agent_id, "static")
            self.assertIsInstance(built.baseline, StaticBaseline)

    def test_builds_prompt_refine_baseline_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "baseline": {
                    "name": "prompt-refine",
                    "class_path": "seagym.baselines.prompt_refine:PromptRefineBaseline",
                    "config": {"update_model_ref": "update_model"},
                    "models": {
                        "update_model": {
                            "provider": "openai_compatible",
                            "model": "deepseek-v4-flash",
                            "api_base": "https://api.deepseek.com",
                            "api_key_env": "DEEPSEEK_API_KEY",
                        }
                    },
                }
            }
            built = build_baseline(config, run_dir=Path(tmp) / "run", base_dir=Path(tmp))

            self.assertIsInstance(built.baseline, PromptRefineBaseline)
            baseline = built.baseline
            assert isinstance(baseline, PromptRefineBaseline)
            self.assertEqual(baseline.refiner_model.name, "deepseek-v4-flash")


    def test_method_baselines_inherit_base_baseline_directly(self) -> None:
        class_paths = {
            "ace": ("seagym.baselines.ace:ACEBaseline", ACEBaseline),
            "gepa": ("seagym.baselines.gepa:GEPABaseline", GEPABaseline),
            "tf-grpo": ("seagym.baselines.tf_grpo:TFGRPOBaseline", TFGRPOBaseline),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (class_path, expected_type) in class_paths.items():
                config = {
                    "baseline": {
                        "name": name,
                        "class_path": class_path,
                        "config": {
                            "project_dir": f"reference/{name}",
                        },
                    }
                }
                built = build_baseline(config, run_dir=Path(tmp) / name, base_dir=Path.cwd())

                self.assertEqual(built.agent_id, name)
                self.assertIsInstance(built.baseline, expected_type)
                self.assertIsInstance(built.baseline, BaseBaseline)
                self.assertNotIsInstance(built.baseline, StaticBaseline)

    def test_native_baseline_initialization_exposes_prompt_template_state(self) -> None:
        class_paths = {
            "ace": "seagym.baselines.ace:ACEBaseline",
            "gepa": "seagym.baselines.gepa:GEPABaseline",
            "tf-grpo": "seagym.baselines.tf_grpo:TFGRPOBaseline",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, class_path in class_paths.items():
                built = build_baseline(
                    {
                        "baseline": {
                            "name": name,
                            "class_path": class_path,
                            "config": {
                                "project_dir": f"reference/{name}",
                            },
                        }
                    },
                    run_dir=Path(tmp) / name,
                    base_dir=Path.cwd(),
                )
                state = built.baseline.initialize(Path(tmp) / "run")
                self.assertIn("prompt_template_path", state.metadata, msg=name)
                self.assertTrue(Path(state.metadata["prompt_template_path"]).exists(), msg=name)

    def test_harbor_rollout_wraps_native_prompt_state_as_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ace",
                        "class_path": "seagym.baselines.ace:ACEBaseline",
                        "config": {
                            "project_dir": "reference/ace",
                        },
                    }
                },
                run_dir=Path(tmp) / "ace",
                base_dir=Path.cwd(),
            )
            state = built.baseline.initialize(Path(tmp) / "run")
            prompt_path = Path(state.metadata["prompt_template_path"])
            prompt_path.write_text("Learned {{ literal }} strategy", encoding="utf-8")
            rollout_agent = HarborRolloutAgent(agent_id="opencode")

            spec = rollout_agent.harbor_agent_spec(state)

            harbor_prompt_path = Path(spec.kwargs["prompt_template_path"])
            self.assertNotEqual(harbor_prompt_path, prompt_path)
            text = harbor_prompt_path.read_text(encoding="utf-8")
            self.assertIn("{% raw %}", text)
            self.assertIn("Learned {{ literal }} strategy", text)
            self.assertIn("{{ instruction }}", text)


    def test_rollout_agent_can_select_preinstalled_opencode_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_rollout_agent(
                {
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
                                "provider": "harbor",
                                "model": "deepseek/deepseek-v4-flash",
                            }
                        },
                    }
                },
                run_dir=Path(tmp) / "run",
                base_dir=Path.cwd(),
            )

            self.assertEqual(built.agent_id, "opencode")
            self.assertIsInstance(built.rollout_agent, HarborRolloutAgent)
            agent = built.rollout_agent
            assert isinstance(agent, HarborRolloutAgent)
            self.assertEqual(
                agent.agent_import_path,
                "seagym.rollout_agents.opencode_preinstalled:PreinstalledOpenCode",
            )
            self.assertEqual(built.rollout_model, "deepseek/deepseek-v4-flash")

    def test_native_baseline_checkpoint_reload_rebinds_prompt_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                (
                    "ace",
                    "seagym.baselines.ace:ACEBaseline",
                    ACEBaseline,
                    "reference/ace",
                    "ace_prompt.md",
                    "skillbook_path",
                ),
                (
                    "gepa",
                    "seagym.baselines.gepa:GEPABaseline",
                    GEPABaseline,
                    "reference/gepa",
                    "candidate.txt",
                    "candidate_path",
                ),
                (
                    "tf-grpo",
                    "seagym.baselines.tf_grpo:TFGRPOBaseline",
                    TFGRPOBaseline,
                    "reference/tf-grpo",
                    "tf_grpo_experiences.md",
                    "experience_path",
                ),
            ]
            for name, class_path, expected_cls, project_dir, prompt_filename, state_path_key in cases:
                with self.subTest(name=name):
                    built = build_baseline(
                        {
                            "baseline": {
                                "name": name,
                                "class_path": class_path,
                                "config": {
                                    "project_dir": project_dir,
                                },
                            }
                        },
                        run_dir=root / name,
                        base_dir=Path.cwd(),
                    )
                    baseline = built.baseline
                    self.assertIsInstance(baseline, expected_cls)
                    state = baseline.initialize(root / "run")
                    original_prompt = state.state_dir / prompt_filename
                    original_prompt.write_text(f"{name} checkpoint prompt", encoding="utf-8")
                    checkpoint = baseline.save_checkpoint(state, root / "checkpoints" / name / "A_0")
                    original_prompt.write_text(f"{name} stale source prompt", encoding="utf-8")

                    baseline.state_dir = root / "restored" / name
                    loaded = baseline.load_checkpoint(checkpoint)

                    loaded_prompt = Path(loaded.metadata["prompt_template_path"])
                    self.assertEqual(loaded_prompt, baseline.state_dir / prompt_filename)
                    self.assertEqual(loaded_prompt.read_text(encoding="utf-8"), f"{name} checkpoint prompt")
                    self.assertTrue(str(loaded.metadata[state_path_key]).startswith(str(baseline.state_dir)))
                    persisted = json.loads((baseline.state_dir / "baseline_state.json").read_text(encoding="utf-8"))
                    self.assertEqual(persisted["prompt_template_path"], str(loaded_prompt))
                    spec = HarborRolloutAgent(agent_id="opencode").harbor_agent_spec(loaded)
                    harbor_prompt_path = Path(spec.kwargs["prompt_template_path"])
                    self.assertTrue(str(harbor_prompt_path).startswith(str(baseline.state_dir)))
                    self.assertIn("{{ instruction }}", harbor_prompt_path.read_text(encoding="utf-8"))


    def test_tf_grpo_rollout_config_uses_train_group_attempts(self) -> None:
        config = {
            "rollout_agent": {
                "name": "opencode",
                "class_path": "seagym.rollout_agents.harbor:HarborRolloutAgent",
                "config": {
                    "agent": "opencode",
                    "n_attempts": 2,
                    "attempt_modes": ["train"],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            rollout_build = build_rollout_agent(config, run_dir=Path(tmp) / "run", base_dir=Path.cwd())

            self.assertIsInstance(rollout_build.rollout_agent, HarborRolloutAgent)
            agent = rollout_build.rollout_agent
            assert isinstance(agent, HarborRolloutAgent)
            self.assertEqual(agent.n_attempts, 2)
            self.assertEqual(agent.attempt_modes, ("train",))

    def test_builds_rollout_agents_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harbor = build_rollout_agent(
                {
                    "rollout_agent": {
                        "name": "codex",
                        "class_path": "seagym.rollout_agents.harbor:HarborRolloutAgent",
                        "config": {"agent": "codex"},
                    }
                },
                run_dir=Path(tmp),
                base_dir=Path(tmp),
            )
            ahe = build_rollout_agent(
                {
                    "rollout_agent": {
                        "name": "ahe-nexau",
                        "class_path": "seagym.rollout_agents.ahe_nexau:AHENexAURolloutAgent",
                        "config": {"agent": "ahe-nexau", "model": "deepseek-v4-flash"},
                    }
                },
                run_dir=Path(tmp),
                base_dir=Path(tmp),
            )

            self.assertIsInstance(harbor.rollout_agent, HarborRolloutAgent)
            self.assertIsInstance(ahe.rollout_agent, AHENexAURolloutAgent)

    def test_missing_class_path_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "baseline.class_path is required"):
                build_baseline({"baseline": {"name": "bad"}}, run_dir=Path(tmp) / "run", base_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
