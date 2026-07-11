from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seagym.baselines import BaselineState, build_baseline
from seagym.baselines.ahe import (
    AHEBaseline,
    _link_optional_trial_artifacts,
    _materialize_ahe_evidence,
    _materialize_ahe_trial,
    _run_ahe_post_update_hooks,
    _workspace_change_summary,
)
from seagym.baselines.data import Trajectory, TrajectoryBatch
from seagym.rollout_agents import build_rollout_agent
from seagym.rollout_agents.ahe_nexau import (
    AHENexAUHarborAgent,
    AHENexAURolloutAgent,
    _nexau_harbor_command,
    _nexau_install_template_path,
    _render_nexau_install_script,
    _stage_ahe_workspace_bundle,
)


AHE_REFERENCE_AGENT_DIR = Path("reference/agentic-harness-engineering/agents/code_agent_simple")


class AHEBaselineTest(unittest.TestCase):
    def test_builds_ahe_baseline_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "baseline": {
                    "name": "ahe",
                    "class_path": "seagym.baselines.ahe:AHEBaseline",
                    "config": {
                        "project_dir": "reference/agentic-harness-engineering",
                        "model": "deepseek-v4-flash",
                        "api_base": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "api_type": "openai_responses",
                        "reasoning": {"effort": "high", "summary": "detailed"},
                    },
                }
            }
            built = build_baseline(config, run_dir=Path(tmp) / "run", base_dir=Path.cwd())

            self.assertEqual(built.agent_id, "ahe")
            self.assertIsInstance(built.baseline, AHEBaseline)
            baseline = built.baseline
            assert isinstance(baseline, AHEBaseline)
            self.assertEqual(baseline.api_type, "openai_responses")
            self.assertEqual(baseline.reasoning, {"effort": "high", "summary": "detailed"})
            self.assertEqual(
                baseline._build_native_config()["llm"]["reasoning"],
                {"effort": "high", "summary": "detailed"},
            )

    def test_ahe_runtime_setup_config_is_loaded_from_baseline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "ahe",
                        "class_path": "seagym.baselines.ahe:AHEBaseline",
                        "config": {
                            "project_dir": "reference/agentic-harness-engineering",
                            "setup_commands": ["python -m pip install nexau"],
                            "setup_timeout_sec": 123,
                            "runtime_env": {"PYTHONPATH": "reference/agentic-harness-engineering"},
                        },
                    }
                },
                run_dir=Path(tmp) / "run",
                base_dir=Path.cwd(),
            )

            baseline = built.baseline
            self.assertIsInstance(baseline, AHEBaseline)
            assert isinstance(baseline, AHEBaseline)
            self.assertTrue(baseline.runtime.enabled)
            self.assertEqual(baseline.runtime.setup_commands, ["python -m pip install nexau"])
            self.assertEqual(baseline.runtime.setup_timeout_sec, 123)
            self.assertEqual(baseline.runtime.env["PYTHONPATH"], "reference/agentic-harness-engineering")


    def test_ahe_update_model_config_maps_deepseek_openai_and_glm(self) -> None:
        cases = [
            (
                {"provider": "deepseek", "model": "deepseek/deepseek-v4-flash", "api_key_env": "DEEPSEEK_API_KEY"},
                ("deepseek-v4-flash", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
            ),
            (
                {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
                ("gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
            ),
            (
                {
                    "provider": "openai_compatible",
                    "model": "glm-4-plus",
                    "api_base": "https://api.z.ai/api/coding/paas/v4",
                    "api_key_env": "GLM_API_KEY",
                },
                ("glm-4-plus", "https://api.z.ai/api/coding/paas/v4", "GLM_API_KEY"),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for raw_model, expected in cases:
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "ahe",
                            "class_path": "seagym.baselines.ahe:AHEBaseline",
                            "config": {
                                "project_dir": "reference/agentic-harness-engineering",
                                "update_model_ref": "update_model",
                            },
                            "models": {"update_model": raw_model},
                        }
                    },
                    run_dir=Path(tmp) / raw_model["provider"],
                    base_dir=Path.cwd(),
                )

                baseline = built.baseline
                self.assertIsInstance(baseline, AHEBaseline)
                assert isinstance(baseline, AHEBaseline)
                self.assertEqual((baseline.model, baseline.api_base, baseline.api_key_env), expected)
                native_config = baseline._build_native_config()
                self.assertEqual(native_config["llm"]["model"], expected[0])
                self.assertEqual(native_config["llm"]["base_url"], expected[1])


    def test_ahe_materializes_native_update_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "code_agent.yaml").write_text("type: agent\n", encoding="utf-8")
            batch = TrajectoryBatch(
                task_ids=["terminal-bench/task-one"],
                view_name="train",
                mode="train",
                trajectories=[
                    Trajectory(
                        task_id="terminal-bench/task-one",
                        attempt_id="task-one__abc123",
                        view_name="train",
                        mode="train",
                        success=False,
                        reward=0.0,
                        score=0.0,
                        rewards={"reward": 0.0},
                        error=None,
                        refs={
                            "harbor_task_name": "task-one",
                            "all_attempts": [
                                {
                                    "attempt_id": "task-one__attempt1",
                                    "success": False,
                                    "reward": 0.0,
                                    "score": 0.0,
                                    "rewards": {"reward": 0.0},
                                    "refs": {"harbor_task_name": "task-one", "trial_name": "task-one__attempt1"},
                                },
                                {
                                    "attempt_id": "task-one__attempt2",
                                    "success": True,
                                    "reward": 1.0,
                                    "score": 1.0,
                                    "rewards": {"reward": 1.0},
                                    "refs": {"harbor_task_name": "task-one", "trial_name": "task-one__attempt2"},
                                },
                            ],
                        },
                    )
                ],
            )

            evidence = _materialize_ahe_evidence(
                evolve=_FakeAHEEvolve(),
                trajectories=batch,
                exp_dir=root / "state",
                workspace=workspace,
                iteration=1,
                iteration_dir=root / "state" / "updates" / "ahe_iteration_0001",
            )

            job_dir = Path(evidence["job_dir"])
            self.assertTrue((job_dir / "task-one__attempt1" / "result.json").exists())
            self.assertTrue((job_dir / "task-one__attempt2" / "result.json").exists())
            self.assertEqual((job_dir / "task-one__attempt2" / "verifier" / "reward.txt").read_text().strip(), "1.0")
            self.assertEqual(evidence["k"], 2)
            self.assertTrue(Path(evidence["manifest"]).exists())
            self.assertIn("AHE native query", evidence["query"])


    def test_ahe_update_passes_absolute_native_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            code_agent = project / "agents" / "code_agent_simple"
            code_agent.mkdir(parents=True)
            (code_agent / "code_agent.yaml").write_text("llm_config: {}\n", encoding="utf-8")
            (project / "agents" / "evolve_agent").mkdir(parents=True)
            fake_evolve = _RecordingAHEEvolve()

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                baseline = AHEBaseline(baseline_id="ahe", state_dir=Path("relative-state"), project_dir=project)
                state = baseline.initialize(root / "run")
                batch = TrajectoryBatch(
                    task_ids=["terminal-bench/task-one"],
                    view_name="train",
                    mode="train",
                    trajectories=[
                        Trajectory(
                            task_id="terminal-bench/task-one",
                            attempt_id="task-one__trial",
                            view_name="train",
                            mode="train",
                            success=False,
                            reward=0.0,
                            score=0.0,
                            rewards={"reward": 0.0},
                            refs={"harbor_task_name": "task-one", "trial_name": "task-one__trial"},
                        )
                    ],
                )
                with patch("seagym.baselines.ahe.baseline._load_ahe_evolve", return_value=fake_evolve):
                    baseline.update(batch, state)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(fake_evolve.calls, 1)
            self.assertTrue(fake_evolve.exp_dir.is_absolute())
            self.assertTrue(fake_evolve.job_dir.is_absolute())
            self.assertTrue(fake_evolve.iteration_dir.is_absolute())
            self.assertEqual(fake_evolve.exp_dir, state.state_dir)
            self.assertEqual(fake_evolve.exp_dir, root.resolve() / "relative-state")


    def test_ahe_materialized_trial_symlink_resolves_source_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_trial = root / "source" / "task-one__trial"
            (source_trial / "agent").mkdir(parents=True)
            (source_trial / "agent" / "nexau_in_memory_tracer.cleaned.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            result_path = source_trial / "result.json"
            result_path.write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}) + "\n",
                encoding="utf-8",
            )
            job_dir = root / "state" / "updates" / "ahe_iteration_0001" / "input" / "benchmark" / "seagym_train_batch"
            job_dir.mkdir(parents=True)
            trajectory = Trajectory(
                task_id="terminal-bench/task-one",
                attempt_id="task-one__trial",
                view_name="train",
                mode="train",
                success=True,
                reward=1.0,
                score=1.0,
                rewards={"reward": 1.0},
                refs={
                    "harbor_task_name": "task-one",
                    "trial_name": "task-one__trial",
                    "result_path": str(result_path),
                },
            )

            manifest = _materialize_ahe_trial(job_dir, trajectory, index=1)

            trial_dir = job_dir / "task-one__trial"
            self.assertTrue(trial_dir.is_symlink())
            self.assertTrue((trial_dir / "result.json").exists())
            self.assertEqual(Path(os.readlink(trial_dir)), source_trial.resolve())
            self.assertEqual(manifest["source_trial_dir"], str(source_trial.resolve()))
            self.assertTrue(manifest["cleaned_trace_present"])


    def test_ahe_optional_trial_artifact_symlinks_resolve_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_trial = root / "source" / "task-one__trial"
            (source_trial / "agent").mkdir(parents=True)
            (source_trial / "agent" / "nexau_in_memory_tracer.cleaned.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            trial_dir = root / "state" / "updates" / "ahe_iteration_0001" / "input" / "benchmark" / "trial"
            trial_dir.mkdir(parents=True)

            _link_optional_trial_artifacts(source_trial, trial_dir)

            agent_link = trial_dir / "agent"
            self.assertTrue(agent_link.is_symlink())
            self.assertEqual(Path(os.readlink(agent_link)), (source_trial / "agent").resolve())
            self.assertTrue((agent_link / "nexau_in_memory_tracer.cleaned.json").exists())


    def test_ahe_workspace_change_detects_committed_update(self) -> None:
        before = {
            "head": "old-head",
            "tree": "old-tree",
            "status": [],
            "diff_hash": "empty",
        }
        after = {
            "head": "new-head",
            "tree": "new-tree",
            "status": [],
            "diff_hash": "empty",
        }

        summary = _workspace_change_summary(before, after)

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["status"], [])


    def test_ahe_post_update_hooks_archive_reference_artifacts(self) -> None:
        class FakeEvolve:
            def save_evolve_summary(self, iteration_dir: Path, iteration: int, result: str) -> None:
                evolve_dir = iteration_dir / "evolve"
                evolve_dir.mkdir(parents=True, exist_ok=True)
                (evolve_dir / "evolve_summary.md").write_text(
                    f"iteration={iteration}\n{result}\n",
                    encoding="utf-8",
                )

            def archive_change_manifest(self, exp_dir: Path, iteration: int) -> None:
                src = exp_dir / "change_manifest.json"
                dst = exp_dir / "updates" / f"ahe_iteration_{iteration:04d}" / "evolve" / "change_manifest.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "state"
            iteration_dir = exp_dir / "updates" / "ahe_iteration_0001"
            exp_dir.mkdir(parents=True)
            (exp_dir / "change_manifest.json").write_text('{"changes": []}\n', encoding="utf-8")

            artifacts = _run_ahe_post_update_hooks(
                evolve=FakeEvolve(),
                exp_dir=exp_dir,
                iteration=1,
                iteration_dir=iteration_dir,
                result="done",
            )

            self.assertTrue((iteration_dir / "evolve" / "evolve_summary.md").exists())
            self.assertTrue((iteration_dir / "evolve" / "change_manifest.json").exists())
            self.assertEqual(artifacts["evolve_summary"], str(iteration_dir / "evolve" / "evolve_summary.md"))
            self.assertEqual(artifacts["change_manifest"], str(iteration_dir / "evolve" / "change_manifest.json"))


    def test_ahe_rollout_staging_uploads_workspace_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            code_agent = project / "agents" / "code_agent_simple"
            code_agent.mkdir(parents=True)
            (code_agent / "code_agent.yaml").write_text("llm_config: {}\n", encoding="utf-8")
            state = root / "state"
            workspace = state / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "code_agent.yaml").write_text("llm_config: {model: evolved}\n", encoding="utf-8")
            (workspace / "start.py").write_text("print('hello')\n", encoding="utf-8")
            (workspace / ".git").mkdir()
            staging_root = root / "staging"
            agent_logs = root / "trial" / "agent"
            agent_logs.mkdir(parents=True)

            workspace_bundle = _stage_ahe_workspace_bundle(project, state, staging_root)

            self.assertTrue((workspace_bundle / "code_agent.yaml").exists())
            self.assertTrue((workspace_bundle / "start.py").exists())
            self.assertFalse((workspace_bundle / ".git").exists())
            self.assertEqual(list(agent_logs.iterdir()), [])


    def test_ahe_rollout_command_uses_reference_nexau_harbor_entrypoint(self) -> None:
        command = _nexau_harbor_command("solve this task")

        self.assertIn("/opt/nexau-venv/bin/nexau-harbor run", command)
        self.assertIn("--config_path /nexau-workspace/code_agent.yaml", command)
        self.assertIn("--log_dir_path /logs/agent", command)
        self.assertIn("| tee /logs/agent/nexau.txt", command)


    def test_ahe_rollout_uses_reference_task_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = BaselineState(
                root / "state",
                {
                    "project_dir": str(root / "reference"),
                    "agent_config_path": str(root / "state" / "workspace" / "code_agent.yaml"),
                },
            )
            agent = AHENexAURolloutAgent(agent_id="ahe-nexau")

            spec = agent.harbor_agent_spec(state)

            self.assertEqual(spec.kwargs["sandbox_work_dir"], ".")


    def test_ahe_rollout_model_config_maps_deepseek_openai_and_glm_env(self) -> None:
        cases = [
            (
                {"provider": "deepseek", "model": "deepseek/deepseek-v4-flash", "api_key_env": "DEEPSEEK_API_KEY"},
                ("deepseek-v4-flash", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
            ),
            (
                {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
                ("gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
            ),
            (
                {
                    "provider": "openai_compatible",
                    "model": "glm-4-plus",
                    "api_base": "https://api.z.ai/api/coding/paas/v4",
                    "api_key_env": "GLM_API_KEY",
                },
                ("glm-4-plus", "https://api.z.ai/api/coding/paas/v4", "GLM_API_KEY"),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = BaselineState(
                root / "state",
                {
                    "project_dir": str(root / "reference"),
                    "agent_config_path": str(root / "state" / "workspace" / "code_agent.yaml"),
                },
            )
            for raw_model, expected in cases:
                built = build_rollout_agent(
                    {
                        "rollout_agent": {
                            "name": "ahe-nexau",
                            "class_path": "seagym.rollout_agents.ahe_nexau:AHENexAURolloutAgent",
                            "config": {
                                "agent": "ahe-nexau",
                                "model_ref": "rollout_model",
                                "api_type": "openai_responses",
                                "reasoning": {"effort": "high", "summary": "detailed"},
                            },
                            "models": {"rollout_model": raw_model},
                        }
                    },
                    run_dir=root / raw_model["provider"],
                    base_dir=Path.cwd(),
                )

                agent = built.rollout_agent
                self.assertIsInstance(agent, AHENexAURolloutAgent)
                assert isinstance(agent, AHENexAURolloutAgent)
                self.assertEqual((agent.model, agent.api_base, agent.api_key_env), expected)
                self.assertEqual(agent.api_type, "openai_responses")
                self.assertEqual(agent.reasoning, {"effort": "high", "summary": "detailed"})
                spec = agent.harbor_agent_spec(state)
                self.assertEqual(spec.env["LLM_MODEL"], expected[0])
                self.assertEqual(spec.env["LLM_BASE_URL"], expected[1])
                self.assertEqual(spec.env["LLM_API_KEY"], os.environ.get(expected[2], ""))

    @unittest.skipUnless(AHE_REFERENCE_AGENT_DIR.exists(), "AHE reference submodule is not initialized")
    def test_ahe_rollout_setup_uses_reference_install_template(self) -> None:
        project_dir = Path("reference/agentic-harness-engineering")
        with patch.dict(os.environ, {"SEAGYM_AHE_USE_PREBUILT_E2B_TEMPLATE": ""}, clear=False):
            template = _nexau_install_template_path(project_dir)

        self.assertEqual(template.name, "install-nexau_saas_e2b.j2")
        self.assertTrue(template.exists())

        with tempfile.TemporaryDirectory() as tmp:
            script = _render_nexau_install_script(project_dir, Path(tmp) / "agent")

            text = script.read_text(encoding="utf-8")
            self.assertIn("uv venv $NEXAU_VENV", text)
            self.assertIn("NEXAU_VENV", text)
            self.assertIn("/opt/nexau-venv", text)
            self.assertTrue(script.exists())

    @unittest.skipUnless(AHE_REFERENCE_AGENT_DIR.exists(), "AHE reference submodule is not initialized")
    def test_ahe_rollout_setup_can_use_prebuilt_template(self) -> None:
        project_dir = Path("reference/agentic-harness-engineering")
        with patch.dict(os.environ, {"SEAGYM_AHE_USE_PREBUILT_E2B_TEMPLATE": "True"}, clear=False):
            template = _nexau_install_template_path(project_dir)

        self.assertEqual(template.name, "install-nexau.sh.j2")
        self.assertTrue(template.exists())

    def test_ahe_harbor_agent_preserves_base_extra_env_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = AHENexAUHarborAgent(
                logs_dir=Path(tmp),
                model_name="deepseek/deepseek-v4-flash",
                project_dir="reference/agentic-harness-engineering",
                state_dir=tmp,
                extra_env={"LLM_MODEL": "deepseek-v4-flash"},
            )

        self.assertEqual(agent.extra_env, {"LLM_MODEL": "deepseek-v4-flash"})

    @unittest.skipUnless(AHE_REFERENCE_AGENT_DIR.exists(), "AHE reference submodule is not initialized")
    def test_ahe_checkpoint_reload_rebinds_rollout_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = build_baseline(
                {
                    "baseline": {
                        "name": "agentic-harness-engineering",
                        "class_path": "seagym.baselines.ahe:AHEBaseline",
                        "config": {
                            "project_dir": "reference/agentic-harness-engineering",
                        },
                    }
                },
                run_dir=Path(tmp) / "ahe",
                base_dir=Path.cwd(),
            )
            baseline = built.baseline
            self.assertIsInstance(baseline, AHEBaseline)
            assert isinstance(baseline, AHEBaseline)
            state = baseline.initialize(Path(tmp) / "run")
            checkpoint = baseline.save_checkpoint(state, Path(tmp) / "checkpoints" / "initial")
            manifest = json.loads((checkpoint.checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertIn("agent_config_path", manifest["state_metadata"])

            baseline.state_dir = Path(tmp) / "restored_state"
            loaded = baseline.load_checkpoint(checkpoint)

            self.assertEqual(Path(loaded.metadata["workspace_dir"]), baseline.state_dir / "workspace")
            self.assertEqual(Path(loaded.metadata["agent_config_path"]), baseline.state_dir / "workspace" / "code_agent.yaml")
            spec = AHENexAURolloutAgent(agent_id="ahe-nexau").harbor_agent_spec(loaded)
            self.assertEqual(spec.kwargs["state_dir"], str(baseline.state_dir))


class _FakeAHEEvolve:
    def compute_stats(self, job_dir: Path, k: int = 1):
        return {
            "pass_rate": 0.0,
            "n_pass": 0,
            "n_fail": 1,
            "n_exception": 0,
            "n_total": 1,
            "k": k,
            "exception_types": {},
            "task_results": {"task-one": "fail"},
            "timeout_tasks": set(),
        }

    def update_task_history(self, exp_dir: Path, iteration: int, task_results: dict, per_task_rollouts=None):
        return {"task-one": [[iteration, "fail"]]}

    def compute_task_stability(self, task_history: dict):
        return {"stable_pass": [], "stable_fail": ["task-one"], "unstable": [], "possibly_unstable": [], "infra_only": []}

    def compute_iteration_diff(self, current_results, prev_results, current_rollouts=None, prev_rollouts=None):
        return None

    def update_best_ever(self, exp_dir: Path, iteration: int, stats: dict):
        return {"iteration": iteration, "pass_rate": stats["pass_rate"]}

    def update_history_before(self, exp_dir: Path, iteration: int, stats: dict, job_dir: Path, diff=None):
        return None

    def build_evolution_query(self, **kwargs):
        return f"AHE native query: {kwargs['job_dir']}"


class _RecordingAHEEvolve(_FakeAHEEvolve):
    def __init__(self) -> None:
        self.calls = 0
        self.exp_dir = Path()
        self.job_dir = Path()
        self.iteration_dir = Path()

    def run_evolve_agent(self, *, config, exp_dir: Path, iteration: int, query: str, job_dir: Path, iteration_dir: Path):
        del config, iteration, query
        self.calls += 1
        self.exp_dir = exp_dir
        self.job_dir = job_dir
        self.iteration_dir = iteration_dir
        return "done"



if __name__ == "__main__":
    unittest.main()
