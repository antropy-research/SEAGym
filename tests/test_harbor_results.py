from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import os
import json
import importlib.util
from types import SimpleNamespace

try:
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig
except ModuleNotFoundError:  # pragma: no cover - optional Harbor test dependency
    AgentFactory = None
    AgentConfig = None

from seagym.baselines import Checkpoint, PromptRefineBaseline, StaticBaseline, TrajectoryBatch
from seagym.envs.harbor_env import HarborEnv, normalize_harbor_trial_result
from seagym.envs.harbor_env.progress import run_harbor_with_progress
from seagym.rollout_agents.harbor import HarborRolloutAgent
try:
    from seagym.envs.harbor_env.e2b_runtime import (
        DEFAULT_E2B_SANDBOX_TIMEOUT_SEC,
        E2BOneHourEnvironment,
        _ahe_template_alias,
        _opencode_template_alias,
    )
except ModuleNotFoundError:  # pragma: no cover - optional Harbor/E2B test dependency
    DEFAULT_E2B_SANDBOX_TIMEOUT_SEC = 2400
    E2BOneHourEnvironment = None

    def _ahe_template_alias(environment_name: str) -> str:
        return environment_name.rsplit("/", 1)[-1].replace(".", "-")

    def _opencode_template_alias(environment_name: str) -> str:
        return f"seagym-opencode-{_ahe_template_alias(environment_name)}"
from seagym.utils import write_json
from seagym.data import TaskRecord
from seagym.envs import TaskRunResult
try:
    from tests.fixtures.agents.static_harbor_agent import StaticHarborAgent
except ModuleNotFoundError:  # pragma: no cover - optional Harbor test dependency
    StaticHarborAgent = None


STATIC_AGENT_IMPORT_PATH = "tests.fixtures.agents.static_harbor_agent:StaticHarborAgent"


class HarborResultsTest(unittest.TestCase):
    def test_harbor_progress_drains_large_stdout_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = run_harbor_with_progress(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stdout.write('x' * 200000); sys.stdout.flush(); "
                        "sys.stderr.write('y' * 200000); sys.stderr.flush()"
                    ),
                ],
                job_dir=Path(tmp) / "job",
                job_name="large-output",
                task_count=1,
                n_concurrent=1,
                view_name="validation",
                mode="validation",
                poll_interval_sec=0.01,
                status_interval_sec=0.01,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertLessEqual(len(completed.stdout), 20000)
        self.assertLessEqual(len(completed.stderr), 20000)
        self.assertTrue(completed.stdout)
        self.assertTrue(completed.stderr)

    def test_normalizes_trial_result_rewards_and_refs(self) -> None:
        task = TaskRecord.from_dict(
            {
                "task_id": "terminal-bench/demo",
                "source": {"type": "harbor", "dataset": "terminal-bench-2", "task_name": "demo"},
                "attributes": {"domain": "code"},
                "scoring": {
                    "main_reward_key": "reward",
                    "success_threshold": 1.0,
                    "score_transform": "binary_threshold",
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            write_json(
                result_path,
                {
                    "trial_name": "demo__abc",
                    "trial_uri": "file:///tmp/demo__abc",
                    "source": "terminal-bench-2",
                    "task_name": "terminal-bench/demo",
                    "task_checksum": "abc123",
                    "config": {"job_id": "job-1"},
                    "started_at": "2026-07-11T00:00:00Z",
                    "finished_at": "2026-07-11T00:01:05.250000Z",
                    "agent_result": {"n_input_tokens": 10, "cost_usd": 0.25},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": None,
                },
            )

            result = normalize_harbor_trial_result(
                task,
                result_path,
                view_name="update_validation",
                mode="validation",
                agent_id="oracle",
            )

            self.assertTrue(result.success)
            self.assertEqual(result.score, 1.0)
            self.assertEqual(result.rewards, {"reward": 1.0})
            self.assertEqual(result.cost["n_input_tokens"], 10.0)
            self.assertEqual(result.runtime_seconds, 65.25)
            self.assertEqual(result.refs["runtime_source"], "trial_elapsed")
            self.assertEqual(result.refs["finished_at"], "2026-07-11T00:01:05.250000Z")
            self.assertEqual(result.refs["job_id"], "job-1")
            self.assertEqual(result.refs["trial_name"], "demo__abc")

    def test_normalizes_trial_result_token_cost_from_nexau_trace_fallback(self) -> None:
        task = TaskRecord.from_dict(
            {
                "task_id": "terminal-bench/demo",
                "source": {"type": "harbor", "dataset": "terminal-bench-2", "task_name": "demo"},
                "attributes": {"domain": "code"},
                "scoring": {
                    "main_reward_key": "reward",
                    "success_threshold": 1.0,
                    "score_transform": "binary_threshold",
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp)
            result_path = trial_dir / "result.json"
            trace_path = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
            trace_path.parent.mkdir(parents=True)
            write_json(
                result_path,
                {
                    "trial_name": "demo__abc",
                    "config": {"job_id": "job-1"},
                    "agent_result": {
                        "n_input_tokens": None,
                        "n_cache_tokens": None,
                        "n_output_tokens": None,
                        "cost_usd": None,
                    },
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": None,
                },
            )
            write_json(trace_path, {"total_tokens": 76922, "calculated_total_cost": "N/A"})

            result = normalize_harbor_trial_result(
                task,
                result_path,
                view_name="train",
                mode="train",
                agent_id="ahe-nexau",
            )

            self.assertEqual(result.cost, {"total_tokens": 76922.0})
            self.assertEqual(result.refs["cost_source"], "nexau_cleaned_trace")
            self.assertEqual(result.refs["cost_path"], str(trace_path))

    def test_harbor_env_builds_batch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one"), _task(root, "terminal-bench/task-two")]
            env = HarborEnv(root / "jobs", harbor_bin="harbor", n_concurrent=2)

            command, job_name = env.build_batch_command(tasks, agent_id="oracle")

            self.assertEqual(command[:4], ["harbor", "run", "-p", str(root.resolve())])
            self.assertEqual(command.count("-i"), 2)
            self.assertIn("task-one", command)
            self.assertIn("task-two", command)
            self.assertIn("-l", command)
            self.assertEqual(command[command.index("-l") + 1], "2")
            self.assertIn("-n", command)
            self.assertEqual(command[command.index("-n") + 1], "2")
            self.assertTrue(job_name.startswith("seagym-batch-task-one-2-"))

    def test_harbor_env_writes_ordered_task_configs_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            second = _task(root, "terminal-bench/task-two")
            first = _task(root, "terminal-bench/task-one")
            env = HarborEnv(root / "jobs", harbor_bin="harbor", n_concurrent=2, preserve_task_order=True)

            command, _ = env.build_batch_command([second, first], agent_id="oracle")

            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["datasets"], [])
            self.assertEqual(
                [Path(task["path"]).name for task in config["tasks"]],
                ["task-two", "task-one"],
            )

    def test_harbor_env_builds_n_attempts_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(root / "jobs", harbor_bin="harbor", n_concurrent=2)
            env.n_attempts = 5

            command, _ = env.build_batch_command(tasks, agent_id="oracle")

            self.assertIn("-k", command)
            self.assertEqual(command[command.index("-k") + 1], "5")

    def test_harbor_env_patches_agent_and_verifier_timeouts_in_selected_dataset_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one"), _task(root, "terminal-bench/task-two")]
            for task_name in ("task-one", "task-two"):
                (root / task_name / "task.toml").write_text(
                    "[environment]\ndocker_image = \"ubuntu:22.04\"\n\n"
                    "[agent]\ntimeout_sec = 900.0\n\n"
                    "[verifier]\ntimeout_sec = 1800.0\n",
                    encoding="utf-8",
                )
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                n_concurrent=2,
                agent_override_timeout_sec=3600,
                verifier_override_timeout_sec=600,
            )

            command, _ = env.build_batch_command(tasks, agent_id="oracle")

            patched_root = Path(command[command.index("-p") + 1])
            self.assertNotEqual(patched_root, root)
            task_one_text = (patched_root / "task-one" / "task.toml").read_text()
            task_two_text = (patched_root / "task-two" / "task.toml").read_text()
            self.assertRegex(task_one_text, r"(?ms)^\[agent\].*timeout_sec = 3600\.0")
            self.assertRegex(task_one_text, r"(?ms)^\[verifier\].*timeout_sec = 600\.0")
            self.assertRegex(task_two_text, r"(?ms)^\[agent\].*timeout_sec = 3600\.0")
            self.assertRegex(task_two_text, r"(?ms)^\[verifier\].*timeout_sec = 600\.0")

    def test_harbor_env_builds_single_config_job_for_mixed_dataset_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_a = root / "dataset-a"
            group_b = root / "dataset-b"
            group_a.mkdir()
            group_b.mkdir()
            tasks = [
                *[_task(group_a, f"terminal-bench/a-{index}") for index in range(10)],
                *[_task(group_b, f"terminal-bench/b-{index}") for index in range(10)],
            ]
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                n_concurrent=16,
                agent_override_timeout_sec=1800,
                verifier_override_timeout_sec=600,
            )

            command, job_name = env.build_batch_command(tasks, agent_id="oracle")

            self.assertIn("--config", command)
            self.assertIn("-n", command)
            self.assertEqual(command[command.index("-n") + 1], "16")
            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["job_name"], job_name)
            self.assertEqual(config["n_concurrent_trials"], 16)
            self.assertEqual(config["agents"][0]["override_timeout_sec"], 1800.0)
            self.assertEqual(config["verifier"]["override_timeout_sec"], 600.0)
            self.assertEqual(
                sorted((Path(dataset["path"]).name, len(dataset["task_names"])) for dataset in config["datasets"]),
                [("dataset-a", 10), ("dataset-b", 10)],
            )

    def test_ahe_template_alias_matches_official_builder_alias(self) -> None:
        self.assertEqual(
            _ahe_template_alias("terminal-bench/install-windows-3-11"),
            "install-windows-3-11",
        )
        self.assertEqual(_ahe_template_alias("terminal-bench/foo.bar"), "foo-bar")

    def test_opencode_template_alias_is_prefixed_from_task_name(self) -> None:
        self.assertEqual(
            _opencode_template_alias("terminal-bench/install-windows-3-11"),
            "seagym-opencode-install-windows-3-11",
        )
        self.assertEqual(_opencode_template_alias("swe-bench/foo.bar"), "seagym-opencode-foo-bar")

    @unittest.skipIf(E2BOneHourEnvironment is None, "Harbor/E2B dependencies are not installed")
    def test_e2b_runtime_sandbox_timeout_uses_seagym_cap(self) -> None:
        env = object.__new__(E2BOneHourEnvironment)
        env.environment_name = "terminal-bench/task-one"
        env.session_id = "session-1"
        env._template_name = "task-one"
        env._seagym_sandbox_timeout_sec = DEFAULT_E2B_SANDBOX_TIMEOUT_SEC
        env._network_policy = SimpleNamespace(network_mode="public")
        env._sandbox_create_network_options = lambda: None
        create = AsyncMock(return_value="sandbox")

        with patch("e2b.AsyncSandbox.create", create):
            asyncio.run(env._create_sandbox())

        self.assertEqual(env._sandbox, "sandbox")
        self.assertEqual(create.call_args.kwargs["timeout"], 2400)
        self.assertEqual(create.call_args.kwargs["template"], "task-one")
        self.assertEqual(create.call_args.kwargs["allow_internet_access"], True)

    @unittest.skipIf(E2BOneHourEnvironment is None, "Harbor/E2B dependencies are not installed")
    def test_e2b_runtime_sandbox_timeout_can_be_overridden(self) -> None:
        env = object.__new__(E2BOneHourEnvironment)
        env.environment_name = "terminal-bench/task-one"
        env.session_id = "session-1"
        env._template_name = "task-one"
        env._seagym_sandbox_timeout_sec = 1800
        env._network_policy = SimpleNamespace(network_mode="public")
        env._sandbox_create_network_options = lambda: None
        create = AsyncMock(return_value="sandbox")

        with patch("e2b.AsyncSandbox.create", create):
            asyncio.run(env._create_sandbox())

        self.assertEqual(create.call_args.kwargs["timeout"], 1800)

    def test_harbor_env_builds_custom_agent_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                agent_import_path=STATIC_AGENT_IMPORT_PATH,
                agent_kwargs={
                    "run_command": "true",
                    "state_dir": "/tmp/seagym-state",
                    "enabled": True,
                    "options": {"mode": "smoke"},
                },
                agent_env={"SEAGYM_STATIC": "1"},
            )

            command, _ = env.build_batch_command(tasks, agent_id="static-baseline")

            self.assertIn("--agent-import-path", command)
            self.assertIn(STATIC_AGENT_IMPORT_PATH, command)
            self.assertNotIn("-a", command)
            self.assertEqual(command.count("--agent-kwarg"), 4)
            self.assertIn("run_command=true", command)
            self.assertIn("state_dir=/tmp/seagym-state", command)
            self.assertIn("enabled=true", command)
            self.assertIn('options={"mode": "smoke"}', command)
            self.assertIn("--agent-env", command)
            self.assertIn("SEAGYM_STATIC=1", command)

    def test_harbor_env_builds_model_and_agent_env_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                model_name="gpt-5.3-codex",
                agent_env={"CODEX_FORCE_AUTH_JSON": "true"},
                verifier_env={"HTTP_PROXY": "http://192.168.5.2:7890"},
                extra_args=["--agent-timeout-multiplier", "2"],
            )

            command, _ = env.build_batch_command(tasks, agent_id="codex")

            self.assertIn("-a", command)
            self.assertEqual(command[command.index("-a") + 1], "codex")
            self.assertIn("--model", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-5.3-codex")
            self.assertIn("--agent-env", command)
            self.assertIn("CODEX_FORCE_AUTH_JSON=true", command)
            self.assertIn("--verifier-env", command)
            self.assertIn("HTTP_PROXY=http://192.168.5.2:7890", command)
            self.assertIn("--agent-timeout-multiplier", command)

    def test_harbor_env_strips_proxy_env_for_e2b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                env="e2b",
                agent_env={
                    "HTTP_PROXY": "http://192.168.5.2:7890",
                    "CUSTOM_AGENT_ENV": "1",
                },
                verifier_env={
                    "HTTPS_PROXY": "http://192.168.5.2:7890",
                    "CUSTOM_VERIFIER_ENV": "1",
                },
            )

            command, _ = env.build_batch_command(tasks, agent_id="oracle")

            self.assertIn("CUSTOM_AGENT_ENV=1", command)
            self.assertIn("CUSTOM_VERIFIER_ENV=1", command)
            self.assertNotIn("HTTP_PROXY=http://192.168.5.2:7890", command)
            self.assertNotIn("HTTPS_PROXY=http://192.168.5.2:7890", command)

    def test_harbor_env_builds_non_docker_environment_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(root / "jobs", harbor_bin="harbor", env="daytona")

            command, _ = env.build_batch_command(tasks, agent_id="oracle")

            self.assertIn("-e", command)
            self.assertEqual(command[command.index("-e") + 1], "daytona")

    def test_harbor_env_builds_custom_environment_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(
                root / "jobs",
                harbor_bin="harbor",
                env="docker",
                environment_import_path="my_envs:CustomEnvironment",
                environment_kwargs={"workspace_size": 4, "prebuild": True},
            )

            command, _ = env.build_batch_command(tasks, agent_id="oracle")

            self.assertNotIn("--environment-import-path", command)
            self.assertIn("-e", command)
            self.assertEqual(command[command.index("-e") + 1], "my_envs:CustomEnvironment")
            self.assertEqual(command.count("--environment-kwarg"), 2)
            self.assertIn("workspace_size=4", command)
            self.assertIn("prebuild=true", command)

    def test_harbor_env_batch_preserves_single_local_path_command_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task-one"
            task_dir.mkdir()
            task = TaskRecord.from_dict(
                {
                    "task_id": "terminal-bench/task-one",
                    "source": {
                        "type": "harbor",
                        "dataset": "terminal-bench-2",
                        "task_name": "task-one",
                        "local_path": str(task_dir),
                    },
                    "attributes": {"domain": "code"},
                }
            )
            env = HarborEnv(root / "jobs", harbor_bin="harbor", n_concurrent=1)

            command, _ = env.build_batch_command([task], agent_id="oracle")

            self.assertEqual(command[:4], ["harbor", "run", "-p", str(task_dir.resolve())])
            self.assertNotIn("-i", command)
            self.assertNotIn("-l", command)

    def test_harbor_env_batch_maps_trials_and_marks_missing_task_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            tasks = [_task(root, "terminal-bench/task-one"), _task(root, "terminal-bench/task-two")]
            env = HarborEnv(jobs_dir, harbor_bin="harbor", n_concurrent=2)

            class FakeProcess:
                def __init__(self, command):
                    self.command = command
                    self.returncode = None

                def poll(self):
                    if self.returncode is None:
                        self.returncode = 0
                    return self.returncode

                def communicate(self):
                    job_name = self.command[self.command.index("--job-name") + 1]
                    result_path = jobs_dir / job_name / "task-one-trial" / "result.json"
                    write_json(
                        result_path,
                        {
                            "trial_name": "task-one-trial",
                            "trial_uri": f"file://{result_path.parent}",
                            "source": "terminal-bench-2",
                            "task_name": "terminal-bench/task-one",
                            "task_checksum": "abc123",
                            "config": {"job_id": job_name},
                            "agent_result": {"n_input_tokens": 5},
                            "verifier_result": {"rewards": {"reward": 1.0}},
                            "exception_info": None,
                        },
                    )
                    return "ok", ""

            def fake_popen(command, stdout, stderr, text):
                job_name = command[command.index("--job-name") + 1]
                write_json(
                    jobs_dir / job_name / "result.json",
                    {
                        "stats": {
                            "n_completed_trials": 1,
                            "n_running_trials": 0,
                            "n_pending_trials": 1,
                            "n_errored_trials": 0,
                            "n_cancelled_trials": 0,
                            "n_retries": 0,
                        }
                    },
                )
                return FakeProcess(command)

            with patch("seagym.envs.harbor_env.progress.subprocess.Popen", side_effect=fake_popen):
                results = env.run_tasks(
                    tasks,
                    view_name="validation_batch",
                    mode="validation",
                    agent_id="oracle",
                )

            self.assertEqual([result.task_id for result in results], ["terminal-bench/task-one", "terminal-bench/task-two"])
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].score, 1.0)
            self.assertIn("job_dir", results[0].refs)
            self.assertFalse(results[1].success)
            self.assertIn("no trial result matched", results[1].error or "")

    def test_harbor_env_parses_landed_results_when_cli_summary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            tasks = [_task(root, "hle/hle__abc123")]
            env = HarborEnv(jobs_dir, harbor_bin="harbor", n_concurrent=1)

            class FakeProcess:
                def __init__(self, command):
                    self.command = command
                    self.returncode = None

                def poll(self):
                    if self.returncode is None:
                        self.returncode = 1
                    return self.returncode

                def communicate(self):
                    job_name = self.command[self.command.index("--job-name") + 1]
                    result_path = jobs_dir / job_name / "hle__abc123__trial" / "result.json"
                    write_json(
                        result_path,
                        {
                            "trial_name": "hle__abc123__trial",
                            "trial_uri": f"file://{result_path.parent}",
                            "source": "hle",
                            "task_name": "hle__abc123",
                            "task_checksum": "abc123",
                            "config": {"job_id": job_name},
                            "agent_result": {"n_input_tokens": 5},
                            "verifier_result": {"rewards": {"reward": 1.0}},
                            "exception_info": None,
                        },
                    )
                    return "20/20 Mean: 0.300", "ValueError: too many values to unpack"

            def fake_popen(command, stdout, stderr, text):
                job_name = command[command.index("--job-name") + 1]
                write_json(
                    jobs_dir / job_name / "result.json",
                    {
                        "stats": {
                            "n_completed_trials": 1,
                            "n_running_trials": 0,
                            "n_pending_trials": 0,
                            "n_errored_trials": 0,
                            "n_cancelled_trials": 0,
                            "n_retries": 0,
                        }
                    },
                )
                return FakeProcess(command)

            with patch("seagym.envs.harbor_env.progress.subprocess.Popen", side_effect=fake_popen):
                results = env.run_tasks(
                    tasks,
                    view_name="train",
                    mode="train",
                    agent_id="oracle",
                )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].score, 1.0)
            self.assertIsNone(results[0].error)
            self.assertEqual(results[0].refs["harbor_returncode"], 1)
            self.assertIn("harbor_warning", results[0].refs)
            self.assertIn("too many values", results[0].refs["harbor_warning"])

    def test_harbor_env_returns_all_attempts_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            tasks = [_task(root, "terminal-bench/task-one")]
            env = HarborEnv(jobs_dir, harbor_bin="harbor", n_concurrent=2)
            env.n_attempts = 2

            class FakeProcess:
                def __init__(self, command):
                    self.command = command
                    self.returncode = None

                def poll(self):
                    if self.returncode is None:
                        self.returncode = 0
                    return self.returncode

                def communicate(self):
                    job_name = self.command[self.command.index("--job-name") + 1]
                    for idx, reward in enumerate((0.0, 1.0)):
                        result_path = jobs_dir / job_name / f"task-one-trial-{idx}" / "result.json"
                        write_json(
                            result_path,
                            {
                                "trial_name": f"task-one-trial-{idx}",
                                "trial_uri": f"file://{result_path.parent}",
                                "source": "terminal-bench-2",
                                "task_name": "terminal-bench/task-one",
                                "task_checksum": "abc123",
                                "config": {"job_id": job_name},
                                "agent_result": {"n_input_tokens": 5 + idx},
                                "verifier_result": {"rewards": {"reward": reward}},
                                "exception_info": None,
                            },
                        )
                    return "ok", ""

            def fake_popen(command, stdout, stderr, text):
                job_name = command[command.index("--job-name") + 1]
                write_json(
                    jobs_dir / job_name / "result.json",
                    {
                        "stats": {
                            "n_completed_trials": 2,
                            "n_running_trials": 0,
                            "n_pending_trials": 0,
                            "n_errored_trials": 0,
                            "n_cancelled_trials": 0,
                            "n_retries": 0,
                        }
                    },
                )
                return FakeProcess(command)

            with patch("seagym.envs.harbor_env.progress.subprocess.Popen", side_effect=fake_popen):
                results = env.run_task_attempts(
                    tasks,
                    view_name="train",
                    mode="train",
                    agent_id="oracle",
                )

            self.assertEqual(len(results), 2)
            self.assertEqual([result.refs["attempt_index"] for result in results], [0, 1])
            self.assertEqual([result.score for result in results], [0.0, 1.0])

    def test_static_baseline_noop_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            baseline = StaticBaseline(baseline_id="static", state_dir=state_dir)
            state = baseline.initialize(root)
            batch = TrajectoryBatch.from_task_results(
                [
                    _task_result(
                        "task-one",
                        view_name="train",
                        mode="train",
                    )
                ],
                task_ids=["task-one"],
                view_name="train",
                mode="train",
            )

            update = baseline.update(batch, state)
            snapshot = baseline.save_checkpoint(state, root / "checkpoints" / "initial")
            loaded = baseline.load_checkpoint(Checkpoint(root / "checkpoints" / "initial"))

            self.assertFalse(update.changed)
            self.assertFalse(snapshot.metadata["changed"] if "changed" in snapshot.metadata else False)
            self.assertTrue((root / "checkpoints" / "initial" / "checkpoint.json").exists())
            self.assertEqual(loaded.metadata["baseline_id"], "static")

    def test_harbor_rollout_agent_exposes_agent_spec(self) -> None:
        rollout_agent = HarborRolloutAgent(
            agent_id="static-baseline",
            agent_import_path=STATIC_AGENT_IMPORT_PATH,
            agent_kwargs={"run_command": "true", "state_dir": "/tmp/state"},
            agent_env={"SEAGYM_STATIC": "1"},
            n_attempts=5,
        )

        spec = rollout_agent.harbor_agent_spec()

        self.assertEqual(spec.agent_id, "static-baseline")
        self.assertEqual(spec.import_path, STATIC_AGENT_IMPORT_PATH)
        self.assertEqual(spec.kwargs["run_command"], "true")
        self.assertEqual(spec.kwargs["state_dir"], "/tmp/state")
        self.assertEqual(spec.env, {"SEAGYM_STATIC": "1"})
        self.assertEqual(spec.n_attempts, 5)

    @unittest.skipIf(AgentFactory is None or StaticHarborAgent is None, "harbor package is not installed")
    def test_static_harbor_agent_import_path_instantiates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgentFactory.create_agent_from_config(
                AgentConfig(
                    import_path=STATIC_AGENT_IMPORT_PATH,
                    kwargs={"run_command": "true", "state_dir": "/tmp/state"},
                    env={"SEAGYM_STATIC": "1"},
                ),
                logs_dir=Path(tmp),
            )

            self.assertIsInstance(agent, StaticHarborAgent)
            self.assertEqual(agent.name(), "seagym-static")
            self.assertEqual(agent.run_command, "true")
            self.assertEqual(agent.state_dir, "/tmp/state")
            self.assertEqual(agent.extra_env, {"SEAGYM_STATIC": "1"})

    def test_prompt_refine_baseline_updates_prompt_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = PromptRefineBaseline(
                baseline_id="prompt-refine",
                state_dir=Path(tmp),
            )
            state = baseline.initialize(Path(tmp))

            baseline._refine_prompt = lambda prompt_input: "Always inspect verifier-visible outputs first."
            update = baseline.update(
                TrajectoryBatch.from_task_results(
                    [_task_result("task-one", score=0.0, success=False, error="missing output")],
                    task_ids=["task-one"],
                    view_name="train",
                    mode="train",
                ),
                state,
            )

            self.assertTrue(update.changed)
            self.assertEqual(update.status, "updated")
            self.assertIn("Always inspect", baseline.prompt)
            self.assertIn("{{ instruction }}", baseline.prompt_template_path.read_text(encoding="utf-8"))
            self.assertTrue(baseline.history_path.exists())

    def test_prompt_refine_baseline_checkpoint_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = PromptRefineBaseline(baseline_id="prompt-refine", state_dir=root / "state")
            state = baseline.initialize(root)
            baseline.prompt = "checkpoint prompt"
            baseline._write_state()

            checkpoint = baseline.save_checkpoint(state, root / "checkpoints" / "initial")
            baseline.prompt = "mutated prompt"
            baseline._write_state()
            loaded = baseline.load_checkpoint(Checkpoint(root / "checkpoints" / "initial"))

            self.assertEqual(checkpoint.metadata["type"], "prompt_refine_checkpoint")
            self.assertTrue(loaded.metadata["loaded"])
            self.assertEqual(baseline.prompt, "checkpoint prompt")

    def test_prompt_refine_baseline_exposes_builtin_llm_agent_with_prompt_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = PromptRefineBaseline(
                baseline_id="prompt-refine",
                state_dir=Path(tmp),
            )
            state = baseline.initialize(Path(tmp))
            rollout_agent = HarborRolloutAgent(
                agent_id="codex",
                agent_kwargs={"reasoning_effort": "low"},
                agent_env={"OPENAI_API_KEY": "env:OPENAI_API_KEY"},
            )

            spec = rollout_agent.harbor_agent_spec(state)

            self.assertEqual(spec.agent_id, "codex")
            self.assertIsNone(spec.import_path)
            self.assertEqual(spec.kwargs["reasoning_effort"], "low")
            self.assertEqual(spec.kwargs["prompt_template_path"], str(baseline.prompt_template_path))
            self.assertEqual(spec.env, {"OPENAI_API_KEY": "env:OPENAI_API_KEY"})

    @unittest.skipIf(importlib.util.find_spec("openai") is None, "openai package is not installed")
    def test_prompt_refine_baseline_supports_deepseek_openai_compatible_refiner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from seagym.models import ModelConfig

            baseline = PromptRefineBaseline(
                baseline_id="prompt-refine",
                state_dir=Path(tmp),
                refiner_model=ModelConfig(
                    name="deepseek-v4-pro",
                    provider="openai_compatible",
                    api_base="https://api.deepseek.com",
                    api_key_env="DEEPSEEK_API_KEY",
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}},
                ),
            )
            captured = {}

            class FakeCompletions:
                def create(self, **kwargs):
                    captured.update(kwargs)
                    return {
                        "choices": [
                            {"message": {"content": "Refined DeepSeek prompt."}}
                        ]
                    }

            class FakeClient:
                def __init__(self, api_key, base_url):
                    captured["api_key"] = api_key
                    captured["base_url"] = base_url
                    self.chat = type("Chat", (), {"completions": FakeCompletions()})()

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                with patch("openai.OpenAI", FakeClient):
                    refined = baseline._refine_prompt("input")

            self.assertEqual(refined, "Refined DeepSeek prompt.")
            self.assertEqual(captured["api_key"], "test-key")
            self.assertEqual(captured["base_url"], "https://api.deepseek.com")
            self.assertEqual(captured["model"], "deepseek-v4-pro")
            self.assertEqual(captured["reasoning_effort"], "high")
            self.assertEqual(captured["extra_body"], {"thinking": {"type": "enabled"}})



def _task(root: Path, task_id: str) -> TaskRecord:
    local_name = task_id.rsplit("/", 1)[-1]
    task_dir = root / local_name
    task_dir.mkdir(exist_ok=True)
    return TaskRecord.from_dict(
        {
            "task_id": task_id,
            "source": {
                "type": "harbor",
                "dataset": "terminal-bench-2",
                "dataset_path": str(root),
                "task_name": local_name,
                "registry_task_name": task_id,
                "local_path": str(task_dir),
            },
            "attributes": {"domain": "code"},
            "scoring": {
                "main_reward_key": "reward",
                "success_threshold": 1.0,
                "score_transform": "binary_threshold",
            },
        }
    )


def _task_result(
    task_id: str,
    *,
    view_name: str = "train",
    mode: str = "train",
    score: float = 1.0,
    success: bool = True,
    error: str | None = None,
) -> TaskRunResult:
    return TaskRunResult(
        task_id=task_id,
        view_name=view_name,
        mode=mode,
        rewards={"reward": score},
        score=score,
        success=success,
        error=error,
    )


def _harbor_artifact_row(root: Path) -> dict:
    trial_dir = root / "job" / "django__django-13212__trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    write_json(
        trial_dir / "result.json",
        {
            "agent_result": {"n_input_tokens": 1},
            "verifier_result": {"rewards": {"reward": 0.0}},
        },
    )
    write_json(
        trial_dir / "agent" / "trajectory.json",
        {
            "steps": [
                {
                    "source": "agent",
                    "message": "(tool use)",
                    "reasoning_content": "I inspected validators.py.",
                    "tool_calls": [{"function_name": "task"}],
                },
                {
                    "source": "agent",
                    "message": "I fixed char field but missed DecimalField and FileField cases.",
                },
            ]
        },
    )
    (trial_dir / "agent" / "opencode.txt").write_text("agent transcript\n", encoding="utf-8")
    write_json(
        trial_dir / "verifier" / "report.json",
        {
            "django__django-13212": {
                "resolved": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["test_char_field"],
                        "failure": ["test_decimal_field", "test_file_field"],
                    }
                },
            }
        },
    )
    (trial_dir / "verifier" / "test-stdout.txt").write_text(
        "DecimalField failure traceback\nFileField failure traceback\n",
        encoding="utf-8",
    )
    return {
        "task_id": "django__django-13212",
        "instruction": "Fix validators.",
        "view_name": "train",
        "mode": "train",
        "success": False,
        "score": 0.0,
        "error": None,
        "rewards": {"reward": 0.0},
        "refs": {"result_path": str(trial_dir / "result.json")},
    }


if __name__ == "__main__":
    unittest.main()
