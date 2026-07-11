from __future__ import annotations

import builtins
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seagym.runtime import inspect_experiment_config, load_env_file, run_container_network_checks, run_runtime_checks


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_top_level_import_does_not_load_external_adapters(self) -> None:
        for name in [
            "seagym.baselines.ace",
            "seagym.baselines.ahe",
            "seagym.baselines.gepa",
            "seagym.baselines.tf_grpo",
            "seagym.rollout_agents.ahe_nexau",
            "seagym.rollout_agents.opencode_preinstalled",
            "seagym.envs.harbor_env",
        ]:
            sys.modules.pop(name, None)

        importlib.import_module("seagym")

        for name in [
            "seagym.baselines.ace",
            "seagym.baselines.ahe",
            "seagym.baselines.gepa",
            "seagym.baselines.tf_grpo",
            "seagym.rollout_agents.ahe_nexau",
            "seagym.rollout_agents.opencode_preinstalled",
            "seagym.envs.harbor_env",
        ]:
            self.assertNotIn(name, sys.modules)

    def test_load_env_file_sets_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# local secrets",
                        "OPENAI_API_KEY='test-key'",
                        'DEEPSEEK_BASE_URL="https://api.deepseek.com"',
                        "export SEAGYM_HARBOR_BIN=harbor",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(env_file)

                self.assertEqual(loaded["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["DEEPSEEK_BASE_URL"], "https://api.deepseek.com")
                self.assertEqual(os.environ["SEAGYM_HARBOR_BIN"], "harbor")

    def test_runtime_checks_can_run_without_harbor_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "DEEPSEEK_API_KEY=secret\nHTTPS_PROXY=http://user:pass@127.0.0.1:7890\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                report = run_runtime_checks(
                    env_file=env_file,
                    key_env_names=["DEEPSEEK_API_KEY"],
                    proxy_env_names=["HTTPS_PROXY"],
                    check_harbor=False,
                )

            self.assertTrue(report.ok)
            checks = {check.name: check for check in report.checks}
            self.assertEqual(checks["env:DEEPSEEK_API_KEY"].status, "ok")
            self.assertEqual(checks["proxy:HTTPS_PROXY"].detail, "http://***@127.0.0.1:7890")

    def test_runtime_checks_uses_seagym_harbor_env_without_legacy_fallback(self) -> None:
        def fake_which(name: str):
            return f"/usr/bin/{name}" if name in {"harbor", "seagym-harbor"} else None

        legacy_harbor_bin = "SE" + "BENCH_HARBOR_BIN"
        with patch.dict(
            os.environ,
            {
                legacy_harbor_bin: "legacy-harbor",
                "SEAGYM_HARBOR_BIN": "seagym-harbor",
            },
            clear=True,
        ):
            with patch("seagym.runtime.checks.shutil.which", side_effect=fake_which):
                with patch("seagym.runtime.checks._check_command"):
                    report = run_runtime_checks(
                        load_env=False,
                        key_env_names=[],
                        proxy_env_names=[],
                        check_harbor=True,
                    )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["harbor_bin"].detail, "seagym-harbor")
        self.assertEqual(by_name["harbor_bin"].status, "ok")

        with patch.dict(os.environ, {legacy_harbor_bin: "legacy-harbor"}, clear=True):
            with patch("seagym.runtime.checks.shutil.which", side_effect=fake_which):
                with patch("seagym.runtime.checks._check_command"):
                    report = run_runtime_checks(
                        load_env=False,
                        key_env_names=[],
                        proxy_env_names=[],
                        check_harbor=True,
                    )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["harbor_bin"].detail, "harbor")
        self.assertEqual(by_name["harbor_bin"].status, "ok")

    def test_missing_env_file_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_runtime_checks(
                env_file=Path(tmp) / ".env",
                key_env_names=[],
                proxy_env_names=[],
                check_harbor=False,
            )

            self.assertTrue(report.ok)
            self.assertEqual(len(report.checks), 1)
            self.assertEqual(report.checks[0].name, "env_file")
            self.assertEqual(report.checks[0].status, "warn")

    def test_config_runtime_check_warns_for_localhost_container_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                """
{
  "backend": {
    "name": "harbor",
    "env": "docker",
    "container_env": {
      "HTTP_PROXY": "http://127.0.0.1:7890"
    }
  },
  "output": {
    "run_dir": "/tmp/seagym_bad_run"
  }
}
""",
                encoding="utf-8",
            )

            checks = inspect_experiment_config(config_path)
            by_name = {check.name: check for check in checks}

            self.assertEqual(by_name["config:run_dir"].status, "warn")
            self.assertEqual(by_name["config:harbor.container_env"].status, "warn")
            self.assertIn("localhost", by_name["config:harbor.container_env"].detail)

    def test_config_runtime_check_accepts_results_run_dir_and_container_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            run_dir = (Path.cwd() / "results" / "runs" / "runtime_check_ok").resolve()
            config_path.write_text(
                f"""
{{
  "backend": {{
    "name": "harbor",
    "env": "docker",
    "container_env": {{
      "HTTP_PROXY": "${{SEAGYM_CONTAINER_HTTP_PROXY}}"
    }}
  }},
  "output": {{
    "run_dir": "{run_dir}"
  }}
}}
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"SEAGYM_CONTAINER_HTTP_PROXY": "http://192.168.5.2:7890"}):
                checks = inspect_experiment_config(config_path)
            by_name = {check.name: check for check in checks}

            self.assertEqual(by_name["config:run_dir"].status, "ok")
            self.assertEqual(by_name["config:harbor.container_env"].status, "ok")

    def test_config_runtime_check_checks_e2b_backend(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"e2b", "dockerfile_parse", "dirhash"}:
                return object()
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            run_dir = (Path.cwd() / "results" / "runs" / "runtime_check_e2b").resolve()
            config_path.write_text(
                f"""
{{
  "backend": {{
    "name": "harbor",
    "env": "e2b"
  }},
  "output": {{
    "run_dir": "{run_dir}"
  }}
}}
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"E2B_API_KEY": "test-key"}):
                with patch("builtins.__import__", side_effect=fake_import):
                    checks = inspect_experiment_config(config_path)
        by_name = {check.name: check for check in checks}

        self.assertEqual(by_name["config:run_dir"].status, "ok")
        self.assertEqual(by_name["config:harbor.e2b_extra"].status, "ok")
        self.assertEqual(by_name["config:harbor.e2b_api_key"].status, "ok")

    def test_config_runtime_check_fails_e2b_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                """
{
  "backend": {
    "name": "harbor",
    "env": "e2b"
  },
  "output": {
    "run_dir": "results/runs/runtime_check_e2b_missing_key"
  }
}
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                checks = inspect_experiment_config(config_path)
        by_name = {check.name: check for check in checks}

        self.assertEqual(by_name["config:harbor.e2b_api_key"].status, "fail")

    def test_runtime_check_rejects_missing_baseline_class_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "incomplete.json"
            config_path.write_text(
                """
{
  "baseline": {
    "name": "missing-class-path"
  }
}
""",
                encoding="utf-8",
            )
            checks = inspect_experiment_config(config_path)
        by_name = {check.name: check for check in checks}

        self.assertEqual(by_name["config:baseline.lifecycle"].status, "fail")
        self.assertIn("baseline.class_path", by_name["config:baseline.lifecycle"].detail)

    def test_container_network_check_uses_verifier_env(self) -> None:
        def fake_run(command, check, capture_output, text, timeout):
            self.assertEqual(command[:3], ["docker", "run", "--rm"])
            self.assertIn("-e", command)
            self.assertIn("HTTP_PROXY=http://192.168.5.2:7890", command)
            self.assertIn("https://github.com", command)
            return __import__("subprocess").CompletedProcess(command, 0, stdout="HTTP 200\n", stderr="")

        with patch("seagym.runtime.checks.shutil.which", return_value="/usr/bin/docker"):
            with patch("seagym.runtime.checks.subprocess.run", side_effect=fake_run):
                checks = run_container_network_checks(
                    ["https://github.com"],
                    env={"HTTP_PROXY": "http://192.168.5.2:7890"},
                    timeout_seconds=1,
                )

        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, "ok")
        self.assertIn("HTTP 200", checks[0].detail)


if __name__ == "__main__":
    unittest.main()
