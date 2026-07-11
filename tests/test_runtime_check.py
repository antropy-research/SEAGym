from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seagym.cli import main as cli_main
from seagym.runtime import RuntimeCheckOptions, inspect_runtime, inspect_task_resources
from seagym.data import TaskRecord
from seagym.envs import TaskRunResult
from seagym.utils import read_json


class RuntimeCheckTest(unittest.TestCase):
    def test_runtime_check_writes_manifest_without_running_canary(self) -> None:
        config_path = Path("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_runtime(
                RuntimeCheckOptions(
                    config_path=config_path,
                    run_dir=Path(tmp) / "run",
                    load_env=False,
                    host_probe_urls=[],
                    container_probe_urls=[],
                    canary=False,
                )
            )

            manifest = read_json(report.manifest_path)

        self.assertTrue(report.ok)
        self.assertIn("code_001", report.task_ids)
        self.assertIn("tool_val_001", report.task_ids)
        self.assertEqual(manifest["canary"]["enabled"], False)
        self.assertEqual(manifest["batch_plan"]["experiment_id"], "pilot")

    def test_runtime_check_runs_harbor_canary_when_requested(self) -> None:
        config_path = Path("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            fake_result = __import__("seagym.envs", fromlist=["TaskRunResult"]).TaskRunResult(
                task_id="code_001",
                view_name="runtime_canary",
                mode="runtime_check",
                rewards={"reward": 1.0},
                score=1.0,
                success=True,
            )
            with patch("seagym.runtime.inspect.run_harbor_canary", return_value=[fake_result]) as canary:
                report = inspect_runtime(
                    RuntimeCheckOptions(
                        config_path=config_path,
                        run_dir=Path(tmp) / "run",
                        load_env=False,
                        canary=True,
                        canary_task_limit=1,
                    )
                )

        self.assertTrue(report.ok)
        canary.assert_called_once()
        self.assertEqual(report.canary_results[0].task_id, "code_001")

    def test_task_resource_check_fails_for_missing_local_path(self) -> None:
        task = TaskRecord.from_dict(
            {
                "task_id": "missing",
                "source": {
                    "type": "harbor",
                    "dataset": "demo",
                    "dataset_path": "/definitely/missing/seagym/path",
                    "task_name": "missing",
                },
                "attributes": {"domain": "code"},
            }
        )

        checks = inspect_task_resources([task])

        self.assertEqual(checks[0].status, "fail")

    def test_runtime_check_report_redacts_canary_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = TaskRunResult(
                task_id="demo",
                view_name="runtime_canary",
                mode="runtime_check",
                rewards={"reward": 0.0},
                score=0.0,
                success=False,
                refs={
                    "command": [
                        "harbor",
                        "--verifier-env",
                        "HTTP_PROXY=http://user:pass@192.168.5.2:7890",
                    ],
                    "OPENAI_API_KEY": "secret",
                },
            )
            fake_result = [result]
            with patch("seagym.runtime.inspect.run_harbor_canary", return_value=fake_result):
                report = inspect_runtime(
                    RuntimeCheckOptions(
                        config_path=Path("tests/fixtures/pilot/configs/pilot.json"),
                        run_dir=Path(tmp) / "run",
                        load_env=False,
                        canary=True,
                    )
                )

            manifest = read_json(report.manifest_path)

        canary = manifest["canary"]["results"][0]
        self.assertEqual(canary["refs"]["OPENAI_API_KEY"], "<redacted>")
        self.assertIn("HTTP_PROXY=http://***@192.168.5.2:7890", canary["refs"]["command"])

    def test_cli_inspect_runtime_writes_report_without_default_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runtime_cli"
            stdout = io.StringIO()
            with patch("seagym.runtime.inspect.run_harbor_canary") as canary:
                with contextlib.redirect_stdout(stdout):
                    cli_main(
                        [
                            "inspect",
                            "runtime",
                            "tests/fixtures/pilot/configs/pilot.json",
                            "--run-dir",
                            str(run_dir),
                            "--no-load-env",
                        ]
                    )

            canary.assert_not_called()
            payload = json.loads(stdout.getvalue())
            manifest_exists = (Path(payload["manifest_path"])).exists()
            expected_manifest_exists = (run_dir / "runtime" / "runtime_check.json").exists()

        self.assertTrue(payload["ok"])
        self.assertEqual(Path(payload["manifest_path"]).name, "runtime_check.json")
        self.assertTrue(manifest_exists)
        self.assertTrue(expected_manifest_exists)
        self.assertEqual(payload["canary_results"], [])

    def test_cli_inspect_config_loads_default_env_file_for_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(f"SEAGYM_RESULTS_ROOT={root / 'results'}\n", encoding="utf-8")
            (root / "tasks.json").write_text(json.dumps({"version": "test", "tasks": []}), encoding="utf-8")
            (root / "split.json").write_text(
                json.dumps({"splits": {"train": [], "val": [], "test": []}}),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "experiment_id": "cli-env",
                        "task_dataset": {"path": "tasks.json"},
                        "split_manifest": {"path": "split.json"},
                        "output": {"run_dir": "results://cli_env/run"},
                    }
                ),
                encoding="utf-8",
            )
            old_cwd = Path.cwd()
            stdout = io.StringIO()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    with contextlib.redirect_stdout(stdout):
                        cli_main(["inspect", "config", "config.json"])
            finally:
                os.chdir(old_cwd)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["run_dir"], str(root / "results" / "cli_env" / "run"))


if __name__ == "__main__":
    unittest.main()
