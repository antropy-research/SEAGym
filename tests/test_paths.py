from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seagym.baselines.native_runtime import NativeRuntimeConfig
from seagym.config.experiment import ExperimentConfig
from seagym.data.datasets import _load_task_index_json
from seagym.paths import resolve_portable_path
from seagym.utils import write_json


class PortablePathTest(unittest.TestCase):
    def test_repo_anchor_resolves_from_repo_root(self):
        resolved = resolve_portable_path("repo://reference/ace")
        self.assertEqual(resolved, Path.cwd() / "reference" / "ace")

    def test_data_and_results_anchors_use_environment_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                "os.environ",
                {
                    "SEAGYM_DATA_ROOT": str(root / "datasets"),
                    "SEAGYM_RESULTS_ROOT": str(root / "results"),
                },
            ):
                self.assertEqual(resolve_portable_path("data://hle"), root / "datasets" / "hle")
                self.assertEqual(
                    resolve_portable_path("results://my_run"),
                    root / "results" / "my_run",
                )

    def test_missing_data_root_has_clear_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SEAGYM_DATA_ROOT"):
                resolve_portable_path("data://hle")

    def test_experiment_config_keeps_config_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "runs" / "example" / "configs"
            config_dir.mkdir(parents=True)
            config = ExperimentConfig.from_dict(
                {
                    "experiment_id": "portable",
                    "seed": 42,
                    "task_dataset": {"path": "../tasks/index.json"},
                    "split_manifest": {"path": "../splits/split.json"},
                    "output": {"run_dir": "../../../results/runs/portable"},
                },
                base_dir=config_dir,
            )
            self.assertEqual(config.task_dataset_path, (config_dir.parent / "tasks" / "index.json").resolve())
            self.assertEqual(config.split_manifest_path, (config_dir.parent / "splits" / "split.json").resolve())
            self.assertEqual(config.run_dir, (Path(tmp) / "results" / "runs" / "portable").resolve())

    def test_task_index_normalizes_data_anchor_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_index_path = root / "index.json"
            write_json(
                task_index_path,
                {
                    "version": "test",
                    "tasks": [
                        {
                            "task_id": "hle/example",
                            "source": {
                                "type": "harbor_local",
                                "dataset_path": "data://hle",
                                "local_path": "data://hle/hle__example",
                            },
                            "attributes": {},
                            "scoring": {"reward_key": "reward"},
                        }
                    ],
                },
            )
            with patch.dict("os.environ", {"SEAGYM_DATA_ROOT": str(root / "datasets")}):
                index = _load_task_index_json(task_index_path)
            task = index.require("hle/example")
            self.assertEqual(task.source["dataset_path"], str(root / "datasets" / "hle"))
            self.assertEqual(task.source["local_path"], str(root / "datasets" / "hle" / "hle__example"))

    def test_native_runtime_resolves_anchored_python_bin_but_preserves_plain_relative(self):
        runtime = NativeRuntimeConfig.from_config({"python_bin": ".venv/bin/python"})
        self.assertEqual(runtime.python_bin, ".venv/bin/python")

        anchored = NativeRuntimeConfig.from_config({"python_bin": "repo://reference/ace/.venv/bin/python"})
        self.assertEqual(anchored.python_bin, str(Path.cwd() / "reference" / "ace" / ".venv" / "bin" / "python"))


if __name__ == "__main__":
    unittest.main()
