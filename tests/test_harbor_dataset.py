from __future__ import annotations

import tempfile
import textwrap
import unittest
import json
from pathlib import Path

from seagym.data import load_task_index


class HarborDatasetTest(unittest.TestCase):
    def test_loads_local_harbor_task_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "demo-task"
            task_dir.mkdir()
            (task_dir / "task.toml").write_text(
                textwrap.dedent(
                    """
                    version = "1.0"

                    [task]
                    name = "demo/demo-task"
                    keywords = ["debugging", "python"]

                    [metadata]
                    difficulty = "easy"
                    category = "debugging"
                    tags = ["swe-bench"]
                    """
                ).strip(),
                encoding="utf-8",
            )

            task_index = load_task_index(root)

            self.assertEqual(task_index.version, f"harbor-dir:{root.name}")
            task = task_index.require("demo/demo-task")
            self.assertEqual(task.source["type"], "harbor")
            self.assertEqual(task.source["dataset_path"], str(root))
            self.assertEqual(task.source["task_name"], "demo-task")
            self.assertEqual(task.source["registry_task_name"], "demo/demo-task")
            self.assertEqual(task.source["local_path"], str(task_dir))
            self.assertEqual(task.attributes["domain"], "code")
            self.assertEqual(task.attributes["task_type"], "debugging")
            self.assertEqual(task.attributes["skills"], ["debugging", "python"])
            self.assertEqual(task.scoring.main_reward_key, "reward")

    def test_task_index_json_resolves_relative_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "inputs"
            index_dir.mkdir()
            payload = {
                "version": "test",
                "tasks": [
                    {
                        "task_id": "demo/task",
                        "source": {
                            "type": "harbor",
                            "dataset_path": "../dataset",
                            "local_path": "../dataset/task",
                            "task_name": "task",
                        },
                        "attributes": {},
                        "scoring": {"main_reward_key": "reward", "success_threshold": 1.0},
                    }
                ],
            }
            index_path = index_dir / "task_index.json"
            index_path.write_text(json.dumps(payload), encoding="utf-8")

            task_index = load_task_index(index_path)

            task = task_index.require("demo/task")
            self.assertEqual(task.source["dataset_path"], str((root / "dataset").resolve()))
            self.assertEqual(task.source["local_path"], str((root / "dataset" / "task").resolve()))


if __name__ == "__main__":
    unittest.main()
