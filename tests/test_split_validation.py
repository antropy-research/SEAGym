from __future__ import annotations

import unittest

from seagym.data import ScoringRule, SplitManifest, TaskIndex, TaskRecord
from seagym.data import validate_split_manifest


class SplitValidationTest(unittest.TestCase):
    def test_rejects_overlapping_base_splits(self) -> None:
        task_index = _task_index(["a", "b"])
        split = SplitManifest(
            split_id="demo",
            split_version="v1",
            seed=0,
            train=["a"],
            val=["a"],
            test=["b"],
        )

        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_split_manifest(task_index, split)

    def test_rejects_duplicate_ids_within_split(self) -> None:
        task_index = _task_index(["a", "b"])
        split = SplitManifest(
            split_id="demo",
            split_version="v1",
            seed=0,
            train=["a", "a"],
            val=[],
            test=["b"],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_split_manifest(task_index, split)


def _task_index(task_ids: list[str]) -> TaskIndex:
    return TaskIndex(
        version="test",
        tasks={
            task_id: TaskRecord(
                task_id=task_id,
                source={"type": "test"},
                attributes={"domain": "code"},
                scoring=ScoringRule(),
            )
            for task_id in task_ids
        },
    )


if __name__ == "__main__":
    unittest.main()
