from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from seagym import SEAGymDataModule
from seagym.config import load_experiment_context
from seagym.data.types import SplitManifest
from seagym.utils import read_json


class SEAGymDataModuleTest(unittest.TestCase):
    def test_materializes_deterministic_views(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        first = SEAGymDataModule(ctx).build()
        second = SEAGymDataModule(ctx).build()

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.train_batches), 2)
        self.assertEqual(set(first.views), {"update_validation", "replay", "final"})
        self.assertEqual(first.views["update_validation"], ["code_val_001", "tool_val_001"])
        self.assertEqual(set(first.views["final"]), {"id_test", "ood_test"})
        self.assertEqual(first.views["final"]["ood_test"], ["data_test_001"])

    def test_allows_empty_val_and_test_views(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        raw = dict(ctx.config.raw)
        raw["schedule"] = {
            **raw["schedule"],
            "train_size": 4,
            "val_size": 0,
            "test_size": 0,
            "batch_size": 4,
            "num_epochs": 3,
        }
        raw["evaluation_strategy"] = {
            "update_validation": {"enabled": False, "source_split": "val", "val_size": 0},
            "replay": {"enabled": False},
            "final_test_views": [],
        }
        config = replace(
            ctx.config,
            raw=raw,
            schedule=ctx.config.schedule.from_dict(raw["schedule"]),
            evaluation_strategy=raw["evaluation_strategy"],
        )
        plan = SEAGymDataModule(replace(ctx, config=config)).build()

        self.assertEqual(len(plan.train_batches), 3)
        self.assertEqual(plan.views["update_validation"], [])
        self.assertEqual(plan.views["replay"], [])
        self.assertEqual(plan.views["final"], {})

    def test_each_epoch_reuses_same_materialized_train_set(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        raw = dict(ctx.config.raw)
        raw["schedule"] = {
            **raw["schedule"],
            "train_size": 4,
            "val_size": 0,
            "test_size": 0,
            "batch_size": 2,
            "num_epochs": 3,
        }
        raw["evaluation_strategy"] = {
            "update_validation": {"enabled": False, "source_split": "val", "val_size": 0},
            "replay": {"enabled": False},
            "final_test_views": [],
        }
        config = replace(
            ctx.config,
            raw=raw,
            schedule=ctx.config.schedule.from_dict(raw["schedule"]),
            evaluation_strategy=raw["evaluation_strategy"],
        )
        split = SplitManifest(
            split_id=ctx.split.split_id,
            split_version=ctx.split.split_version,
            seed=ctx.split.seed,
            train=["train_001", "train_002", "train_003", "train_004", "train_005", "train_006"],
            val=[],
            test=[],
        )

        plan = SEAGymDataModule(replace(ctx, config=config, split=split)).build()

        self.assertEqual(len(plan.train_batches), 6)
        epoch_task_sets = []
        for epoch_index in range(3):
            start = epoch_index * 2
            epoch_task_sets.append({task_id for batch in plan.train_batches[start : start + 2] for task_id in batch})
        self.assertEqual(epoch_task_sets[0], epoch_task_sets[1])
        self.assertEqual(epoch_task_sets[0], epoch_task_sets[2])

    def test_stratified_batching_balances_configured_attributes(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        raw = dict(ctx.config.raw)
        raw["schedule"] = {
            **raw["schedule"],
            "train_size": 4,
            "val_size": 0,
            "test_size": 0,
            "batch_size": 2,
            "num_epochs": 2,
        }
        raw["dataloader"] = {
            **raw["dataloader"],
            "batching_strategy": "stratified",
            "stratify_by": ["attributes.domain"],
            "shuffle_train": True,
        }
        raw["evaluation_strategy"] = {
            "update_validation": {"enabled": False, "source_split": "val", "val_size": 0},
            "replay": {"enabled": False},
            "final_test_views": [],
        }
        config = replace(
            ctx.config,
            raw=raw,
            schedule=ctx.config.schedule.from_dict(raw["schedule"]),
            dataloader=ctx.config.dataloader.from_dict(raw["dataloader"]),
            evaluation_strategy=raw["evaluation_strategy"],
        )

        plan = SEAGymDataModule(replace(ctx, config=config)).build()

        self.assertEqual(len(plan.train_batches), 4)
        for batch in plan.train_batches:
            domains = {ctx.task_index.require(task_id).attributes["domain"] for task_id in batch}
            self.assertEqual(domains, {"code", "tool_use"})
        self.assertEqual(
            {task_id for batch in plan.train_batches[:2] for task_id in batch},
            {task_id for batch in plan.train_batches[2:] for task_id in batch},
        )

    def test_writes_and_reuses_frozen_batch_plan(self) -> None:
        ctx = load_experiment_context("tests/fixtures/pilot/configs/pilot.json")
        with tempfile.TemporaryDirectory() as tmp:
            batch_plan_path = Path(tmp) / "batch_plan.json"
            raw = dict(ctx.config.raw)
            raw["dataloader"] = {
                **raw["dataloader"],
                "seed": 123,
                "batch_plan_path": str(batch_plan_path),
            }
            config = replace(
                ctx.config,
                raw=raw,
                dataloader=ctx.config.dataloader.from_dict(raw["dataloader"]),
            )
            first = SEAGymDataModule(replace(ctx, config=config)).build()
            self.assertTrue(batch_plan_path.exists())

            frozen = read_json(batch_plan_path)
            self.assertEqual(frozen["train_batches"], first.train_batches)

            changed_raw = dict(raw)
            changed_raw["experiment_id"] = "pilot_other_baseline"
            changed_raw["seed"] = 999
            changed_raw["schedule"] = {
                **changed_raw["schedule"],
                "batch_size": 1,
            }
            changed_config = replace(
                ctx.config,
                experiment_id="pilot_other_baseline",
                seed=999,
                raw=changed_raw,
                schedule=ctx.config.schedule.from_dict(changed_raw["schedule"]),
                dataloader=ctx.config.dataloader.from_dict(changed_raw["dataloader"]),
            )
            second = SEAGymDataModule(replace(ctx, config=changed_config)).build()

        self.assertEqual(second.train_batches, first.train_batches)
        self.assertEqual(second.views, first.views)
        self.assertEqual(second.experiment_id, "pilot_other_baseline")
        self.assertNotEqual(second.run_id, first.run_id)


if __name__ == "__main__":
    unittest.main()
