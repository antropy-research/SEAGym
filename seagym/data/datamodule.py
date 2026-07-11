from __future__ import annotations

"""Deterministic materialization of train batches and evaluation views.

The SEAGymDataModule is the ML-style bridge between static data specs and executable
evaluation loops. It consumes a task index, a train/val/test split, and an
experiment strategy, then writes down the exact task ids that will be used.

Inputs:
- `ExperimentContext.config.schedule`: train size, batch size, epochs, per-batch
  update count, view sizes, and validation frequency.
- `ExperimentContext.config.dataloader`: deterministic shuffling / drop-last.
- `ExperimentContext.config.evaluation_strategy`: rules for update-validation,
  replay, final views, and negative-transfer probes.
- `ExperimentContext.split`: conventional `train`, `val`, and `test` ids.

Outputs:
- `BatchPlan.train_batches`: ordered train batches.
- `BatchPlan.views.update_validation`: frozen V_update-val task ids.
- `BatchPlan.views.replay`: replay task ids, if enabled.
- `BatchPlan.views.final`: named final-test views.

BDD expectations:
- Given the same config, split, task index, and seed, `build()` must return the
  same output byte-for-byte.
- Given final view filters, the loader filters by stable task attributes only,
  not by hidden verifier outputs or dynamic execution results.
- Given an invalid base split name, the loader fails early.

Future work:
- Allow users to supply Python strategy objects for richer sampling policies.
- Persist strategy code/version refs in the output manifest.
- Support step-dependent replay materialization without putting replay keys
  into the split manifest.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from seagym.data.samplers import select_subset, shuffled
from seagym.data.stratified import stratified_subset
from seagym.data.types import TaskRecord
from seagym.utils import read_json, stable_dict_hash, write_json

if TYPE_CHECKING:
    from seagym.config import ExperimentContext


@dataclass(frozen=True)
class BatchPlan:
    run_id: str
    split_id: str
    experiment_id: str
    seed: int
    train_batches: list[list[str]]
    views: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "split_id": self.split_id,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "train_batches": self.train_batches,
            "views": self.views,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchPlan":
        train_batches = data.get("train_batches")
        views = data.get("views")
        if not isinstance(train_batches, list) or not all(isinstance(batch, list) for batch in train_batches):
            raise ValueError("BatchPlan missing train_batches list")
        if not isinstance(views, dict):
            raise ValueError("BatchPlan missing views object")
        return cls(
            run_id=str(data.get("run_id", "unknown")),
            split_id=str(data.get("split_id", "unknown")),
            experiment_id=str(data.get("experiment_id", "unknown")),
            seed=int(data.get("seed", 0)),
            train_batches=[[str(task_id) for task_id in batch] for batch in train_batches],
            views=_stringify_view_ids(views),
        )


class SEAGymDataModule:
    def __init__(self, context: "ExperimentContext"):
        self.context = context

    def build(self) -> BatchPlan:
        frozen_path = self.context.config.dataloader.batch_plan_path
        if frozen_path is not None and frozen_path.exists():
            return self._load_frozen_batch_plan(frozen_path)
        plan = self._build_materialized_plan()
        if frozen_path is not None:
            write_json(frozen_path, plan.to_dict())
        return plan

    def _build_materialized_plan(self) -> BatchPlan:
        batch_size = self.context.config.schedule.batch_size
        if batch_size <= 0:
            raise ValueError("schedule.batch_size must be positive")
        if self.context.config.schedule.num_epochs <= 0:
            raise ValueError("schedule.num_epochs must be positive")
        train_set = self._materialize_train_set()
        train_batches: list[list[str]] = []
        for epoch_index in range(self.context.config.schedule.num_epochs):
            epoch_batches = self._materialize_train_batches_for_epoch(train_set, epoch_index, batch_size)
            train_batches.extend(epoch_batches)
        if not train_batches:
            raise ValueError("No train batches generated")

        views = {
            "update_validation": self._materialize_update_validation(),
            "replay": self._materialize_replay(train_batches),
            "final": self._materialize_final_views(),
        }
        return BatchPlan(
            run_id=self._run_id(),
            split_id=self.context.split.split_id,
            experiment_id=self.context.config.experiment_id,
            seed=self.context.config.seed,
            train_batches=train_batches,
            views=views,
        )

    def _materialize_train_set(self) -> list[str]:
        candidates = list(self.context.split.train)
        sample_size = self.context.config.schedule.train_size
        if self.context.config.dataloader.batching_strategy == "stratified":
            records = [self.context.task_index.require(task_id) for task_id in candidates]
            target_size = len(records) if sample_size is None else min(int(sample_size), len(records))
            selected = stratified_subset(
                records,
                target_size=target_size,
                key=self._stratification_key,
                seed=self._seed("train"),
            )
            selected_ids = {task.task_id for task in selected}
            return [task_id for task_id in candidates if task_id in selected_ids]
        return select_subset(candidates, sample_size, self._seed("train"))

    def _materialize_train_batches_for_epoch(
        self,
        train_set: list[str],
        epoch_index: int,
        batch_size: int,
    ) -> list[list[str]]:
        if self.context.config.dataloader.batching_strategy == "stratified":
            return self._stratified_train_batches(train_set, epoch_index, batch_size)
        train = list(train_set)
        if self.context.config.dataloader.shuffle_train:
            train = shuffled(train, self._seed(f"train_shuffle:{epoch_index}"))
        epoch_batches = [train[i : i + batch_size] for i in range(0, len(train), batch_size)]
        if self.context.config.dataloader.drop_last and epoch_batches and len(epoch_batches[-1]) < batch_size:
            epoch_batches = epoch_batches[:-1]
        return epoch_batches

    def _stratified_train_batches(
        self,
        train_set: list[str],
        epoch_index: int,
        batch_size: int,
    ) -> list[list[str]]:
        groups: dict[tuple[object, ...], list[str]] = {}
        for task_id in train_set:
            task = self.context.task_index.require(task_id)
            groups.setdefault(self._stratification_key(task), []).append(task_id)
        for stratum, task_ids in groups.items():
            if self.context.config.dataloader.shuffle_train:
                groups[stratum] = shuffled(task_ids, self._seed(f"train_shuffle:{epoch_index}:{repr(stratum)}"))

        batches: list[list[str]] = []
        while True:
            remaining = {stratum: len(task_ids) for stratum, task_ids in groups.items() if task_ids}
            if not remaining:
                break
            batch_target = min(batch_size, sum(remaining.values()))
            if self.context.config.dataloader.drop_last and batch_target < batch_size:
                break
            counts = _allocate_counts(remaining, batch_target)
            batch: list[str] = []
            for stratum in sorted(counts, key=str):
                for _ in range(counts[stratum]):
                    if groups[stratum]:
                        batch.append(groups[stratum].pop(0))
            if len(batch) < batch_target:
                for stratum in sorted(groups, key=str):
                    while groups[stratum] and len(batch) < batch_target:
                        batch.append(groups[stratum].pop(0))
            if batch:
                batches.append(batch)
        return batches

    def _stratification_key(self, task: TaskRecord) -> tuple[object, ...]:
        paths = self.context.config.dataloader.stratify_by
        if not paths:
            return ("all",)
        return tuple(_get_path({"attributes": task.attributes, "source": task.source}, path) for path in paths)

    def _materialize_update_validation(self) -> list[str]:
        strategy = self.context.config.evaluation_strategy.get("update_validation", {})
        if not strategy.get("enabled", True):
            return []
        source_split = str(strategy.get("source_split", "val"))
        candidates = self._base_split(source_split)
        sample_size = strategy.get("val_size", self.context.config.schedule.val_size)
        if sample_size is not None and int(sample_size) <= 0:
            return []
        if not candidates and not strategy.get("allow_empty", False):
            raise ValueError("update-validation view is empty")
        return select_subset(candidates, sample_size, self._seed("update_validation"))

    def _materialize_replay(self, train_batches: list[list[str]]) -> list[str]:
        strategy = self.context.config.evaluation_strategy.get("replay", {})
        if not strategy.get("enabled", False):
            return []
        source = strategy.get("source", "seen_train_tasks")
        if source == "seen_train_tasks":
            candidates = [task_id for batch in train_batches for task_id in batch]
        else:
            candidates = self._base_split(str(source))
        sample_size = strategy.get("replay_size")
        if sample_size is not None and int(sample_size) <= 0:
            return []
        return select_subset(candidates, sample_size, self._seed("replay"))

    def _materialize_final_views(self) -> dict[str, list[str]]:
        views: dict[str, list[str]] = {}
        for view in self.context.config.evaluation_strategy.get("final_test_views", []):
            if not isinstance(view, dict):
                raise ValueError("final_test_views entries must be objects")
            name = str(view["name"])
            candidates = self._base_split(str(view.get("source_split", "test")))
            candidates = self._filter_tasks(candidates, view.get("filter") or {})
            sample_size = view.get("test_size", self.context.config.schedule.test_size)
            if sample_size is not None and int(sample_size) <= 0:
                views[name] = []
            else:
                views[name] = select_subset(candidates, sample_size, self._seed(f"final:{name}"))

        probe = self.context.config.evaluation_strategy.get("negative_transfer_probe", {})
        if probe.get("enabled", False):
            candidates = self._base_split(str(probe.get("source_split", "test")))
            candidates = self._filter_tasks(candidates, probe.get("filter") or {})
            views["negative_transfer_probe"] = select_subset(
                candidates,
                probe.get("probe_size"),
                self._seed("negative_transfer_probe"),
            )
        return views

    def _base_split(self, name: str) -> list[str]:
        if name == "train":
            return list(self.context.split.train)
        if name == "val":
            return list(self.context.split.val)
        if name == "test":
            return list(self.context.split.test)
        raise ValueError(f"Unknown base split {name!r}")

    def _filter_tasks(self, task_ids: list[str], filters: dict[str, Any]) -> list[str]:
        selected = task_ids
        for key, expected in filters.items():
            values = expected if isinstance(expected, list) else [expected]
            selected = [
                task_id
                for task_id in selected
                if _lookup(self.context.task_index.require(task_id).attributes, key) in values
            ]
        return selected

    def _seed(self, namespace: str) -> int:
        seed = self.context.config.dataloader.seed
        if seed is None:
            seed = self.context.config.seed
        return int(stable_dict_hash({"seed": seed, "namespace": namespace}), 16)

    def _run_id(self) -> str:
        return f"{self.context.config.experiment_id}-{stable_dict_hash(self.context.config.raw)}"

    def _load_frozen_batch_plan(self, path) -> BatchPlan:
        frozen = BatchPlan.from_dict(read_json(path))
        self._validate_task_ids(frozen.train_batches, "train_batches")
        self._validate_task_ids(frozen.views, "views")
        if frozen.split_id not in {self.context.split.split_id, "unknown"}:
            raise ValueError(
                f"Frozen BatchPlan split_id {frozen.split_id!r} does not match "
                f"current split_id {self.context.split.split_id!r}"
            )
        return BatchPlan(
            run_id=self._run_id(),
            split_id=self.context.split.split_id,
            experiment_id=self.context.config.experiment_id,
            seed=self.context.config.dataloader.seed
            if self.context.config.dataloader.seed is not None
            else self.context.config.seed,
            train_batches=frozen.train_batches,
            views=frozen.views,
        )

    def _validate_task_ids(self, value: Any, location: str) -> None:
        missing: list[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str):
                if item not in self.context.task_index.tasks:
                    missing.append(item)
                return
            if isinstance(item, list):
                for child in item:
                    collect(child)
                return
            if isinstance(item, dict):
                for child in item.values():
                    collect(child)

        collect(value)
        if missing:
            raise ValueError(f"Frozen BatchPlan {location} references unknown task ids: {sorted(set(missing))}")


def _lookup(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if part == "attributes":
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _get_path(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _allocate_counts(stratum_sizes: dict[tuple[object, ...], int], target_size: int) -> dict[tuple[object, ...], int]:
    total = sum(stratum_sizes.values())
    if total <= 0:
        return {stratum: 0 for stratum in stratum_sizes}
    raw = {stratum: size * target_size / total for stratum, size in stratum_sizes.items()}
    counts = {stratum: int(value) for stratum, value in raw.items()}
    remaining = target_size - sum(counts.values())
    order = sorted(
        stratum_sizes,
        key=lambda stratum: (raw[stratum] - counts[stratum], str(stratum)),
        reverse=True,
    )
    for stratum in order[:remaining]:
        counts[stratum] += 1
    return counts


def _stringify_view_ids(value: Any) -> Any:
    if isinstance(value, str):
        return str(value)
    if isinstance(value, list):
        return [_stringify_view_ids(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stringify_view_ids(item) for key, item in value.items()}
    return value
