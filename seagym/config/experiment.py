from __future__ import annotations

"""Experiment configuration types and loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seagym.data.datasets import load_task_index
from seagym.data.splits import load_split_manifest, validate_split_manifest
from seagym.data.types import SplitManifest, TaskIndex
from seagym.paths import resolve_portable_path
from seagym.utils import read_json


@dataclass(frozen=True)
class ScheduleConfig:
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    batch_size: int = 10
    num_epochs: int = 1
    num_updates_per_batch: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScheduleConfig":
        data = data or {}
        allowed_keys = {
            "train_size",
            "val_size",
            "test_size",
            "batch_size",
            "num_epochs",
            "num_updates_per_batch",
        }
        unknown = sorted(set(data) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown schedule config keys: {unknown}")
        train_size = data.get("train_size")
        val_size = data.get("val_size")
        test_size = data.get("test_size")
        return cls(
            train_size=None if train_size is None else int(train_size),
            val_size=None if val_size is None else int(val_size),
            test_size=None if test_size is None else int(test_size),
            batch_size=int(data.get("batch_size", 10)),
            num_epochs=int(data.get("num_epochs", 1)),
            num_updates_per_batch=int(data.get("num_updates_per_batch", 1)),
        )


@dataclass(frozen=True)
class SEAGymDataModuleConfig:
    shuffle_train: bool = True
    drop_last: bool = False
    batching_strategy: str = "shuffle"
    stratify_by: tuple[str, ...] = ()
    seed: int | None = None
    batch_plan_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, base_dir: Path | None = None) -> "SEAGymDataModuleConfig":
        data = data or {}
        batching_strategy = str(data.get("batching_strategy", "shuffle"))
        if batching_strategy not in {"shuffle", "stratified"}:
            raise ValueError(f"Unknown dataloader.batching_strategy: {batching_strategy}")
        raw_stratify_by = data.get("stratify_by") or []
        if not isinstance(raw_stratify_by, list | tuple):
            raise ValueError("dataloader.stratify_by must be a list")
        seed = data.get("seed")
        batch_plan_path = data.get("batch_plan_path")
        return cls(
            shuffle_train=bool(data.get("shuffle_train", True)),
            drop_last=bool(data.get("drop_last", False)),
            batching_strategy=batching_strategy,
            stratify_by=tuple(str(item) for item in raw_stratify_by),
            seed=None if seed is None else int(seed),
            batch_plan_path=None
            if batch_plan_path in (None, "")
            else _resolve(base_dir or Path.cwd(), batch_plan_path),
        )


@dataclass(frozen=True)
class RuntimeSchedulingConfig:
    enabled: bool = False
    apply_to: tuple[str, ...] = ("train",)
    policy: str = "fixed"
    ema_k: float = 5.0
    cold_start: str = "none"
    random_seed: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, default_seed: int) -> "RuntimeSchedulingConfig":
        data = data or {}
        allowed_keys = {
            "enabled",
            "apply_to",
            "policy",
            "estimator",
            "runtime_field",
            "random_seed",
            "diagnostics",
        }
        unknown = sorted(set(data) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown runtime_scheduling config keys: {unknown}")
        apply_to = data.get("apply_to", ["train"])
        if not isinstance(apply_to, list | tuple) or not all(isinstance(item, str) for item in apply_to):
            raise ValueError("runtime_scheduling.apply_to must be a list of modes")
        policy = str(data.get("policy", "fixed"))
        if policy not in {"fixed", "random", "lpt"}:
            raise ValueError(f"Unknown runtime_scheduling.policy: {policy}")
        estimator = data.get("estimator") or {}
        if not isinstance(estimator, dict):
            raise ValueError("runtime_scheduling.estimator must be an object")
        kind = str(estimator.get("kind", "ema"))
        if kind != "ema":
            raise ValueError(f"Unknown runtime_scheduling.estimator.kind: {kind}")
        ema_k = float(estimator.get("k", 5.0))
        if ema_k <= 0:
            raise ValueError("runtime_scheduling.estimator.k must be positive")
        cold_start = str(estimator.get("cold_start", "none"))
        if cold_start != "none":
            raise ValueError("runtime_scheduling.estimator.cold_start currently supports only 'none'")
        runtime_field = str(data.get("runtime_field", "runtime_seconds"))
        if runtime_field != "runtime_seconds":
            raise ValueError("runtime_scheduling.runtime_field must be 'runtime_seconds'")
        diagnostics = data.get("diagnostics") or {}
        if not isinstance(diagnostics, dict):
            raise ValueError("runtime_scheduling.diagnostics must be an object")
        return cls(
            enabled=bool(data.get("enabled", False)),
            apply_to=tuple(apply_to),
            policy=policy,
            ema_k=ema_k,
            cold_start=cold_start,
            random_seed=int(data.get("random_seed", default_seed)),
            diagnostics=dict(diagnostics),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int
    task_dataset_path: Path
    split_manifest_path: Path
    run_dir: Path
    schedule: ScheduleConfig
    dataloader: SEAGymDataModuleConfig
    runtime_scheduling: RuntimeSchedulingConfig
    evaluation_strategy: dict[str, Any]
    metrics: dict[str, Any]
    raw: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path,
        path: Path | None = None,
    ) -> "ExperimentConfig":
        task_dataset = data.get("task_dataset") or {}
        split_manifest = data.get("split_manifest") or {}
        output = data.get("output") or {}
        task_path = _resolve(base_dir, task_dataset.get("path"))
        split_path = _resolve(base_dir, split_manifest.get("path"))
        run_dir = _resolve(base_dir, output.get("run_dir", "results/runs/default"))
        return cls(
            experiment_id=str(data.get("experiment_id", "default")),
            seed=int(data.get("seed", 0)),
            task_dataset_path=task_path,
            split_manifest_path=split_path,
            run_dir=run_dir,
            schedule=ScheduleConfig.from_dict(data.get("schedule")),
            dataloader=SEAGymDataModuleConfig.from_dict(data.get("dataloader"), base_dir=base_dir),
            runtime_scheduling=RuntimeSchedulingConfig.from_dict(
                data.get("runtime_scheduling"),
                default_seed=int(data.get("seed", 0)),
            ),
            evaluation_strategy=data.get("evaluation_strategy") or {},
            metrics=data.get("metrics") or {},
            raw=data,
            path=path,
        )


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_file = Path(config_path).resolve()
    return ExperimentConfig.from_dict(
        read_json(config_file),
        base_dir=config_file.parent,
        path=config_file,
    )


@dataclass(frozen=True)
class ExperimentContext:
    """Loaded experiment config plus validated task index and split."""

    config: ExperimentConfig
    task_index: TaskIndex
    split: SplitManifest


def load_experiment_context(config_path: str | Path) -> ExperimentContext:
    config = load_experiment_config(config_path)
    task_index = load_task_index(config.task_dataset_path)
    split = load_split_manifest(config.split_manifest_path)
    validate_split_manifest(task_index, split)
    return ExperimentContext(config=config, task_index=task_index, split=split)


def _resolve(base_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Expected path value")
    return resolve_portable_path(value, base_dir=base_dir)
