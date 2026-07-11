"""SEAGym training framework primitives.

The package intentionally exposes small building blocks instead of a single
opaque evaluator entrypoint.
"""

from importlib.metadata import PackageNotFoundError, version

from .data import (
    SEAGymDataModule,
    BatchPlan,
    SplitManifest,
    TaskIndex,
    TaskRecord,
    load_task_index,
    load_split_manifest,
    validate_split_manifest,
)
from .envs import DeterministicEnv, Env, TaskRunResult
from .logging import ArtifactLayout, write_run_reports
from .metrics import Metric, MetricRegistry, default_metric_registry
from .models import ModelConfig
from .baselines import BaseBaseline, Baseline, BaselineState, StaticBaseline, TrajectoryBatch
from .rollout_agents import RolloutAgent
from .trainers import ExecutionEngine, SEAGymTrainer, TrainerState, UpdateValidationLoop
from .trainers.run import RunOptions

try:
    __version__ = version("seagym")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ArtifactLayout",
    "BaseBaseline",
    "Baseline",
    "BaselineState",
    "SEAGymDataModule",
    "BatchPlan",
    "DeterministicEnv",
    "Env",
    "ExecutionEngine",
    "Metric",
    "MetricRegistry",
    "ModelConfig",
    "RolloutAgent",
    "RunOptions",
    "SEAGymTrainer",
    "SplitManifest",
    "StaticBaseline",
    "TaskRunResult",
    "TaskIndex",
    "TaskRecord",
    "TrajectoryBatch",
    "TrainerState",
    "UpdateValidationLoop",
    "__version__",
    "default_metric_registry",
    "load_split_manifest",
    "load_task_index",
    "validate_split_manifest",
    "write_run_reports",
]
