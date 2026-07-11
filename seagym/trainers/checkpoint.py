from __future__ import annotations

"""Checkpoint metadata for SEAGym trainer runs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainerState:
    epoch: int
    train_batch_index: int
    global_step: int
    updates_completed: int
    num_train_tasks_seen: int
    checkpoint_id: str | None = None
    previous_update_validation_results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_batch_index": self.train_batch_index,
            "global_step": self.global_step,
            "updates_completed": self.updates_completed,
            "num_train_tasks_seen": self.num_train_tasks_seen,
            "checkpoint_id": self.checkpoint_id,
            "previous_update_validation_results": self.previous_update_validation_results or [],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        previous = data.get("previous_update_validation_results") or []
        if not isinstance(previous, list):
            previous = []
        return cls(
            epoch=int(data.get("epoch", 0)),
            train_batch_index=int(data.get("train_batch_index", 0)),
            global_step=int(data.get("global_step", data.get("updates_completed", 0))),
            updates_completed=int(data.get("updates_completed", data.get("global_step", 0))),
            num_train_tasks_seen=int(data.get("num_train_tasks_seen", 0)),
            checkpoint_id=None if data.get("checkpoint_id") in (None, "") else str(data.get("checkpoint_id")),
            previous_update_validation_results=[
                item for item in previous if isinstance(item, dict)
            ],
        )


def write_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    checkpoint_type: str,
    run_id: str,
    experiment_id: str,
    trainer_state: TrainerState,
    metadata: dict[str, Any] | None = None,
    refs: dict[str, Any] | None = None,
    baseline_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": checkpoint_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "trainer_state": trainer_state.to_dict(),
        "metadata": metadata or {},
        "refs": refs or {},
    }
    if baseline_manifest is not None:
        manifest["baseline"] = baseline_manifest
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_checkpoint_manifest(checkpoint_dir_or_file: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir_or_file)
    manifest_path = path if path.name == "checkpoint.json" else path / "checkpoint.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint manifest must be an object: {manifest_path}")
    return data


def write_latest_checkpoint(run_dir: Path, checkpoint_id: str) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "latest_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "path": f"{checkpoint_id}/checkpoint.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_checkpoint(run_dir: Path, checkpoint: str | Path) -> Path:
    value = Path(checkpoint)
    if str(checkpoint) == "latest":
        latest_path = run_dir / "checkpoints" / "latest_checkpoint.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"Latest checkpoint ref not found: {latest_path}")
        data = json.loads(latest_path.read_text(encoding="utf-8"))
        rel_path = data.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(f"Latest checkpoint ref missing path: {latest_path}")
        return (latest_path.parent / rel_path).resolve().parent
    if value.is_absolute() or value.exists() or len(value.parts) > 1:
        return value.parent if value.name == "checkpoint.json" else value
    return run_dir / "checkpoints" / value
