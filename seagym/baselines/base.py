from __future__ import annotations

"""Baseline lifecycle interface.

SEAGym evaluates baselines through a common ML/RL-style lifecycle. A baseline
may be implemented inside this repository or may call a native external project;
that distinction is not part of the framework-level control flow.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Protocol

from .data import TrajectoryBatch


@dataclass
class BaselineState:
    state_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_dir: Path
    state_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateResult:
    update_index: int
    changed: bool
    status: str = "updated"
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "baseline_update",
            "update_index": self.update_index,
            "changed": self.changed,
            "status": self.status,
            "metrics": self.metrics,
            "logs": self.logs,
            "artifacts": self.artifacts,
        }


class Baseline(Protocol):
    baseline_id: str

    def initialize(self, run_dir: Path) -> BaselineState:
        ...

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        ...

    def save_checkpoint(self, state: BaselineState, path: Path) -> Checkpoint:
        ...

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        ...

    def report(self, state: BaselineState) -> dict[str, Any]:
        return {}


@dataclass
class BaseBaseline:
    """Common state/checkpoint lifecycle for concrete baselines."""

    baseline_id: str
    state_dir: Path
    update_index: int = 0

    def __post_init__(self) -> None:
        self.state_dir = self.state_dir.resolve()

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        models: dict[str, Any],
        state_dir: Path,
        run_dir: Path,
        base_dir: Path | None,
    ) -> "BaseBaseline":
        del config, models, run_dir, base_dir
        return cls(baseline_id=name, state_dir=state_dir)

    def initialize(self, run_dir: Path) -> BaselineState:
        del run_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "baseline_id": self.baseline_id,
            "type": self.__class__.__name__,
        }
        (self.state_dir / "baseline_state.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return BaselineState(self.state_dir, metadata)

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        raise NotImplementedError(f"{self.__class__.__name__}.update() is not implemented")

    def save_checkpoint(self, state: BaselineState, path: Path) -> Checkpoint:
        path.mkdir(parents=True, exist_ok=True)
        destination = path / "baseline_state"
        if destination.exists():
            shutil.rmtree(destination)
        if state.state_dir.exists():
            shutil.copytree(state.state_dir, destination)
        manifest = {
            "type": "baseline_checkpoint",
            "baseline_id": self.baseline_id,
            "state_ref": destination.name,
            "update_index": self.update_index,
            "state_metadata": state.metadata,
        }
        (path / "checkpoint.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return Checkpoint(checkpoint_dir=path, state_ref=str(destination), metadata=manifest)

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        self.state_dir = self.state_dir.resolve()
        manifest_path = checkpoint.checkpoint_dir / "checkpoint.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest.get("baseline"), dict):
                manifest = manifest["baseline"]
            source = manifest.get("state_ref")
            if source:
                source_path = _resolve_checkpoint_ref(Path(str(source)), checkpoint.checkpoint_dir)
                if self.state_dir.exists():
                    shutil.rmtree(self.state_dir)
                shutil.copytree(source_path, self.state_dir)
                self.update_index = int(manifest.get("update_index", self.update_index))
                metadata = manifest.get("state_metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                return BaselineState(self.state_dir, {**metadata, "loaded": True, "manifest": manifest})
        return BaselineState(self.state_dir, {"loaded": False})

    def report(self, state: BaselineState) -> dict[str, Any]:
        return {"baseline_id": self.baseline_id, "state_dir": str(state.state_dir), "update_index": self.update_index}

    def next_update_dir(self, state: BaselineState, prefix: str) -> Path:
        self.update_index += 1
        update_dir = state.state_dir.resolve() / "updates" / f"{prefix}_{self.update_index:04d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        return update_dir

    def write_trajectories(
        self,
        trajectories: TrajectoryBatch,
        update_dir: Path,
        *,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        records = trajectories.to_dict()["trajectories"]
        if max_records is not None:
            records = records[:max_records]
        (update_dir / "trajectories.json").write_text(
            json.dumps({"records": records, "batch": trajectories.to_dict()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return records


def _resolve_checkpoint_ref(path: Path, checkpoint_dir: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = checkpoint_dir / path
    if candidate.exists():
        return candidate
    return path
