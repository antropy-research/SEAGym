from __future__ import annotations

"""Run directory options for SEAGym training."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path | None = None
    run_dir: Path | None = None
    run_name: str | None = None
    resume: bool = False
    resume_from_checkpoint: str | Path | None = None
    overwrite: bool = False


def make_run_dir(
    *,
    experiment_id: str,
    output_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    if run_dir is not None:
        return Path(run_dir).resolve()
    root = Path(output_dir or "results/runs").resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = _slug(run_name or experiment_id or "run")
    return root / f"{timestamp}_{name}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"
