from __future__ import annotations

"""Run-local artifact layout for SEAGym experiments.

The layout keeps one SEAGym run self-contained: immutable inputs, normalized
machine-readable records, human-readable reports, and raw Harbor job evidence
all live under the same run directory.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class ArtifactLayout:
    run_dir: Path
    inputs_dir: Path
    records_dir: Path
    reports_dir: Path
    checkpoints_dir: Path
    harbor_jobs_dir: Path
    metrics_path: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "ArtifactLayout":
        root = Path(run_dir).resolve()
        return cls(
            run_dir=root,
            inputs_dir=root / "inputs",
            records_dir=root / "records",
            reports_dir=root / "reports",
            checkpoints_dir=root / "checkpoints",
            harbor_jobs_dir=root / "harbor" / "jobs",
            metrics_path=root / "metrics.json",
        )

    @property
    def experiment_config_path(self) -> Path:
        return self.inputs_dir / "experiment_config.json"

    @property
    def split_manifest_path(self) -> Path:
        return self.inputs_dir / "split_manifest.json"

    @property
    def batch_plan_path(self) -> Path:
        return self.inputs_dir / "batch_plan.json"

    @property
    def evaluation_points_path(self) -> Path:
        return self.records_dir / "evaluation_points.jsonl"

    @property
    def task_results_path(self) -> Path:
        return self.records_dir / "task_results.jsonl"

    @property
    def verifier_results_path(self) -> Path:
        return self.records_dir / "verifier_results.jsonl"

    @property
    def metric_inputs_path(self) -> Path:
        return self.records_dir / "metric_inputs.jsonl"

    @property
    def agent_updates_path(self) -> Path:
        return self.records_dir / "agent_updates.jsonl"

    @property
    def agent_checkpoints_path(self) -> Path:
        return self.records_dir / "agent_checkpoints.jsonl"

    @property
    def scheduling_decisions_path(self) -> Path:
        return self.records_dir / "scheduling_decisions.jsonl"

    @property
    def task_runtimes_path(self) -> Path:
        return self.records_dir / "task_runtimes.jsonl"

    @property
    def scheduling_history_path(self) -> Path:
        return self.run_dir / "runtime" / "scheduling_history.json"

    @property
    def scheduling_summary_path(self) -> Path:
        return self.reports_dir / "scheduling_summary.json"

    @property
    def summary_path(self) -> Path:
        return self.reports_dir / "summary.md"

    @property
    def tasks_csv_path(self) -> Path:
        return self.reports_dir / "tasks.csv"

    @property
    def evaluation_points_csv_path(self) -> Path:
        return self.reports_dir / "evaluation_points.csv"

    @property
    def failures_path(self) -> Path:
        return self.reports_dir / "failures.md"

    def prepare(self, *, overwrite: bool = False) -> None:
        preserved_runtime: Path | None = None
        if self.run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Run directory already exists: {self.run_dir}")
            runtime_dir = self.run_dir / "runtime"
            if runtime_dir.exists():
                preserved_runtime = self.run_dir.parent / f".{self.run_dir.name}.runtime-preserve"
                if preserved_runtime.exists():
                    shutil.rmtree(preserved_runtime)
                shutil.copytree(runtime_dir, preserved_runtime)
            shutil.rmtree(self.run_dir)
        for path in (
            self.inputs_dir,
            self.records_dir,
            self.reports_dir,
            self.checkpoints_dir,
            self.harbor_jobs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if preserved_runtime is not None:
            shutil.copytree(preserved_runtime, self.run_dir / "runtime")
            shutil.rmtree(preserved_runtime)
