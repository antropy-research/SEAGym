from __future__ import annotations

"""Human-readable run reports built from normalized SEAGym artifacts."""

import csv
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout
from seagym.utils import read_json, read_jsonl


def write_run_reports(run_dir: str | Path) -> None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(layout.metrics_path) if layout.metrics_path.exists() else {}
    task_rows = read_jsonl(layout.task_results_path)
    points = read_jsonl(layout.evaluation_points_path)
    _write_tasks_csv(layout.tasks_csv_path, task_rows)
    _write_evaluation_points_csv(layout.evaluation_points_csv_path, points)
    _write_failures(layout.failures_path, task_rows)
    _write_summary(layout.summary_path, layout, metrics, task_rows, points)


def _write_tasks_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "domain",
        "view_name",
        "mode",
        "evaluation_point_id",
        "agent_id",
        "agent_checkpoint_id",
        "baseline_role",
        "score",
        "success",
        "error",
        "attempt_count",
        "attempt_successes",
        "attempt_best_score",
        "harbor_job_dir",
        "harbor_result_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            refs = row.get("refs") or {}
            attrs = row.get("attributes") or {}
            attempt_summary = _attempt_summary(row)
            writer.writerow(
                {
                    "task_id": row.get("task_id"),
                    "domain": attrs.get("domain"),
                    "view_name": row.get("view_name"),
                    "mode": row.get("mode"),
                    "evaluation_point_id": row.get("evaluation_point_id"),
                    "agent_id": row.get("agent_id"),
                    "agent_checkpoint_id": row.get("agent_checkpoint_id"),
                    "baseline_role": row.get("baseline_role"),
                    "score": row.get("score"),
                    "success": row.get("success"),
                    "error": row.get("error") or "",
                    "attempt_count": attempt_summary["count"],
                    "attempt_successes": attempt_summary["successes"],
                    "attempt_best_score": attempt_summary["best_score"],
                    "harbor_job_dir": refs.get("job_dir") or "",
                    "harbor_result_path": refs.get("result_path") or "",
                }
            )


def _write_evaluation_points_csv(path: Path, points: list[dict[str, Any]]) -> None:
    fields = [
        "evaluation_point_id",
        "type",
        "train_batch_index",
        "num_train_tasks_seen",
        "view",
        "score",
        "num_tasks",
        "baseline_score",
        "gain_vs_A_0",
        "update_label",
        "delta_prev",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in points:
            update = point.get("update_assessment") or {}
            step = point.get("step") or {}
            evaluations = point.get("evaluations") or {}
            if not evaluations:
                evaluations = {"": {}}
            for view, summary in evaluations.items():
                summary = summary or {}
                writer.writerow(
                    {
                        "evaluation_point_id": point.get("evaluation_point_id"),
                        "type": point.get("type"),
                        "train_batch_index": step.get("train_batch_index"),
                        "num_train_tasks_seen": step.get("num_train_tasks_seen"),
                        "view": view,
                        "score": summary.get("score"),
                        "num_tasks": summary.get("num_tasks"),
                        "baseline_score": summary.get("baseline_score", ""),
                        "gain_vs_A_0": summary.get("gain_vs_A_0", ""),
                        "update_label": update.get("label"),
                        "delta_prev": update.get("delta_prev", ""),
                    }
                )


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if not row.get("success") or row.get("error")]
    lines = ["# Failed Task Runs", ""]
    if not failures:
        lines.append("No failed task runs.")
    for row in failures:
        refs = row.get("refs") or {}
        lines.extend(
            [
                f"## {row.get('task_id')}",
                "",
                f"- View: `{row.get('view_name')}`",
                f"- Score: `{row.get('score')}`",
                f"- Error: `{row.get('error') or ''}`",
                f"- Harbor job: `{refs.get('job_dir') or ''}`",
                f"- Harbor result: `{refs.get('result_path') or ''}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    layout: ArtifactLayout,
    metrics: dict[str, Any],
    task_rows: list[dict[str, Any]],
    points: list[dict[str, Any]],
) -> None:
    run_id = task_rows[0].get("run_id") if task_rows else layout.run_dir.name
    experiment_id = task_rows[0].get("experiment_id") if task_rows else "unknown"
    split_id = task_rows[0].get("split_id") if task_rows else "unknown"
    agents = sorted({str(row.get("agent_id")) for row in task_rows if row.get("agent_id")})
    views = sorted({str(row.get("view_name")) for row in task_rows if row.get("view_name")})
    successes = sum(1 for row in task_rows if row.get("success"))
    failures = len(task_rows) - successes
    attempt_count = sum(_attempt_summary(row)["count"] for row in task_rows)
    attempt_successes = sum(_attempt_summary(row)["successes"] for row in task_rows)
    attempt_failures = attempt_count - attempt_successes

    lines = [
        "# SEAGym Run Summary",
        "",
        f"- Run: `{run_id}`",
        f"- Experiment: `{experiment_id}`",
        f"- Split: `{split_id}`",
        f"- Agents: `{', '.join(agents) if agents else 'unknown'}`",
        f"- Task runs: `{len(task_rows)}`",
        f"- Successes: `{successes}`",
        f"- Failures: `{failures}`",
        f"- Task attempts: `{attempt_count}`",
        f"- Attempt successes: `{attempt_successes}`",
        f"- Attempt failures: `{attempt_failures}`",
        f"- Views: `{', '.join(views) if views else 'none'}`",
        "",
        "## Metrics",
        "",
        "| Metric | View | Value |",
        "|---|---|---:|",
    ]
    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            for view, view_value in sorted(value.items()):
                lines.append(f"| `{metric_name}` | `{view}` | `{view_value}` |")
        else:
            lines.append(f"| `{metric_name}` |  | `{value}` |")

    lines.extend(
        [
            "",
            "## Evaluation Points",
            "",
            "| Point | Type | Seen Tasks | View | Score | A0 Score | Gain | Tasks | Label | Delta Prev |",
            "|---|---|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for point in points:
        step = point.get("step") or {}
        update = point.get("update_assessment") or {}
        for view, summary in (point.get("evaluations") or {}).items():
            summary = summary or {}
            lines.append(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                    point.get("evaluation_point_id"),
                    point.get("type"),
                    step.get("num_train_tasks_seen"),
                    view,
                    summary.get("score"),
                    summary.get("baseline_score", ""),
                    summary.get("gain_vs_A_0", ""),
                    summary.get("num_tasks"),
                    update.get("label"),
                    update.get("delta_prev", ""),
                )
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Task results CSV: `{_relative(layout.tasks_csv_path, layout.run_dir)}`",
            f"- Evaluation points CSV: `{_relative(layout.evaluation_points_csv_path, layout.run_dir)}`",
            f"- Failures: `{_relative(layout.failures_path, layout.run_dir)}`",
            f"- Raw Harbor jobs: `{_relative(layout.harbor_jobs_dir, layout.run_dir)}`",
            f"- Normalized records: `{_relative(layout.records_dir, layout.run_dir)}`",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _attempt_summary(row: dict[str, Any]) -> dict[str, Any]:
    refs = row.get("refs") or {}
    attempts = refs.get("all_attempts")
    if not isinstance(attempts, list) or not attempts:
        return {
            "count": 1,
            "successes": 1 if row.get("success") else 0,
            "best_score": row.get("score"),
        }
    successes = 0
    scores: list[float] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("success"):
            successes += 1
        score = attempt.get("score")
        if isinstance(score, int | float):
            scores.append(float(score))
    return {
        "count": len(attempts),
        "successes": successes,
        "best_score": max(scores) if scores else row.get("score"),
    }
