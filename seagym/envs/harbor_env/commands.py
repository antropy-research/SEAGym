from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from seagym.data.types import TaskRecord


def format_agent_kwarg(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def harbor_batch_job_name(tasks: list[TaskRecord]) -> str:
    first = str(tasks[0].source.get("task_name") or tasks[0].task_id).replace("/", "__")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in first)
    return f"seagym-batch-{safe}-{len(tasks)}-{int(time.time() * 1000)}"


def task_run_path(task: TaskRecord) -> Path | None:
    local_path = task.source.get("local_path")
    dataset_path = task.source.get("dataset_path")
    if not (dataset_path or local_path):
        return None
    return Path(str(dataset_path or local_path)).resolve()


def task_local_path(task: TaskRecord) -> Path | None:
    """Return the concrete task directory used for an ordered Harbor job."""
    local_path = task.source.get("local_path")
    if local_path:
        return Path(str(local_path)).resolve()
    dataset_path = task.source.get("dataset_path")
    task_name = task.source.get("task_name")
    if dataset_path and task_name:
        return Path(str(dataset_path)).resolve() / str(task_name).rsplit("/", 1)[-1]
    return None


def group_tasks_by_run_path(tasks: list[TaskRecord]) -> dict[Path, list[TaskRecord]]:
    groups: dict[Path, list[TaskRecord]] = {}
    for task in tasks:
        run_path = task_run_path(task)
        if run_path is None:
            run_path = Path(f"__missing__/{task.task_id}")
        groups.setdefault(run_path, []).append(task)
    return groups


def templatize_env(env: dict[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in env.items():
        if key in os.environ and os.environ[key] == value:
            values[key] = f"${{{key}}}"
        else:
            values[key] = value
    return values


def materialize_timeout_patched_dataset(
    source: Path,
    destination: Path,
    *,
    task_names: list[str] | None,
    agent_timeout_sec: float | None,
    verifier_timeout_sec: float | None,
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if task_names is None:
        shutil.copytree(source, destination)
        patch_task_timeouts(
            destination / "task.toml",
            agent_timeout_sec=agent_timeout_sec,
            verifier_timeout_sec=verifier_timeout_sec,
        )
        return destination

    needed = {name.rsplit("/", 1)[-1] for name in task_names}
    destination.mkdir(parents=True, exist_ok=True)
    for task_dir in sorted(source.iterdir()):
        if not task_dir.is_dir() or task_dir.name not in needed:
            continue
        target = destination / task_dir.name
        shutil.copytree(task_dir, target)
        patch_task_timeouts(
            target / "task.toml",
            agent_timeout_sec=agent_timeout_sec,
            verifier_timeout_sec=verifier_timeout_sec,
        )
    return destination


def patch_task_timeouts(
    task_toml: Path,
    *,
    agent_timeout_sec: float | None,
    verifier_timeout_sec: float | None,
) -> None:
    if not task_toml.exists():
        return
    text = task_toml.read_text(encoding="utf-8")
    if agent_timeout_sec is not None:
        text = patch_timeout_section(text, section="agent", timeout_sec=agent_timeout_sec)
    if verifier_timeout_sec is not None:
        text = patch_timeout_section(text, section="verifier", timeout_sec=verifier_timeout_sec)
    task_toml.write_text(text, encoding="utf-8")


def patch_timeout_section(text: str, *, section: str, timeout_sec: float) -> str:
    replacement = f"timeout_sec = {timeout_sec:.1f}"
    section_match = re.search(rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)", text)
    if section_match is None:
        return text.rstrip() + f"\n\n[{section}]\n{replacement}\n"
    section_text = section_match.group(0)
    if re.search(r"(?m)^timeout_sec\s*=", section_text):
        section_text = re.sub(r"(?m)^timeout_sec\s*=.*$", replacement, section_text, count=1)
    else:
        section_text = section_text.rstrip() + f"\n{replacement}\n"
    return text[: section_match.start()] + section_text + text[section_match.end() :]
