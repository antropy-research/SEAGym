from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any


def _ensure_git_repo(path: Path) -> None:
    if (path / ".git").exists():
        return
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "seagym",
        "GIT_AUTHOR_EMAIL": "seagym@example.invalid",
        "GIT_COMMITTER_NAME": "seagym",
        "GIT_COMMITTER_EMAIL": "seagym@example.invalid",
    }
    subprocess.run(["git", "init"], cwd=path, check=False, capture_output=True, text=True, env=env)
    subprocess.run(["git", "add", "."], cwd=path, check=False, capture_output=True, text=True, env=env)
    subprocess.run(["git", "commit", "-m", "initial ahe workspace"], cwd=path, check=False, capture_output=True, text=True, env=env)


def _patch_code_agent_config(
    config_path: Path,
    *,
    api_type: str,
    reasoning: dict[str, Any] | None,
    max_iterations: int | None,
) -> None:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML object in {config_path}")
    llm_config = config.setdefault("llm_config", {})
    if not isinstance(llm_config, dict):
        raise ValueError(f"Expected llm_config object in {config_path}")
    llm_config["api_type"] = api_type
    if reasoning:
        llm_config["reasoning"] = dict(reasoning)
    if max_iterations is not None:
        config["max_iterations"] = max_iterations
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _workspace_diff_summary(workspace: Path) -> dict[str, Any]:
    return _workspace_change_summary({}, _workspace_git_summary(workspace))


def _workspace_git_summary(workspace: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    diff_hash = hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "tree": tree.stdout.strip() if tree.returncode == 0 else None,
        "status": lines,
        "diff_hash": diff_hash,
    }


def _workspace_change_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    tracked_keys = ("head", "tree", "status", "diff_hash")
    changed = any(before.get(key) != after.get(key) for key in tracked_keys)
    return {
        "changed": changed,
        "status": after.get("status", []),
        "before": before,
        "after": after,
    }
