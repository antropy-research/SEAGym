from __future__ import annotations

"""Portable path anchors for release configs."""

import os
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
_UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


def resolve_portable_path(
    value: Any,
    *,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve repo/data/results anchors plus existing absolute/relative paths."""

    if value in (None, ""):
        raise ValueError("Expected path value")
    text = _expand_env_vars(str(value))
    root = repo_root or REPO_ROOT
    for scheme, env_name in (
        ("repo://", None),
        ("data://", "SEAGYM_DATA_ROOT"),
        ("results://", "SEAGYM_RESULTS_ROOT"),
    ):
        if text.startswith(scheme):
            suffix = text[len(scheme) :].lstrip("/")
            anchor = root if env_name is None else _required_env_path(env_name)
            return anchor / suffix
    path = Path(text)
    if path.is_absolute():
        return path
    return ((base_dir or Path.cwd()) / path).resolve()


def resolve_optional_portable_path(
    value: Any,
    *,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_portable_path(value, base_dir=base_dir, repo_root=repo_root)


def has_portable_anchor(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("repo://", "data://", "results://")) or "$" in value


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set to resolve portable release paths")
    return Path(_expand_env_vars(value))


def _expand_env_vars(value: str) -> str:
    expanded = os.path.expandvars(value)
    match = _UNRESOLVED_ENV.search(expanded)
    if match:
        raise ValueError(f"Environment variable is not set in path: {value!r}")
    return expanded
