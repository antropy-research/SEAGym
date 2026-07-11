from __future__ import annotations

"""Small utilities for method-specific baseline adapters."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from typing import Any

from seagym.paths import resolve_optional_portable_path


def resolve_optional_path(value: Any, *, base_dir: Path | None) -> Path | None:
    return resolve_optional_portable_path(value, base_dir=base_dir)


def optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


@contextmanager
def prepend_sys_path(path: Path):
    resolved = str(path.resolve())
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        try:
            sys.path.remove(resolved)
        except ValueError:
            pass


def load_import_path(import_path: str) -> Any:
    module_name, sep, attr = import_path.partition(":")
    if not sep:
        module_name, _, attr = import_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {import_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
