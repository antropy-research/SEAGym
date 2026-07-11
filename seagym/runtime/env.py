from __future__ import annotations

"""Environment file loading for local SEAGym runs."""

import os
from pathlib import Path


DEFAULT_ENV_FILE = Path(".env")


def load_env_file(path: str | Path = DEFAULT_ENV_FILE, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for lineno, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Invalid env line {env_path}:{lineno}: {raw_line!r}")
        key = key.strip()
        value = _strip_quotes(value.strip())
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
