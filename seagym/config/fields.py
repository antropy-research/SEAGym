from __future__ import annotations

"""Small helpers for reading loose experiment config dictionaries."""

import os
from pathlib import Path
from typing import Any, Mapping

from seagym.paths import resolve_portable_path


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be an object")
    return section


def config_get(config: Mapping[str, Any], section: str, key: str, *, default: Any = None) -> Any:
    return config_section(config, section).get(key, default)


def as_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected object value, got: {value!r}")
    return {
        str(key): expand_env_template(item)
        for key, item in value.items()
    }


def as_str_dict(value: Any) -> dict[str, str]:
    return {key: str(item) for key, item in as_dict(value).items()}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list value, got: {value!r}")
    return [str(expand_env_template(item)) for item in value]


def optional_resolved_path(value: Any, *, base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_portable_path(expand_env_template(value), base_dir=base_dir)


def required_path(value: Any, *, section: str, key: str, base_dir: Path | None = None) -> Path:
    if value in (None, ""):
        raise ValueError(f"Config section {section!r} must define {key!r}")
    return resolve_portable_path(expand_env_template(value), base_dir=base_dir)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def expand_env_templates(values: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(expand_env_template(value)) for key, value in values.items()}


def expand_env_template(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}") and len(value) > 3:
        return os.environ.get(value[2:-1], "")
    return value


def resolve_path(base_dir: Path, value: object) -> Path:
    return resolve_portable_path(value, base_dir=base_dir)
