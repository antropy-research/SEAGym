from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapter_utils import resolve_optional_path


def _redacted_lm_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    redacted = dict(kwargs or {})
    if "api_key" in redacted:
        redacted["api_key"] = "<redacted>"
    return redacted


def _redacted_native_adapter(config: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(config)
    for key in list(redacted):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "<redacted>"
    return redacted


def _resolve_native_adapter_paths(config: dict[str, Any], *, base_dir: Path | None) -> dict[str, Any]:
    values = dict(config)
    if values.get("instruction_prompt_path") not in (None, ""):
        values["instruction_prompt_path"] = str(resolve_optional_path(values["instruction_prompt_path"], base_dir=base_dir))
    return values
