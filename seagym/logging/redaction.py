from __future__ import annotations

"""Shared redaction helpers for run artifacts and runtime diagnostics."""

from typing import Any


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if sensitive_key(str(key)) else redact_sensitive(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        if "=" in value:
            key, sep, raw = value.partition("=")
            if sensitive_key(key):
                return f"{key}{sep}<redacted>"
            if key.upper().endswith("PROXY"):
                return f"{key}{sep}{redact_url(raw)}"
        return redact_url(value)
    return value


def sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"))


def redact_url(value: str) -> str:
    if not value or "@" not in value:
        return value
    scheme, sep, rest = value.partition("://")
    _, _, host = rest.rpartition("@")
    if sep:
        return f"{scheme}://***@{host}"
    return f"***@{host}"
