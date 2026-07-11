from __future__ import annotations

"""Utilities for normalizing method/provider token usage."""

import json
from pathlib import Path
from typing import Any


TOKEN_TOTAL_KEYS = ("total_tokens", "totalTokens", "n_total_tokens", "tokens")
INPUT_TOKEN_KEYS = ("input_tokens", "prompt_tokens", "n_input_tokens")
OUTPUT_TOKEN_KEYS = ("output_tokens", "completion_tokens", "n_output_tokens")
CACHE_TOKEN_KEYS = ("cache_tokens", "cached_tokens", "cache_read_tokens", "n_cache_tokens")


def extract_token_cost(value: Any) -> dict[str, float]:
    """Extract minimal numeric token cost from structured provider/native output.

    Prefer an already-normalized aggregate on the current object. Only recurse
    when the current object does not expose aggregate usage, which avoids double
    counting files that contain both top-level totals and per-span usage.
    """
    usage = _extract_usage(value)
    if not usage:
        return {}
    cost: dict[str, float] = {"total_tokens": usage["total_tokens"]}
    if usage.get("input_tokens"):
        cost["input_tokens"] = usage["input_tokens"]
    if usage.get("output_tokens"):
        cost["output_tokens"] = usage["output_tokens"]
    if usage.get("cache_tokens"):
        cost["cache_tokens"] = usage["cache_tokens"]
    return cost


def extract_token_cost_from_json_file(path: str | Path) -> dict[str, float]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return extract_token_cost(json.loads(file_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_usage(value: Any) -> dict[str, float]:
    object_usage = _object_usage(value)
    if object_usage:
        return object_usage
    if isinstance(value, dict):
        direct = _direct_usage(value)
        if direct:
            return direct
        total = 0.0
        input_tokens = 0.0
        output_tokens = 0.0
        cache_tokens = 0.0
        for child in value.values():
            child_usage = _extract_usage(child)
            total += child_usage.get("total_tokens", 0.0)
            input_tokens += child_usage.get("input_tokens", 0.0)
            output_tokens += child_usage.get("output_tokens", 0.0)
            cache_tokens += child_usage.get("cache_tokens", 0.0)
        return _usage(total, input_tokens, output_tokens, cache_tokens)
    if isinstance(value, list):
        total = 0.0
        input_tokens = 0.0
        output_tokens = 0.0
        cache_tokens = 0.0
        for child in value:
            child_usage = _extract_usage(child)
            total += child_usage.get("total_tokens", 0.0)
            input_tokens += child_usage.get("input_tokens", 0.0)
            output_tokens += child_usage.get("output_tokens", 0.0)
            cache_tokens += child_usage.get("cache_tokens", 0.0)
        return _usage(total, input_tokens, output_tokens, cache_tokens)
    return {}


def _object_usage(value: Any) -> dict[str, float]:
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return {}
    usage_method = getattr(value, "usage", None)
    if callable(usage_method):
        try:
            usage = usage_method()
        except Exception:
            usage = None
        cost = _usage_from_attrs(usage)
        if cost:
            return cost
    for attr in ("raw", "output", "result"):
        child = getattr(value, attr, None)
        cost = _extract_usage(child)
        if cost:
            return cost
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            cost = _extract_usage(model_dump())
        except Exception:
            cost = {}
        if cost:
            return cost
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            cost = _extract_usage(as_dict())
        except Exception:
            cost = {}
        if cost:
            return cost
    object_dict = getattr(value, "__dict__", None)
    if isinstance(object_dict, dict):
        return _extract_usage(object_dict)
    return {}


def _direct_usage(value: dict[str, Any]) -> dict[str, float]:
    usage_obj = value.get("usage")
    if isinstance(usage_obj, dict):
        direct = _direct_usage(usage_obj)
        if direct:
            return direct
    total = _first_number(value, TOKEN_TOTAL_KEYS)
    input_tokens = _first_number(value, INPUT_TOKEN_KEYS)
    output_tokens = _first_number(value, OUTPUT_TOKEN_KEYS)
    cache_tokens = _first_number(value, CACHE_TOKEN_KEYS)
    if total is None and any(token is not None for token in (input_tokens, output_tokens, cache_tokens)):
        total = float(input_tokens or 0.0) + float(output_tokens or 0.0) + float(cache_tokens or 0.0)
    if total is None:
        return {}
    return _usage(float(total), float(input_tokens or 0.0), float(output_tokens or 0.0), float(cache_tokens or 0.0))


def _usage_from_attrs(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _direct_usage(value)
    total = _first_attr_number(value, TOKEN_TOTAL_KEYS)
    input_tokens = _first_attr_number(value, INPUT_TOKEN_KEYS)
    output_tokens = _first_attr_number(value, OUTPUT_TOKEN_KEYS)
    cache_tokens = _first_attr_number(value, CACHE_TOKEN_KEYS)
    if total is None and any(token is not None for token in (input_tokens, output_tokens, cache_tokens)):
        total = float(input_tokens or 0.0) + float(output_tokens or 0.0) + float(cache_tokens or 0.0)
    if total is None:
        return {}
    return _usage(float(total), float(input_tokens or 0.0), float(output_tokens or 0.0), float(cache_tokens or 0.0))


def _usage(total: float, input_tokens: float = 0.0, output_tokens: float = 0.0, cache_tokens: float = 0.0) -> dict[str, float]:
    if total <= 0:
        return {}
    usage = {"total_tokens": total}
    if input_tokens:
        usage["input_tokens"] = input_tokens
    if output_tokens:
        usage["output_tokens"] = output_tokens
    if cache_tokens:
        usage["cache_tokens"] = cache_tokens
    return usage


def _first_number(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        token_value = value.get(key)
        if isinstance(token_value, int | float):
            return float(token_value)
    return None


def _first_attr_number(value: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        token_value = getattr(value, key, None)
        if isinstance(token_value, int | float):
            return float(token_value)
    return None
