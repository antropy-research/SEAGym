from __future__ import annotations

"""Model configuration objects used by baselines."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str = "litellm"
    api_base: str | None = None
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
