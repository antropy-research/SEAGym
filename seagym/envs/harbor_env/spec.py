from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HarborAgentSpec:
    """Configuration needed by HarborEnv to instantiate a Harbor agent."""

    agent_id: str
    import_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    n_attempts: int = 1


PROXY_ENV_KEYS = {
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
}
