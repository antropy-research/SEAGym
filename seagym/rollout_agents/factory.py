from __future__ import annotations

"""Build rollout agents from experiment config."""

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from .base import RolloutAgent


@dataclass(frozen=True)
class RolloutAgentBuild:
    agent_id: str
    rollout_agent: RolloutAgent
    rollout_model: str | None = None


def build_rollout_agent(config: dict[str, Any], *, run_dir: Path, base_dir: Path | None = None) -> RolloutAgentBuild:
    agent_config = config.get("rollout_agent") or {}
    if not isinstance(agent_config, dict):
        raise ValueError("rollout_agent config must be an object")
    class_path = agent_config.get("class_path")
    if not class_path:
        raise ValueError("rollout_agent.class_path is required")
    rollout_agent = _build_class_path_rollout_agent(
        str(class_path),
        agent_config,
        run_dir=run_dir,
        base_dir=base_dir,
    )
    return RolloutAgentBuild(
        agent_id=rollout_agent.agent_id,
        rollout_agent=rollout_agent,
        rollout_model=_configured_model(agent_config),
    )


def _build_class_path_rollout_agent(
    class_path: str,
    agent_config: dict[str, Any],
    *,
    run_dir: Path,
    base_dir: Path | None,
) -> RolloutAgent:
    cls = _import_class(class_path)
    name = str(agent_config.get("name") or agent_config.get("id") or "rollout-agent")
    raw_config = dict(agent_config.get("config") or {})
    models = agent_config.get("models") or {}
    if hasattr(cls, "from_config"):
        return cls.from_config(  # type: ignore[no-any-return, attr-defined]
            name=name,
            config=raw_config,
            models=models,
            run_dir=run_dir,
            base_dir=base_dir,
        )
    return cls(agent_id=name, **raw_config)


def _import_class(class_path: str) -> type[Any]:
    module_name, sep, attr = class_path.partition(":")
    if not sep:
        module_name, _, attr = class_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid rollout_agent.class_path: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _configured_model(agent_config: dict[str, Any]) -> str | None:
    config = agent_config.get("config") or {}
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    if model not in (None, ""):
        return str(model)
    model_ref = config.get("model_ref") or "rollout_model"
    models = agent_config.get("models") or {}
    if isinstance(models, dict) and isinstance(models.get(model_ref), dict):
        value = models[model_ref].get("model")
        return None if value in (None, "") else str(value)
    return None
