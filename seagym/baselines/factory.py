from __future__ import annotations

"""Build baseline lifecycle objects from experiment config."""

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from seagym.paths import resolve_optional_portable_path

from .base import Baseline


@dataclass(frozen=True)
class BaselineBuild:
    agent_id: str
    baseline: Baseline
    rollout_model: str | None = None


def build_baseline(config: dict[str, Any], *, run_dir: Path, base_dir: Path | None = None) -> BaselineBuild:
    baseline_config = config.get("baseline") or {}
    if not isinstance(baseline_config, dict):
        raise ValueError("baseline config must be an object")
    class_path = baseline_config.get("class_path")
    if not class_path:
        raise ValueError("baseline.class_path is required")
    baseline = _build_class_path_baseline(
        str(class_path),
        baseline_config,
        run_dir=run_dir,
        base_dir=base_dir,
    )
    return BaselineBuild(
        agent_id=baseline.baseline_id,
        baseline=baseline,
        rollout_model=_configured_model(baseline_config),
    )


def _build_class_path_baseline(
    class_path: str,
    baseline_config: dict[str, Any],
    *,
    run_dir: Path,
    base_dir: Path | None,
) -> Baseline:
    cls = _import_class(class_path)
    name = str(baseline_config.get("name") or baseline_config.get("id") or "baseline")
    raw_config = dict(baseline_config.get("config") or {})
    state_config = baseline_config.get("state") or {}
    if not isinstance(state_config, dict):
        raise ValueError("baseline.state must be an object")
    state_dir = _resolve_optional_path(state_config.get("local_dir"), base_dir=base_dir) or run_dir / "agent_state" / name
    models = baseline_config.get("models") or {}
    if hasattr(cls, "from_config"):
        return cls.from_config(  # type: ignore[no-any-return, attr-defined]
            name=name,
            config=raw_config,
            models=models,
            state_dir=state_dir,
            run_dir=run_dir,
            base_dir=base_dir,
        )
    return cls(baseline_id=name, state_dir=state_dir, **raw_config)


def _import_class(class_path: str) -> type[Any]:
    module_name, sep, attr = class_path.partition(":")
    if not sep:
        module_name, _, attr = class_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid baseline.class_path: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_optional_path(value: Any, *, base_dir: Path | None) -> Path | None:
    return resolve_optional_portable_path(value, base_dir=base_dir)


def _configured_model(baseline_config: dict[str, Any]) -> str | None:
    config = baseline_config.get("config") or {}
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    if model not in (None, ""):
        return str(model)
    model_ref = config.get("model_ref") or config.get("rollout_model_ref") or "rollout_model"
    models = baseline_config.get("models") or {}
    if isinstance(models, dict) and isinstance(models.get(model_ref), dict):
        value = models[model_ref].get("model")
        return None if value in (None, "") else str(value)
    return None
