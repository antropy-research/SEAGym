from __future__ import annotations

"""Rollout agents backed by Harbor task execution."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from seagym.baselines.base import BaselineState
from seagym.baselines.data import TaskBatch, TrajectoryBatch
from seagym.data.types import TaskIndex
from seagym.envs.base import TaskEnv
from seagym.envs.harbor_env import HarborAgentSpec

from .base import RolloutAgentState


@dataclass
class HarborRolloutAgent:
    agent_id: str
    agent_import_path: str | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    n_attempts: int = 1
    attempt_modes: tuple[str, ...] = ("train",)

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        models: dict[str, Any],
        run_dir: Path,
        base_dir: Path | None,
    ) -> "HarborRolloutAgent":
        del run_dir, base_dir
        model_ref = str(config.get("model_ref", "rollout_model"))
        return cls(
            agent_id=str(config.get("agent", name)),
            agent_import_path=None if config.get("import_path") in (None, "") else str(config.get("import_path")),
            agent_kwargs=dict(config.get("kwargs") or {}),
            agent_env=_model_env(models.get(model_ref)),
            n_attempts=max(1, int(config.get("n_attempts", 1))),
            attempt_modes=_attempt_modes(config.get("attempt_modes", ["train"])),
        )

    def initialize(self, run_dir: Path) -> RolloutAgentState:
        del run_dir
        return RolloutAgentState({"agent_id": self.agent_id, "agent_import_path": self.agent_import_path})

    def rollout(
        self,
        batch: TaskBatch,
        *,
        env: TaskEnv,
        task_index: TaskIndex,
        baseline_state: BaselineState,
    ) -> TrajectoryBatch:
        n_attempts = self._n_attempts_for_mode(batch.mode)
        agent_spec = self.harbor_agent_spec(baseline_state, n_attempts=n_attempts)
        if hasattr(env, "configure_agent_spec"):
            env.configure_agent_spec(agent_spec)  # type: ignore[attr-defined]
        tasks = [task_index.require(task_id) for task_id in batch.task_ids]
        if n_attempts > 1 and hasattr(env, "run_task_attempts"):
            results = env.run_task_attempts(  # type: ignore[attr-defined]
                tasks,
                view_name=batch.view_name,
                mode=batch.mode,
                agent_id=self.agent_id,
            )
        else:
            results = env.run_tasks(tasks, view_name=batch.view_name, mode=batch.mode, agent_id=self.agent_id)
        return TrajectoryBatch.from_task_results(
            results,
            task_ids=batch.task_ids,
            view_name=batch.view_name,
            mode=batch.mode,
            batch_index=batch.batch_index,
            epoch=batch.epoch,
            refs={"agent_id": self.agent_id, "n_attempts": n_attempts},
        )

    def harbor_agent_spec(
        self,
        baseline_state: BaselineState | None = None,
        *,
        n_attempts: int | None = None,
    ) -> HarborAgentSpec:
        kwargs = dict(self.agent_kwargs)
        if baseline_state is not None and "prompt_template_path" in baseline_state.metadata:
            prompt_template_path = _harbor_prompt_template_path(
                Path(str(baseline_state.metadata["prompt_template_path"])).resolve()
            )
            kwargs["prompt_template_path"] = str(prompt_template_path)
        return HarborAgentSpec(
            agent_id=self.agent_id,
            import_path=self.agent_import_path,
            kwargs=kwargs,
            env=dict(self.agent_env),
            n_attempts=max(1, int(n_attempts if n_attempts is not None else self.n_attempts)),
        )

    def _n_attempts_for_mode(self, mode: str) -> int:
        if mode in self.attempt_modes:
            return self.n_attempts
        return 1


def _model_env(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    exports = raw.get("exports") or {}
    if not isinstance(exports, dict):
        return {}
    values = {
        "model": str(raw.get("model", "")),
        "api_base": str(raw.get("api_base", "")),
        "api_key": os.environ.get(str(raw.get("api_key_env", "")), ""),
    }
    return {str(key): str(value).format(**values) for key, value in exports.items()}


def _attempt_modes(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(str(mode) for mode in raw)
    return ("train",)


def _harbor_prompt_template_path(source_path: Path) -> Path:
    """Return a Harbor-compatible prompt template for baseline prompt state."""
    if not source_path.exists():
        rendered_path = source_path.with_name(f"{source_path.stem}.harbor{source_path.suffix}")
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path.write_text(_render_harbor_prompt_template(""), encoding="utf-8")
        return rendered_path
    text = source_path.read_text(encoding="utf-8")
    if "{{ instruction }}" in text:
        return source_path
    rendered_path = source_path.with_name(f"{source_path.stem}.harbor{source_path.suffix}")
    rendered_path.write_text(_render_harbor_prompt_template(text), encoding="utf-8")
    return rendered_path


def _render_harbor_prompt_template(prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return "{{ instruction }}\n"
    return "{% raw %}\n" + prefix + "\n{% endraw %}\n\n{{ instruction }}\n"
