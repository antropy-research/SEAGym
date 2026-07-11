from __future__ import annotations

"""AHE baseline lifecycle implementation."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any

from seagym.costs import extract_token_cost_from_json_file
from seagym.paths import resolve_portable_path

from ..base import BaseBaseline, BaselineState, Checkpoint, UpdateResult
from ..data import TrajectoryBatch
from ..model_mapping import UpdateModelBinding
from ..native_runtime import NativeRuntimeConfig, native_error_result, patched_runtime_env, run_setup_commands
from .evidence import _materialize_ahe_evidence
from .native import _load_ahe_evolve, _run_ahe_post_update_hooks
from .state import _read_ahe_state_metadata
from .workspace import _ensure_git_repo, _patch_code_agent_config, _workspace_change_summary, _workspace_git_summary


@dataclass
class AHEBaseline(BaseBaseline):
    project_dir: Path = Path("reference/agentic-harness-engineering")
    model: str = "deepseek-v4-flash"
    api_base: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_type: str = "openai_chat_completion"
    reasoning: dict[str, Any] | None = None
    max_iterations: int | None = 300
    agent_config_filename: str = "code_agent.yaml"
    native_config: dict[str, Any] = field(default_factory=dict)
    runtime: NativeRuntimeConfig = field(default_factory=NativeRuntimeConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.state_dir = self.state_dir.resolve()
        self.project_dir = self.project_dir.resolve()

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        models: dict[str, Any],
        state_dir: Path,
        run_dir: Path,
        base_dir: Path | None,
    ) -> "AHEBaseline":
        del run_dir
        update_model = UpdateModelBinding.from_config(
            config,
            models,
            default_model="deepseek/deepseek-chat",
        ).openai_compatible_settings()
        project_dir = _resolve_path(config.get("project_dir", "reference/agentic-harness-engineering"), base_dir=base_dir)
        runtime = NativeRuntimeConfig.from_config(config, base_dir=base_dir)
        return cls(
            baseline_id=name,
            state_dir=state_dir.resolve(),
            project_dir=project_dir,
            model=update_model.model,
            api_base=update_model.base_url,
            api_key_env=update_model.api_key_env,
            api_type=str(config.get("api_type", "openai_chat_completion")),
            reasoning=dict(config["reasoning"]) if isinstance(config.get("reasoning"), dict) else None,
            max_iterations=None if config.get("max_iterations") in (None, "") else int(config["max_iterations"]),
            native_config=dict(config.get("native_config") or {}),
            runtime=runtime,
        )

    @property
    def workspace_dir(self) -> Path:
        return self.state_dir.resolve() / "workspace"

    def initialize(self, run_dir: Path) -> BaselineState:
        del run_dir
        self.state_dir = self.state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        source = self.project_dir / "agents" / "code_agent_simple"
        if not source.exists():
            raise FileNotFoundError(f"AHE code_agent_simple source not found: {source}")
        if not self.workspace_dir.exists():
            shutil.copytree(source, self.workspace_dir)
        evolve_source = self.project_dir / "agents" / "evolve_agent"
        evolve_dest = self.state_dir / "evolve_agent"
        if evolve_source.exists() and not evolve_dest.exists():
            shutil.copytree(evolve_source, evolve_dest)
        _ensure_git_repo(self.workspace_dir)
        metadata = {
            "baseline_id": self.baseline_id,
            "project_dir": str(self.project_dir),
            "workspace_dir": str(self.workspace_dir),
            "agent_config_path": str(self.workspace_dir / self.agent_config_filename),
            "model": self.model,
        }
        (self.state_dir / "baseline_state.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return BaselineState(state_dir=self.state_dir, metadata=metadata)

    def prepare_rollout_workspace(self) -> None:
        _patch_code_agent_config(
            self.workspace_dir / self.agent_config_filename,
            api_type=self.api_type,
            reasoning=self.reasoning,
            max_iterations=self.max_iterations,
        )

    def save_checkpoint(self, state: BaselineState, path: Path) -> Checkpoint:
        checkpoint = super().save_checkpoint(state, path)
        manifest = dict(checkpoint.metadata)
        checkpoint_state_path = Path(str(manifest.get("state_ref", path / "baseline_state")))
        persisted_metadata = _read_ahe_state_metadata(checkpoint_state_path)
        if persisted_metadata:
            manifest["state_metadata"] = persisted_metadata
            (path / "checkpoint.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return Checkpoint(checkpoint_dir=checkpoint.checkpoint_dir, state_ref=checkpoint.state_ref, metadata=manifest)

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        self.state_dir = self.state_dir.resolve()
        state = super().load_checkpoint(checkpoint)
        metadata = _read_ahe_state_metadata(state.state_dir)
        metadata.update(
            {
                "baseline_id": self.baseline_id,
                "project_dir": str(self.project_dir),
                "workspace_dir": str(self.workspace_dir),
                "agent_config_path": str(self.workspace_dir / self.agent_config_filename),
                "model": self.model,
                "loaded": True,
            }
        )
        manifest = state.metadata.get("manifest") if isinstance(state.metadata, dict) else None
        if isinstance(manifest, dict):
            metadata["manifest"] = manifest
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "baseline_state.json").write_text(
            json.dumps({key: value for key, value in metadata.items() if key != "manifest"}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return BaselineState(state.state_dir, metadata)

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        self.update_index += 1
        state_dir = state.state_dir.resolve()
        if self.state_dir.resolve() != state_dir:
            self.state_dir = state_dir
        iteration_dir = (state_dir / "updates" / f"ahe_iteration_{self.update_index:04d}").resolve()
        input_dir = iteration_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        trajectories_path = input_dir / "seagym_trajectories.json"
        trajectories_path.write_text(
            json.dumps(trajectories.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evolve = _load_ahe_evolve(self.project_dir)
        native_evidence = _materialize_ahe_evidence(
            evolve=evolve,
            trajectories=trajectories,
            exp_dir=state_dir,
            workspace=self.workspace_dir,
            iteration=self.update_index,
            iteration_dir=iteration_dir,
        )
        query = native_evidence["query"]
        (input_dir / "evolution_query.md").write_text(query, encoding="utf-8")
        config = self._build_native_config()
        before_workspace = _workspace_git_summary(self.workspace_dir)
        status = "updated"
        error = None
        post_update_artifacts: dict[str, Any] = {}
        if self.runtime.enabled:
            setup = run_setup_commands(self.runtime, project_dir=self.project_dir, state_dir=state_dir)
            if setup.get("ok") is False:
                return native_error_result(
                    update_index=self.update_index,
                    update_dir=iteration_dir,
                    error="AHE native runtime setup failed",
                    stage="ahe_runtime_setup",
                    logs={"setup": setup},
                )
        try:
            with patched_runtime_env(self.runtime):
                result = evolve.run_evolve_agent(
                    config=config,
                    exp_dir=state_dir,
                    iteration=self.update_index,
                    query=query,
                    job_dir=Path(native_evidence["job_dir"]).resolve(),
                    iteration_dir=iteration_dir,
                )
                post_update_artifacts = _run_ahe_post_update_hooks(
                    evolve=evolve,
                    exp_dir=state_dir,
                    iteration=self.update_index,
                    iteration_dir=iteration_dir,
                    result=str(result),
                )
        except Exception as exc:  # pragma: no cover - depends on native AHE/LLM runtime.
            result = ""
            status = "error"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        after_workspace = _workspace_git_summary(self.workspace_dir)
        diff_summary = _workspace_change_summary(before_workspace, after_workspace)
        summary = {
            "baseline": "agentic-harness-engineering",
            "trajectories": str(trajectories_path),
            "iteration_dir": str(iteration_dir),
            "native_evidence": native_evidence,
            "workspace_diff": diff_summary,
            "result": str(result),
        }
        if post_update_artifacts:
            summary["post_update_artifacts"] = post_update_artifacts
        if error is not None:
            summary["error"] = error
        cost = extract_token_cost_from_json_file(iteration_dir / "evolve" / "nexau_in_memory_tracer.cleaned.json")
        if cost:
            summary["cost"] = cost
            summary["cost_source"] = "ahe_evolve_cleaned_trace"
        (iteration_dir / "ahe_update_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if status != "error":
            status = "updated" if diff_summary["changed"] else "no_change"
        return UpdateResult(
            update_index=self.update_index,
            changed=diff_summary["changed"],
            status=status,
            metrics={"num_trajectories": len(trajectories.trajectories)},
            logs=summary,
            artifacts={"iteration_dir": str(iteration_dir)},
        )

    def _build_native_config(self) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env, "")
        llm = {
            "api_key": api_key,
            "base_url": self.api_base,
            "model": self.model,
            "api_type": self.api_type,
        }
        if self.reasoning:
            llm["reasoning"] = dict(self.reasoning)
        config = {
            "llm": llm,
            "evolve_agent": {"llm": llm, "llm_config": llm},
            "agent_debugger": {"enabled": False, "llm": {}},
            "notify": {"enabled": False},
        }
        config.update(self.native_config)
        return config


def _resolve_path(value: Any, *, base_dir: Path | None) -> Path:
    return resolve_portable_path(value, base_dir=base_dir)
