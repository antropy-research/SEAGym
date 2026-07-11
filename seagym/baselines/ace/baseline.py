from __future__ import annotations

"""ACE baseline adapter."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from seagym.costs import extract_token_cost

from ..adapter_utils import jsonable, optional_int, prepend_sys_path, resolve_optional_path
from ..base import BaseBaseline, BaselineState, Checkpoint, UpdateResult
from ..data import TrajectoryBatch
from ..model_mapping import UpdateModelBinding
from ..native_runtime import (
    NativeRuntimeConfig,
    native_error_result,
    patched_runtime_env,
    run_python_script,
    run_setup_commands,
    safe_native_update,
)
from .batch import _batch_reflect_then_update
from .runner import _ACE_RUNTIME_RUNNER
from .state import _file_sha256, _read_text_file, _skillbook_num_skills, _write_state_metadata
from .traces import _materialize_ace_traces


@dataclass
class ACEBaseline(BaseBaseline):
    project_dir: Path | None = None
    max_update_records: int | None = None
    model: str = "gpt-4o-mini"
    epochs: int = 1
    wait: bool = True
    skillbook_filename: str = "skillbook.json"
    prompt_filename: str = "ace_prompt.md"
    update_prompt_variant: str = "ace_default"
    reflector_prompt_path: Path | None = None
    skill_manager_prompt_path: Path | None = None
    skill_manager_system_prompt_path: Path | None = None
    update_mode: str = "native_trace_analyser"
    trace_format: str = "ace_standard"
    feedback_mode: str = "reward_only"
    max_reasoning_chars: int = 20000
    runtime: NativeRuntimeConfig = field(default_factory=NativeRuntimeConfig)

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
    ) -> "ACEBaseline":
        del run_dir
        update_model = UpdateModelBinding.from_config(config, models, default_model="gpt-4o-mini")
        model = update_model.pydantic_ai_model()
        model_env = update_model.pydantic_ai_env(model)
        runtime = NativeRuntimeConfig.from_config(config, base_dir=base_dir)
        if model_env:
            runtime = replace(runtime, env={**model_env, **runtime.env})
        update_prompt_variant = str(config.get("update_prompt_variant", "ace_default"))
        reflector_prompt_path = resolve_optional_path(config.get("reflector_prompt_path"), base_dir=base_dir)
        skill_manager_prompt_path = resolve_optional_path(config.get("skill_manager_prompt_path"), base_dir=base_dir)
        skill_manager_system_prompt_path = resolve_optional_path(config.get("skill_manager_system_prompt_path"), base_dir=base_dir)
        if update_prompt_variant != "ace_default" and not any(
            (reflector_prompt_path, skill_manager_prompt_path, skill_manager_system_prompt_path)
        ):
            raise ValueError("ACE update_prompt_variant values other than 'ace_default' require at least one custom prompt path")
        update_mode = str(config.get("update_mode", "native_trace_analyser"))
        if update_mode not in {"native_trace_analyser", "batch_reflect_then_update"}:
            raise ValueError(f"Unsupported ACE update_mode: {update_mode}")
        return cls(
            baseline_id=name,
            state_dir=state_dir,
            project_dir=resolve_optional_path(config.get("project_dir"), base_dir=base_dir),
            max_update_records=optional_int(config.get("max_update_records")),
            model=model,
            epochs=int(config.get("epochs", 1)),
            wait=bool(config.get("wait", True)),
            skillbook_filename=str(config.get("skillbook_filename", "skillbook.json")),
            prompt_filename=str(config.get("prompt_filename", "ace_prompt.md")),
            update_prompt_variant=update_prompt_variant,
            reflector_prompt_path=reflector_prompt_path,
            skill_manager_prompt_path=skill_manager_prompt_path,
            skill_manager_system_prompt_path=skill_manager_system_prompt_path,
            update_mode=update_mode,
            trace_format=str(config.get("trace_format", "ace_standard")),
            feedback_mode=str(config.get("feedback_mode", "reward_only")),
            max_reasoning_chars=int(config.get("max_reasoning_chars", 20000)),
            runtime=runtime,
        )

    def initialize(self, run_dir: Path) -> BaselineState:
        state = super().initialize(run_dir)
        prompt_path = self.state_dir / self.prompt_filename
        if not prompt_path.exists():
            prompt_path.write_text("", encoding="utf-8")
        state.metadata.update(
            {
                "native_method": "ACE",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "model": self.model,
                "runtime_env_keys": sorted(self.runtime.env),
                "skillbook_path": str(self.state_dir / self.skillbook_filename),
                "prompt_template_path": str(prompt_path),
                "update_prompt_config": self._update_prompt_config(),
                "update_mode": self.update_mode,
                "trace_format": self.trace_format,
                "feedback_mode": self.feedback_mode,
                "max_reasoning_chars": self.max_reasoning_chars,
                "runtime_enabled": self.runtime.enabled,
            }
        )
        return state

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        state = super().load_checkpoint(checkpoint)
        prompt_path = self.state_dir / self.prompt_filename
        skillbook_path = self.state_dir / self.skillbook_filename
        state.metadata.update(
            {
                "baseline_id": self.baseline_id,
                "native_method": "ACE",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "model": self.model,
                "runtime_env_keys": sorted(self.runtime.env),
                "skillbook_path": str(skillbook_path),
                "prompt_template_path": str(prompt_path),
                "update_prompt_config": self._update_prompt_config(),
                "update_mode": self.update_mode,
                "trace_format": self.trace_format,
                "feedback_mode": self.feedback_mode,
                "max_reasoning_chars": self.max_reasoning_chars,
                "runtime_enabled": self.runtime.enabled,
                "loaded": True,
            }
        )
        _write_state_metadata(self.state_dir, state.metadata)
        return state

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        update_dir = self.next_update_dir(state, "ace")
        records = self._write_update_traces(trajectories, update_dir)
        if self.runtime.enabled:
            return self._update_with_runtime(records, state, update_dir)
        return safe_native_update(
            update_index=self.update_index,
            update_dir=update_dir,
            stage="ace_direct_update",
            fn=lambda: self._update_direct(records, state, update_dir),
        )

    def _update_direct(self, records: list[dict[str, Any]], state: BaselineState, update_dir: Path) -> UpdateResult:
        if self.project_dir is None:
            raise RuntimeError("ACEBaseline requires config.project_dir")
        skillbook_path = state.state_dir / self.skillbook_filename
        before_hash = _file_sha256(skillbook_path)
        with patched_runtime_env(self.runtime):
            with prepend_sys_path(self.project_dir):
                from ace import ACELiteLLM  # type: ignore

                kwargs: dict[str, Any] = {
                    "checkpoint_dir": update_dir / "checkpoints",
                }
                if skillbook_path.exists():
                    kwargs["skillbook_path"] = str(skillbook_path)
                kwargs.update(self._custom_update_role_kwargs())
                agent = ACELiteLLM(self.model, **kwargs)
            if self.update_mode == "native_trace_analyser":
                results = agent.learn_from_traces(records, epochs=self.epochs, wait=self.wait)
            else:
                results = _batch_reflect_then_update(agent, records, epochs=self.epochs, update_dir=update_dir)
            agent.save(str(skillbook_path))
            strategies = agent.get_strategies()
        after_hash = _file_sha256(skillbook_path)
        num_skills = _skillbook_num_skills(skillbook_path)
        changed = before_hash != after_hash and (bool(str(strategies).strip()) or num_skills > 0)
        (update_dir / "strategies.md").write_text(strategies, encoding="utf-8")
        prompt_path = state.state_dir / self.prompt_filename
        prompt_path.write_text(strategies, encoding="utf-8")
        state.metadata["prompt_template_path"] = str(prompt_path)
        cost = extract_token_cost(results)
        jsonable_results = jsonable(results)
        (update_dir / "ace_results.json").write_text(
            json.dumps(jsonable_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logs: dict[str, Any] = {}
        if cost:
            logs["cost"] = cost
            logs["cost_source"] = "ace_results_usage"
        return UpdateResult(
            update_index=self.update_index,
            changed=changed,
            status="updated" if changed else "unchanged",
            metrics={
                "num_trajectories": len(records),
                "num_results": len(results),
                "num_skills": num_skills,
            },
            logs=logs,
            artifacts={"skillbook_path": str(skillbook_path), "update_dir": str(update_dir)},
        )

    def _write_update_traces(self, trajectories: TrajectoryBatch, update_dir: Path) -> list[dict[str, Any]]:
        if self.trace_format == "raw":
            return self.write_trajectories(trajectories, update_dir, max_records=self.max_update_records)
        if self.trace_format != "ace_standard":
            raise ValueError(f"Unsupported ACE trace_format: {self.trace_format}")
        raw_records = trajectories.to_dict()["trajectories"]
        if self.max_update_records is not None:
            raw_records = raw_records[: self.max_update_records]
        (update_dir / "seagym_trajectories.json").write_text(
            json.dumps({"records": raw_records, "batch": trajectories.to_dict()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records = _materialize_ace_traces(
            trajectories,
            max_records=self.max_update_records,
            feedback_mode=self.feedback_mode,
            max_reasoning_chars=self.max_reasoning_chars,
        )
        (update_dir / "trajectories.json").write_text(
            json.dumps({"records": records, "batch": trajectories.to_dict(), "trace_format": "ace_standard"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (update_dir / "ace_traces.json").write_text(
            json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return records

    def _update_with_runtime(self, records: list[dict[str, Any]], state: BaselineState, update_dir: Path) -> UpdateResult:
        setup = run_setup_commands(self.runtime, project_dir=self.project_dir, state_dir=state.state_dir)
        if setup.get("ok") is False:
            return native_error_result(
                update_index=self.update_index,
                update_dir=update_dir,
                error="ACE native runtime setup failed",
                stage="ace_runtime_setup",
                logs={"setup": setup},
            )
        script_path = (update_dir / "ace_runtime_update.py").resolve()
        output_path = (update_dir / "runtime_update_result.json").resolve()
        script_path.write_text(_ACE_RUNTIME_RUNNER, encoding="utf-8")
        args = [
            str((update_dir / "trajectories.json").resolve()),
            str(state.state_dir.resolve()),
            str(update_dir.resolve()),
            "" if self.project_dir is None else str(self.project_dir.resolve()),
            self.model,
            str(self.epochs),
            "1" if self.wait else "0",
            self.skillbook_filename,
            self.prompt_filename,
            self.update_prompt_variant,
            "" if self.reflector_prompt_path is None else str(self.reflector_prompt_path.resolve()),
            "" if self.skill_manager_prompt_path is None else str(self.skill_manager_prompt_path.resolve()),
            "" if self.skill_manager_system_prompt_path is None else str(self.skill_manager_system_prompt_path.resolve()),
            self.update_mode,
            str(output_path),
        ]
        completed = run_python_script(self.runtime, script_path=script_path, args=args, cwd=self.project_dir)
        logs = {"setup": setup, "return_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode != 0:
            return native_error_result(
                update_index=self.update_index,
                update_dir=update_dir,
                error="ACE native runtime update failed",
                stage="ace_runtime_update",
                logs=logs,
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        prompt_path = state.state_dir / self.prompt_filename
        state.metadata["prompt_template_path"] = str(prompt_path)
        return UpdateResult(
            update_index=self.update_index,
            changed=bool(result.get("changed", True)),
            status=str(result.get("status", "updated")),
            metrics=dict(result.get("metrics") or {}),
            logs={
                **logs,
                **({"error": result["error"]} if isinstance(result.get("error"), dict) else {}),
                **({"cost": result["cost"], "cost_source": result.get("cost_source")} if isinstance(result.get("cost"), dict) else {}),
            },
            artifacts=dict(result.get("artifacts") or {}),
        )

    def _update_prompt_config(self) -> dict[str, Any]:
        return {
            "variant": self.update_prompt_variant,
            "reflector_prompt_path": None if self.reflector_prompt_path is None else str(self.reflector_prompt_path),
            "skill_manager_prompt_path": None if self.skill_manager_prompt_path is None else str(self.skill_manager_prompt_path),
            "skill_manager_system_prompt_path": (
                None if self.skill_manager_system_prompt_path is None else str(self.skill_manager_system_prompt_path)
            ),
        }

    def _custom_update_role_kwargs(self) -> dict[str, Any]:
        if not any((self.reflector_prompt_path, self.skill_manager_prompt_path, self.skill_manager_system_prompt_path)):
            return {}
        from ace.implementations import Reflector, SkillManager  # type: ignore

        kwargs: dict[str, Any] = {}
        if self.reflector_prompt_path is not None:
            kwargs["reflector"] = Reflector(self.model, prompt_template=_read_text_file(self.reflector_prompt_path))
        if self.skill_manager_prompt_path is not None or self.skill_manager_system_prompt_path is not None:
            skill_manager_kwargs: dict[str, Any] = {}
            if self.skill_manager_prompt_path is not None:
                skill_manager_kwargs["prompt_template"] = _read_text_file(self.skill_manager_prompt_path)
            if self.skill_manager_system_prompt_path is not None:
                skill_manager_kwargs["system_prompt"] = _read_text_file(self.skill_manager_system_prompt_path)
            kwargs["skill_manager"] = SkillManager(self.model, **skill_manager_kwargs)
        return kwargs

