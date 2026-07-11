from __future__ import annotations

"""TF-GRPO baseline adapter."""

from dataclasses import dataclass, field, replace
import importlib
import json
from pathlib import Path
from typing import Any

from seagym.costs import extract_token_cost

from ..adapter_utils import optional_int, prepend_sys_path, resolve_optional_path
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
from .metering import _TFGRPOMeter, _patch_experience_updater_llm
from .prompts import _apply_update_prompt_profile, _canonical_json, _format_experiences_prompt, _patch_tf_grpo_dotenv_lookup
from .runner import _TF_GRPO_RUNTIME_RUNNER
from .state import _write_state_metadata
from .transforms import _filter_update_rollouts, _rollout_diagnostics, _to_tf_grpo_rollout


@dataclass
class TFGRPOBaseline(BaseBaseline):
    project_dir: Path | None = None
    max_update_records: int | None = None
    domain: str = "math"
    experiences_filename: str = "experiences.json"
    prompt_filename: str = "tf_grpo_experiences.md"
    max_workers: int = 4
    given_ground_truth: bool = False
    only_partial_correct: bool = False
    update_prompt_profile: str = "native"
    skip_metadata_fallback: bool = True
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
    ) -> "TFGRPOBaseline":
        del run_dir
        update_model = UpdateModelBinding.from_config(config, models, default_model="deepseek/deepseek-chat")
        runtime = NativeRuntimeConfig.from_config(config, base_dir=base_dir)
        runtime = replace(runtime, env={**update_model.openai_compatible_env(), **runtime.env})
        return cls(
            baseline_id=name,
            state_dir=state_dir,
            project_dir=resolve_optional_path(config.get("project_dir"), base_dir=base_dir),
            max_update_records=optional_int(config.get("max_update_records")),
            domain=str(config.get("domain", "math")),
            experiences_filename=str(config.get("experiences_filename", "experiences.json")),
            prompt_filename=str(config.get("prompt_filename", "tf_grpo_experiences.md")),
            max_workers=int(config.get("max_workers", 4)),
            given_ground_truth=bool(config.get("given_ground_truth", False)),
            only_partial_correct=bool(config.get("only_partial_correct", False)),
            update_prompt_profile=str(config.get("update_prompt_profile", "native")),
            skip_metadata_fallback=bool(config.get("skip_metadata_fallback", True)),
            runtime=runtime,
        )

    def initialize(self, run_dir: Path) -> BaselineState:
        state = super().initialize(run_dir)
        experience_path = self.state_dir / self.experiences_filename
        if not experience_path.exists():
            experience_path.write_text("{}\n", encoding="utf-8")
        prompt_path = self.state_dir / self.prompt_filename
        if not prompt_path.exists():
            prompt_path.write_text("", encoding="utf-8")
        state.metadata.update(
            {
                "native_method": "TF-GRPO",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "domain": self.domain,
                "update_prompt_profile": self.update_prompt_profile,
                "skip_metadata_fallback": self.skip_metadata_fallback,
                "runtime_env_keys": sorted(self.runtime.env),
                "experience_path": str(experience_path),
                "prompt_template_path": str(prompt_path),
                "runtime_enabled": self.runtime.enabled,
            }
        )
        return state

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        state = super().load_checkpoint(checkpoint)
        experience_path = self.state_dir / self.experiences_filename
        prompt_path = self.state_dir / self.prompt_filename
        state.metadata.update(
            {
                "baseline_id": self.baseline_id,
                "native_method": "TF-GRPO",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "domain": self.domain,
                "update_prompt_profile": self.update_prompt_profile,
                "skip_metadata_fallback": self.skip_metadata_fallback,
                "runtime_env_keys": sorted(self.runtime.env),
                "experience_path": str(experience_path),
                "prompt_template_path": str(prompt_path),
                "runtime_enabled": self.runtime.enabled,
                "loaded": True,
            }
        )
        _write_state_metadata(self.state_dir, state.metadata)
        return state

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        update_dir = self.next_update_dir(state, "tf_grpo")
        records = self.write_trajectories(trajectories, update_dir, max_records=self.max_update_records)
        if self.runtime.enabled:
            return self._update_with_runtime(records, state, update_dir)
        return safe_native_update(
            update_index=self.update_index,
            update_dir=update_dir,
            stage="tf_grpo_direct_update",
            fn=lambda: self._update_direct(records, state, update_dir),
        )

    def _update_direct(self, records: list[dict[str, Any]], state: BaselineState, update_dir: Path) -> UpdateResult:
        if self.project_dir is None:
            raise RuntimeError("TFGRPOBaseline requires config.project_dir")
        experience_path = state.state_dir / self.experiences_filename
        experiences = json.loads(experience_path.read_text(encoding="utf-8")) if experience_path.exists() else {}
        rollouts = [_to_tf_grpo_rollout(record) for record in records]
        diagnostics = _rollout_diagnostics(rollouts)
        update_rollouts = _filter_update_rollouts(rollouts, skip_metadata_fallback=self.skip_metadata_fallback)
        skipped_rollouts = [rollout for rollout in rollouts if rollout not in update_rollouts]
        diagnostics["skipped_metadata_fallback_records"] = sum(
            1 for rollout in skipped_rollouts if (rollout.get("seagym_artifacts") or {}).get("used_metadata_fallback")
        )
        diagnostics["skipped_non_trajectory_records"] = sum(
            1
            for rollout in skipped_rollouts
            if not (rollout.get("seagym_artifacts") or {}).get("used_metadata_fallback")
        )
        if rollouts and not update_rollouts:
            prompt_path = state.state_dir / self.prompt_filename
            prompt_path.write_text(_format_experiences_prompt(experiences), encoding="utf-8")
            state.metadata["prompt_template_path"] = str(prompt_path)
            return UpdateResult(
                update_index=self.update_index,
                changed=False,
                status="unchanged",
                metrics={"num_trajectories": len(records), "num_experiences": len(experiences), **diagnostics},
                logs={"skipped_reason": "no TF-GRPO rollouts with real agent trajectories"},
                artifacts={"experience_path": str(experience_path), "update_dir": str(update_dir), "prompt_path": str(prompt_path)},
            )
        with patched_runtime_env(self.runtime):
            with prepend_sys_path(self.project_dir):
                metered = _TFGRPOMeter()
                _patch_tf_grpo_dotenv_lookup()
                if self.domain == "math":
                    experience_module = importlib.import_module("training_free_grpo.math.experience")
                    _apply_update_prompt_profile(experience_module, self.update_prompt_profile)
                    ExperienceUpdater = _patch_experience_updater_llm(experience_module, metered)

                    new_experiences = ExperienceUpdater().run(
                        rollouts=update_rollouts,
                        experiences=experiences,
                        save_dir=str(update_dir),
                        max_workers=self.max_workers,
                        given_ground_truth=self.given_ground_truth,
                        only_partial_correct=self.only_partial_correct,
                    )
                elif self.domain == "web":
                    experience_module = importlib.import_module("training_free_grpo.web.experience")
                    _apply_update_prompt_profile(experience_module, self.update_prompt_profile)
                    ExperienceUpdater = _patch_experience_updater_llm(experience_module, metered)

                    new_experiences = ExperienceUpdater().run(
                        rollouts=update_rollouts,
                        experiences=experiences,
                        save_dir=str(update_dir),
                        max_workers=self.max_workers,
                        given_ground_truth=self.given_ground_truth,
                    )
                else:
                    raise ValueError(f"Unsupported TF-GRPO domain: {self.domain}")
        changed = _canonical_json(new_experiences) != _canonical_json(experiences)
        experience_path.write_text(json.dumps(new_experiences, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prompt_path = state.state_dir / self.prompt_filename
        prompt_path.write_text(_format_experiences_prompt(new_experiences), encoding="utf-8")
        state.metadata["prompt_template_path"] = str(prompt_path)
        logs: dict[str, Any] = {}
        cost = metered.cost() or extract_token_cost(new_experiences)
        if cost:
            logs["cost"] = cost
            logs["cost_source"] = "tf_grpo_llm_usage" if metered.cost() else "tf_grpo_update_outputs"
        return UpdateResult(
            update_index=self.update_index,
            changed=changed,
            status="updated" if changed else "unchanged",
            metrics={"num_trajectories": len(records), "num_experiences": len(new_experiences), **diagnostics},
            logs=logs,
            artifacts={"experience_path": str(experience_path), "update_dir": str(update_dir)},
        )

    def _update_with_runtime(self, records: list[dict[str, Any]], state: BaselineState, update_dir: Path) -> UpdateResult:
        setup = run_setup_commands(self.runtime, project_dir=self.project_dir, state_dir=state.state_dir)
        if setup.get("ok") is False:
            return native_error_result(
                update_index=self.update_index,
                update_dir=update_dir,
                error="TF-GRPO native runtime setup failed",
                stage="tf_grpo_runtime_setup",
                logs={"setup": setup},
            )
        script_path = (update_dir / "tf_grpo_runtime_update.py").resolve()
        output_path = (update_dir / "runtime_update_result.json").resolve()
        script_path.write_text(_TF_GRPO_RUNTIME_RUNNER, encoding="utf-8")
        args = [
            str((update_dir / "trajectories.json").resolve()),
            str(state.state_dir.resolve()),
            str(update_dir.resolve()),
            "" if self.project_dir is None else str(self.project_dir.resolve()),
            self.domain,
            self.experiences_filename,
            self.prompt_filename,
            str(self.max_workers),
            "1" if self.given_ground_truth else "0",
            "1" if self.only_partial_correct else "0",
            self.update_prompt_profile,
            "1" if self.skip_metadata_fallback else "0",
            str(output_path),
        ]
        completed = run_python_script(self.runtime, script_path=script_path, args=args, cwd=self.project_dir)
        logs = {"setup": setup, "return_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode != 0:
            return native_error_result(
                update_index=self.update_index,
                update_dir=update_dir,
                error="TF-GRPO native runtime update failed",
                stage="tf_grpo_runtime_update",
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
            logs={**logs, **({"cost": result["cost"], "cost_source": result.get("cost_source")} if isinstance(result.get("cost"), dict) else {})},
            artifacts=dict(result.get("artifacts") or {}),
        )

