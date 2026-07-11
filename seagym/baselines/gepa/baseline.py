from __future__ import annotations

"""GEPA baseline adapter."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from seagym.costs import extract_token_cost

from ..adapter_utils import load_import_path, optional_int, prepend_sys_path, resolve_optional_path
from ..base import BaseBaseline, BaselineState, Checkpoint, UpdateResult
from ..data import TrajectoryBatch
from ..model_mapping import UpdateModelBinding
from .adapters import (
    _SEAGymHarborGEPAAdapter,
    _SEAGymTerminusAdapter,
    _materialize_terminal_bench_adapter_workspace,
    _pushd,
    _task_id_from_record,
    _terminal_bench_adapter_kwargs,
    _terminal_bench_task_name,
)
from .candidate import _best_score, _gepa_reflection_lm_cost, _render_candidate
from .config import _redacted_lm_kwargs, _redacted_native_adapter, _resolve_native_adapter_paths
from .state import _write_state_metadata


@dataclass
class _GEPARuntime:
    env: Any
    task_index: Any
    rollout_agent: Any
    run_dir: Path
    batch_plan: Any | None = None


@dataclass
class GEPABaseline(BaseBaseline):
    project_dir: Path | None = None
    max_update_records: int | None = None
    seed_candidate: str = ""
    candidate_filename: str = "candidate.txt"
    objective: str = "Improve the rollout policy using SEAGym train trajectories."
    evaluator_import_path: str | None = None
    native_adapter: dict[str, Any] = field(default_factory=dict)
    candidate_component: str = "instruction_prompt"
    reflection_lm: str | None = None
    reflection_lm_kwargs: dict[str, Any] | None = None
    reflection_lm_api_key_env: str | None = None
    max_metric_calls: int = 16
    _runtime: _GEPARuntime | None = field(default=None, init=False, repr=False)

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
    ) -> "GEPABaseline":
        del run_dir
        update_model = UpdateModelBinding.from_config(
            config,
            models,
            default_model="openai/gpt-5.1",
            config_model_keys=("reflection_lm", "model"),
        )
        native_adapter = dict(config.get("native_adapter") or {})
        return cls(
            baseline_id=name,
            state_dir=state_dir,
            project_dir=resolve_optional_path(config.get("project_dir"), base_dir=base_dir),
            max_update_records=optional_int(config.get("max_update_records")),
            seed_candidate=str(config.get("seed_candidate", "")),
            candidate_filename=str(config.get("candidate_filename", "candidate.txt")),
            objective=str(config.get("objective", "Improve the rollout policy using SEAGym train trajectories.")),
            evaluator_import_path=None
            if config.get("evaluator_import_path") in (None, "")
            else str(config.get("evaluator_import_path")),
            native_adapter=_resolve_native_adapter_paths(native_adapter, base_dir=base_dir),
            candidate_component=str(config.get("candidate_component", native_adapter.get("candidate_component", "instruction_prompt"))),
            reflection_lm=update_model.litellm_model(),
            reflection_lm_kwargs=update_model.litellm_kwargs(),
            reflection_lm_api_key_env=update_model.api_key_env,
            max_metric_calls=int(config.get("max_metric_calls", 16)),
        )

    def initialize(self, run_dir: Path) -> BaselineState:
        state = super().initialize(run_dir)
        candidate_path = self.state_dir / self.candidate_filename
        if not candidate_path.exists():
            candidate_path.write_text(self.seed_candidate, encoding="utf-8")
        state.metadata.update(
            {
                "native_method": "GEPA",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "candidate_path": str(candidate_path),
                "prompt_template_path": str(candidate_path),
                "evaluator_import_path": self.evaluator_import_path,
                "native_adapter": _redacted_native_adapter(self.native_adapter),
                "candidate_component": self.candidate_component,
                "reflection_lm": self.reflection_lm,
                "reflection_lm_api_key_env": self.reflection_lm_api_key_env,
                "reflection_lm_kwargs": _redacted_lm_kwargs(self.reflection_lm_kwargs),
            }
        )
        return state

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        state = super().load_checkpoint(checkpoint)
        candidate_path = self.state_dir / self.candidate_filename
        state.metadata.update(
            {
                "baseline_id": self.baseline_id,
                "native_method": "GEPA",
                "project_dir": None if self.project_dir is None else str(self.project_dir),
                "max_update_records": self.max_update_records,
                "candidate_path": str(candidate_path),
                "prompt_template_path": str(candidate_path),
                "evaluator_import_path": self.evaluator_import_path,
                "native_adapter": _redacted_native_adapter(self.native_adapter),
                "candidate_component": self.candidate_component,
                "reflection_lm": self.reflection_lm,
                "reflection_lm_api_key_env": self.reflection_lm_api_key_env,
                "reflection_lm_kwargs": _redacted_lm_kwargs(self.reflection_lm_kwargs),
                "loaded": True,
            }
        )
        _write_state_metadata(self.state_dir, state.metadata)
        return state

    def bind_runtime(
        self,
        *,
        env: Any,
        task_index: Any,
        rollout_agent: Any,
        run_dir: Path,
        batch_plan: Any | None = None,
    ) -> None:
        self._runtime = _GEPARuntime(
            env=env,
            task_index=task_index,
            rollout_agent=rollout_agent,
            run_dir=run_dir,
            batch_plan=batch_plan,
        )

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        if not self.native_adapter and self.evaluator_import_path is None:
            raise RuntimeError(
                "GEPABaseline requires baseline.config.evaluator_import_path or baseline.config.native_adapter. "
                "GEPA must re-evaluate candidate artifacts through a real evaluator; SEAGym will not reuse "
                "observed rollout scores as fake optimization feedback."
            )
        if self.project_dir is None:
            raise RuntimeError("GEPABaseline requires config.project_dir")
        update_dir = self.next_update_dir(state, "gepa")
        records = self.write_trajectories(trajectories, update_dir, max_records=self.max_update_records)
        candidate_path = state.state_dir / self.candidate_filename
        seed_candidate = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else self.seed_candidate
        if self.native_adapter:
            return self._update_with_native_adapter(records, state, update_dir, candidate_path, seed_candidate)
        assert self.evaluator_import_path is not None
        user_evaluator = load_import_path(self.evaluator_import_path)
        with prepend_sys_path(self.project_dir / "src"):
            from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything  # type: ignore

            def evaluator(candidate: str, example: dict[str, Any] | None = None, opt_state: Any = None) -> Any:
                return user_evaluator(
                    candidate=candidate,
                    example=example,
                    trajectories=records,
                    state_dir=str(state.state_dir),
                    update_dir=str(update_dir),
                    opt_state=opt_state,
                )

            gepa_config = GEPAConfig(
                engine=EngineConfig(run_dir=str(update_dir / "gepa"), max_metric_calls=self.max_metric_calls),
                reflection=ReflectionConfig(
                    reflection_lm=self.reflection_lm,
                    reflection_lm_kwargs=self._resolved_reflection_lm_kwargs(),
                ),
            )
            result = optimize_anything(
                seed_candidate=seed_candidate,
                evaluator=evaluator,
                dataset=records,
                objective=self.objective,
                config=gepa_config,
            )
            reflection_cost = _gepa_reflection_lm_cost(gepa_config.reflection.reflection_lm)
        best_candidate = result.best_candidate
        rendered_candidate = _render_candidate(best_candidate)
        return self._finish_update(
            result=result,
            reflection_cost=reflection_cost,
            candidate_path=candidate_path,
            update_dir=update_dir,
            rendered_candidate=rendered_candidate,
            seed_candidate=seed_candidate,
            num_records=len(records),
        )

    def _update_with_native_adapter(
        self,
        records: list[dict[str, Any]],
        state: BaselineState,
        update_dir: Path,
        candidate_path: Path,
        seed_candidate: str,
    ) -> UpdateResult:
        del state
        adapter_type = str(self.native_adapter.get("type", ""))
        if adapter_type == "terminal_bench":
            with prepend_sys_path(self.project_dir / "src"):
                result, reflection_cost = self._run_terminal_bench_adapter(records, update_dir)
        elif adapter_type == "harbor":
            with prepend_sys_path(self.project_dir / "src"):
                result, reflection_cost = self._run_harbor_adapter(records, update_dir)
        else:
            raise ValueError(f"Unsupported GEPA native_adapter.type: {adapter_type!r}")
        rendered_candidate = _render_candidate(result.best_candidate, component_key=self.candidate_component)
        return self._finish_update(
            result=result,
            reflection_cost=reflection_cost,
            candidate_path=candidate_path,
            update_dir=update_dir,
            rendered_candidate=rendered_candidate,
            seed_candidate=seed_candidate,
            num_records=len(records),
        )

    def _run_terminal_bench_adapter(self, records: list[dict[str, Any]], update_dir: Path) -> tuple[Any, dict[str, float]]:
        from gepa import EvaluationBatch, optimize  # type: ignore
        from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (  # type: ignore
            TerminalBenchTask,
            TerminusAdapter,
            get_results,
            run_agent_tb,
        )

        if self.native_adapter.get("model_name") in (None, ""):
            raise ValueError("GEPA terminal_bench native_adapter requires model_name")
        trainset = [
            TerminalBenchTask(
                task_id=_terminal_bench_task_name(record),
                model_name=str(self.native_adapter["model_name"]),
            )
            for record in records
        ]
        if not trainset:
            raise ValueError("GEPA terminal_bench native_adapter requires at least one Terminal-Bench trajectory")
        prompt_path = self.native_adapter.get("instruction_prompt_path")
        if prompt_path in (None, ""):
            prompt_path = _materialize_terminal_bench_adapter_workspace(self.project_dir, update_dir)
        n_concurrent = int(self.native_adapter.get("n_concurrent", 1))
        adapter_kwargs = _terminal_bench_adapter_kwargs(self.native_adapter)
        if adapter_kwargs:
            adapter = _SEAGymTerminusAdapter(
                EvaluationBatch=EvaluationBatch,
                TerminusAdapter=TerminusAdapter,
                get_results=get_results,
                run_agent_tb=run_agent_tb,
                n_concurrent=n_concurrent,
                instruction_prompt_path=str(prompt_path),
                **adapter_kwargs,
            )
        else:
            adapter = TerminusAdapter(
                n_concurrent=n_concurrent,
                instruction_prompt_path=str(prompt_path),
            )
        seed_candidate = {
            self.candidate_component: (self.state_dir / self.candidate_filename).read_text(encoding="utf-8")
            if (self.state_dir / self.candidate_filename).exists()
            else self.seed_candidate
        }
        adapter_cwd = Path(prompt_path).parent.parent
        with _pushd(adapter_cwd):
            result = optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=None,
                adapter=adapter,
                reflection_lm=self.reflection_lm,
                reflection_lm_kwargs=self._resolved_reflection_lm_kwargs(),
                max_metric_calls=self.max_metric_calls,
                reflection_minibatch_size=(
                    None
                    if self.native_adapter.get("reflection_minibatch_size") in (None, "")
                    else int(self.native_adapter["reflection_minibatch_size"])
                ),
                run_dir=str(update_dir / "gepa"),
                use_wandb=bool(self.native_adapter.get("use_wandb", False)),
                cache_evaluation=bool(self.native_adapter.get("cache_evaluation", False)),
                display_progress_bar=bool(self.native_adapter.get("display_progress_bar", False)),
            )
        return result, {}

    def _run_harbor_adapter(self, records: list[dict[str, Any]], update_dir: Path) -> tuple[Any, dict[str, float]]:
        from gepa import optimize  # type: ignore

        if self._runtime is None:
            raise RuntimeError("GEPA harbor native_adapter requires ExecutionEngine runtime binding")
        trainset = [_task_id_from_record(record) for record in records]
        if not trainset:
            raise ValueError("GEPA harbor native_adapter requires at least one trajectory")
        adapter = _SEAGymHarborGEPAAdapter(
            runtime=self._runtime,
            baseline_id=self.baseline_id,
            update_dir=update_dir,
            candidate_component=self.candidate_component,
            candidate_filename=self.candidate_filename,
            candidate_view_name=str(self.native_adapter.get("view_name", "gepa_candidate")),
            candidate_mode=str(self.native_adapter.get("mode", "candidate_eval")),
            max_reflective_records=(
                None
                if self.native_adapter.get("max_reflective_records") in (None, "")
                else int(self.native_adapter["max_reflective_records"])
            ),
        )
        seed_candidate = {
            self.candidate_component: (self.state_dir / self.candidate_filename).read_text(encoding="utf-8")
            if (self.state_dir / self.candidate_filename).exists()
            else self.seed_candidate
        }
        valset = self._harbor_adapter_valset()
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=self.reflection_lm,
            reflection_lm_kwargs=self._resolved_reflection_lm_kwargs(),
            max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=(
                None
                if self.native_adapter.get("reflection_minibatch_size") in (None, "")
                else int(self.native_adapter["reflection_minibatch_size"])
            ),
            run_dir=str(update_dir / "gepa"),
            use_wandb=bool(self.native_adapter.get("use_wandb", False)),
            cache_evaluation=bool(self.native_adapter.get("cache_evaluation", False)),
            display_progress_bar=bool(self.native_adapter.get("display_progress_bar", False)),
        )
        return result, {}

    def _harbor_adapter_valset(self) -> list[str] | None:
        valset_view = self.native_adapter.get("valset_view")
        if valset_view in (None, ""):
            return None
        if self._runtime is None or self._runtime.batch_plan is None:
            raise RuntimeError("GEPA harbor native_adapter valset_view requires ExecutionEngine batch_plan binding")
        view = self._runtime.batch_plan.views.get(str(valset_view))
        if not isinstance(view, list):
            raise ValueError(f"GEPA harbor native_adapter valset_view must reference a list view: {valset_view!r}")
        valset = [str(task_id) for task_id in view]
        if not valset:
            raise ValueError(f"GEPA harbor native_adapter valset_view is empty: {valset_view!r}")
        return valset

    def _finish_update(
        self,
        *,
        result: Any,
        reflection_cost: dict[str, float],
        candidate_path: Path,
        update_dir: Path,
        rendered_candidate: str,
        seed_candidate: str,
        num_records: int,
    ) -> UpdateResult:
        candidate_path.write_text(rendered_candidate, encoding="utf-8")
        (update_dir / "gepa_result.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changed = rendered_candidate != seed_candidate
        logs: dict[str, Any] = {}
        cost = reflection_cost or extract_token_cost(result.to_dict())
        if cost:
            logs["cost"] = cost
            logs["cost_source"] = "gepa_reflection_lm"
        return UpdateResult(
            update_index=self.update_index,
            changed=changed,
            status="updated" if changed else "unchanged",
            metrics={
                "num_trajectories": num_records,
                "num_candidates": getattr(result, "num_candidates", None),
                "best_idx": getattr(result, "best_idx", None),
                "best_score": _best_score(result),
            },
            logs=logs,
            artifacts={"candidate_path": str(candidate_path), "update_dir": str(update_dir)},
        )

    def _resolved_reflection_lm_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.reflection_lm_kwargs or {})
        if self.reflection_lm_api_key_env and "api_key" not in kwargs:
            api_key = os.environ.get(self.reflection_lm_api_key_env)
            if api_key:
                kwargs["api_key"] = api_key
        return kwargs
