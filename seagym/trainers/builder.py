from __future__ import annotations

"""Build trainers from experiment config and runtime overrides."""

from dataclasses import dataclass
from pathlib import Path

from seagym.baselines import Baseline, build_baseline
from seagym.config import load_experiment_context
from seagym.config.fields import as_dict, as_list, as_str_dict, config_get, config_section
from seagym.envs import Env
from seagym.envs.harbor_env import HarborEnv
from seagym.rollout_agents import RolloutAgent, build_rollout_agent
from seagym.runtime import load_env_file


DEFAULT_E2B_ENVIRONMENT_IMPORT_PATH = "seagym.envs.harbor_env.e2b_runtime:E2BOneHourEnvironment"


@dataclass(frozen=True)
class TrainerOverrides:
    run_dir: Path | None = None


@dataclass(frozen=True)
class TrainerComponents:
    agent_id: str
    baseline: Baseline
    rollout_agent: RolloutAgent
    env: Env | None
    run_dir: Path | None
    rollout_model: str | None = None


def build_trainer_components(config_path: str | Path, overrides: TrainerOverrides | None = None) -> TrainerComponents:
    overrides = overrides or TrainerOverrides()
    load_env_file()
    context = load_experiment_context(config_path)
    config = context.config.raw
    effective_run_dir = (overrides.run_dir or context.config.run_dir).resolve()
    backend_name = config_get(config, "backend", "name", default="deterministic")
    baseline = build_baseline(
        config,
        run_dir=effective_run_dir,
        base_dir=Path(config_path).resolve().parent,
    )
    rollout_agent = build_rollout_agent(
        config,
        run_dir=effective_run_dir,
        base_dir=Path(config_path).resolve().parent,
    )
    env = build_env(
        config,
        backend_name=backend_name,
        run_dir=effective_run_dir,
        rollout_agent=rollout_agent.rollout_agent,
        rollout_model=rollout_agent.rollout_model,
    )
    return TrainerComponents(
        agent_id=rollout_agent.agent_id,
        baseline=baseline.baseline,
        rollout_agent=rollout_agent.rollout_agent,
        env=env,
        run_dir=overrides.run_dir,
        rollout_model=rollout_agent.rollout_model,
    )


def build_env(
    config: dict,
    *,
    backend_name: str,
    run_dir: Path,
    rollout_agent: RolloutAgent,
    rollout_model: str | None,
) -> Env | None:
    run_dir = run_dir.resolve()
    if backend_name == "deterministic":
        return None
    if backend_name != "harbor":
        raise ValueError(f"Unsupported backend: {backend_name}")
    backend_config = config_section(config, "backend")
    scheduling_config = config_section(config, "runtime_scheduling")
    backend_env = str(backend_config.get("env", "docker"))
    environment_import_path = (
        None
        if backend_config.get("environment_import_path") in (None, "")
        else str(backend_config.get("environment_import_path"))
    )
    if backend_env == "e2b" and environment_import_path is None:
        environment_import_path = DEFAULT_E2B_ENVIRONMENT_IMPORT_PATH
    return HarborEnv(
        run_dir / "harbor" / "jobs",
        harbor_bin=str(backend_config.get("harbor_bin", "harbor")),
        n_concurrent=int(backend_config.get("n_concurrent", 1)),
        env=backend_env,
        environment_import_path=environment_import_path,
        environment_kwargs=as_dict(backend_config.get("environment_kwargs")),
        yes=bool(backend_config.get("yes", True)),
        agent_spec=None,
        model_name=rollout_model,
        agent_env={
            **as_str_dict(backend_config.get("container_env")),
            **as_str_dict(backend_config.get("agent_env")),
        },
        verifier_env={
            **as_str_dict(backend_config.get("container_env")),
            **as_str_dict(backend_config.get("verifier_env")),
        },
        extra_args=as_list(backend_config.get("extra_args")),
        agent_override_timeout_sec=(
            None
            if backend_config.get("agent_override_timeout_sec") in (None, "")
            else int(backend_config["agent_override_timeout_sec"])
        ),
        verifier_override_timeout_sec=(
            None
            if backend_config.get("verifier_override_timeout_sec") in (None, "")
            else int(backend_config["verifier_override_timeout_sec"])
        ),
        preserve_task_order=bool(scheduling_config.get("enabled", False)),
    )


def trainer_overrides_from_values(
    *,
    run_dir: str | Path | None = None,
) -> TrainerOverrides:
    return TrainerOverrides(
        run_dir=None if run_dir is None else Path(run_dir).resolve(),
    )
