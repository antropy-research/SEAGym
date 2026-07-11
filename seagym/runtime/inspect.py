from __future__ import annotations

"""Inspect runtime resources before a SEAGym run."""

from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
import shutil
from typing import Any

from seagym.config import load_experiment_context
from seagym.config.fields import expand_env_templates
from seagym.data import SEAGymDataModule, TaskRecord
from seagym.envs import TaskRunResult
from seagym.envs.harbor_env import HarborEnv
from seagym.logging import redact_sensitive
from seagym.trainers.builder import DEFAULT_E2B_ENVIRONMENT_IMPORT_PATH
from seagym.utils import write_json

from .checks import CheckResult, run_container_network_checks, run_runtime_checks
from .env import load_env_file


DEFAULT_HOST_PROBE_URLS = [
    "https://github.com",
    "https://astral.sh",
    "https://nodejs.org",
    "https://registry.npmjs.org",
]


@dataclass(frozen=True)
class RuntimeCheckOptions:
    config_path: Path
    run_dir: Path | None = None
    env_file: Path = Path(".env")
    load_env: bool = True
    host_probe_urls: list[str] = field(default_factory=list)
    container_probe_urls: list[str] = field(default_factory=list)
    container_probe_image: str = "python:3.12-alpine"
    canary: bool = False
    canary_task_limit: int = 1
    canary_agent: str = "oracle"
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class RuntimeCheckReport:
    manifest_path: Path
    ok: bool
    checks: list[CheckResult]
    task_ids: list[str]
    canary_results: list[TaskRunResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "manifest_path": str(self.manifest_path),
            "checks": [check.to_dict() for check in self.checks],
            "task_ids": self.task_ids,
            "canary_results": [_task_run_result_dict(result) for result in self.canary_results],
        }


def inspect_runtime(options: RuntimeCheckOptions) -> RuntimeCheckReport:
    if options.load_env:
        load_env_file(options.env_file)

    context = load_experiment_context(options.config_path)
    run_dir = (options.run_dir or context.config.run_dir).resolve()
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "runtime_check.json"

    batch_plan = SEAGymDataModule(context).build()
    task_ids = _materialized_task_ids(batch_plan.to_dict())
    tasks = [context.task_index.require(task_id) for task_id in task_ids]

    backend = context.config.raw.get("backend") if isinstance(context.config.raw, dict) else {}
    backend = backend if isinstance(backend, dict) else {}
    container_env = backend.get("container_env") if isinstance(backend.get("container_env"), dict) else {}
    verifier_env_raw = backend.get("verifier_env") if isinstance(backend.get("verifier_env"), dict) else {}
    verifier_env = expand_env_templates({**container_env, **verifier_env_raw})

    checks: list[CheckResult] = []
    is_harbor_backend = backend.get("name") == "harbor"
    runtime_report = run_runtime_checks(
        env_file=options.env_file,
        load_env=False,
        config_path=options.config_path,
        run_dir=run_dir,
        check_harbor=is_harbor_backend,
        check_docker=is_harbor_backend and str(backend.get("env", "docker")) == "docker",
        harbor_bin=str(backend.get("harbor_bin", "harbor")),
        probe_urls=options.host_probe_urls,
        timeout_seconds=options.timeout_seconds,
    )
    checks.extend(runtime_report.checks)
    checks.extend(inspect_method_runtime(context.config.raw))
    checks.extend(inspect_task_resources(tasks))
    checks.extend(
        run_container_network_checks(
            options.container_probe_urls,
            image=options.container_probe_image,
            env=verifier_env,
            timeout_seconds=options.timeout_seconds,
        )
    )

    canary_results: list[TaskRunResult] = []
    if options.canary:
        canary_results = run_harbor_canary(
            tasks[: max(options.canary_task_limit, 0)],
            run_dir=runtime_dir,
            backend=backend,
            verifier_env=verifier_env,
            agent_id=options.canary_agent,
        )
        for result in canary_results:
            status = "ok" if result.success else "fail"
            detail = f"{result.task_id} score={result.score}"
            if result.error:
                detail += f" error={result.error[:240]}"
            checks.append(CheckResult("harbor_canary", status, detail))

    ok = all(check.status != "fail" for check in checks)
    manifest = {
        "ok": ok,
        "config_path": str(options.config_path),
        "run_dir": str(run_dir),
        "batch_plan": batch_plan.to_dict(),
        "materialized_tasks": [_task_manifest_entry(task) for task in tasks],
        "checks": [check.to_dict() for check in checks],
        "canary": {
            "enabled": options.canary,
            "agent": options.canary_agent,
            "task_limit": options.canary_task_limit,
            "results": [_task_run_result_dict(result) for result in canary_results],
        },
    }
    write_json(manifest_path, manifest)
    return RuntimeCheckReport(
        manifest_path=manifest_path,
        ok=ok,
        checks=checks,
        task_ids=task_ids,
        canary_results=canary_results,
    )


def inspect_task_resources(tasks: list[TaskRecord]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    seen_paths: set[str] = set()
    for task in tasks:
        dataset_path = task.source.get("dataset_path")
        local_path = task.source.get("local_path")
        path_value = dataset_path or local_path
        if not path_value:
            checks.append(CheckResult(f"task:{task.task_id}", "warn", "no local Harbor path recorded"))
            continue
        path = Path(str(path_value)).resolve()
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        checks.append(
            CheckResult(
                f"task_path:{path.name}",
                "ok" if path.exists() else "fail",
                str(path),
            )
        )
    return checks


def inspect_method_runtime(config: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    baseline = config.get("baseline") if isinstance(config, dict) else {}
    baseline = baseline if isinstance(baseline, dict) else {}
    baseline_config = baseline.get("config") if isinstance(baseline.get("config"), dict) else {}
    native_adapter = baseline_config.get("native_adapter") if isinstance(baseline_config.get("native_adapter"), dict) else {}
    if baseline.get("class_path") == "seagym.baselines.gepa:GEPABaseline" and native_adapter.get("type") == "terminal_bench":
        checks.append(
            CheckResult(
                "config:gepa.terminal_bench_package",
                "ok" if importlib.util.find_spec("terminal_bench") else "fail",
                "installed" if importlib.util.find_spec("terminal_bench") else "missing Python package terminal_bench",
            )
        )
        checks.append(
            CheckResult(
                "config:gepa.tb_cli",
                "ok" if shutil.which("tb") else "fail",
                shutil.which("tb") or "missing tb CLI",
            )
        )
    return checks


def run_harbor_canary(
    tasks: list[TaskRecord],
    *,
    run_dir: Path,
    backend: dict[str, Any],
    verifier_env: dict[str, str],
    agent_id: str,
) -> list[TaskRunResult]:
    if not tasks:
        return []
    backend_env = str(backend.get("env", "docker"))
    environment_import_path = backend.get("environment_import_path")
    if environment_import_path in (None, "") and backend_env == "e2b":
        environment_import_path = DEFAULT_E2B_ENVIRONMENT_IMPORT_PATH
    env = HarborEnv(
        run_dir / "harbor_canary_jobs",
        harbor_bin=str(backend.get("harbor_bin", "harbor")),
        n_concurrent=1,
        env=backend_env,
        environment_import_path=None if environment_import_path in (None, "") else str(environment_import_path),
        environment_kwargs=_dict_value(backend.get("environment_kwargs")),
        yes=bool(backend.get("yes", True)),
        verifier_env=verifier_env,
        extra_args=[str(item) for item in (backend.get("extra_args") or [])],
        agent_override_timeout_sec=(
            None
            if backend.get("agent_override_timeout_sec") in (None, "")
            else int(backend["agent_override_timeout_sec"])
        ),
        verifier_override_timeout_sec=(
            None
            if backend.get("verifier_override_timeout_sec") in (None, "")
            else int(backend["verifier_override_timeout_sec"])
        ),
    )
    return env.run_tasks(tasks, view_name="runtime_canary", mode="runtime_check", agent_id=agent_id)


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _materialized_task_ids(batch_plan: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            if value not in ids:
                ids.append(value)
            return
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                add(item)

    add(batch_plan.get("train_batches"))
    add(batch_plan.get("views"))
    return ids


def _task_manifest_entry(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "source": {
            key: task.source.get(key)
            for key in ("type", "dataset", "dataset_path", "local_path", "task_name", "registry_task_name")
            if key in task.source
        },
        "attributes": task.attributes,
        "scoring": task.scoring.to_dict(),
    }


def _task_run_result_dict(result: TaskRunResult) -> dict[str, Any]:
    return redact_sensitive(asdict(result))
