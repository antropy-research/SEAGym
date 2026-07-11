from __future__ import annotations

"""Config-aware runtime checks for SEAGym runs."""

import json
import os
from pathlib import Path
import tempfile

from seagym.config.fields import as_mapping, expand_env_templates, resolve_path
from seagym.logging import redact_url

from .checks import DEFAULT_CONTAINER_PROXY_ENV_NAMES, DEFAULT_PROXY_ENV_NAMES, CheckResult


def inspect_experiment_config(config_path: str | Path, *, run_dir: str | Path | None = None) -> list[CheckResult]:
    """Return non-blocking reproducibility/runtime warnings for a config."""

    config_file = Path(config_path).resolve()
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except OSError as exc:
        return [CheckResult("config", "fail", f"cannot read {config_file}: {exc}")]
    except json.JSONDecodeError as exc:
        return [CheckResult("config", "fail", f"invalid JSON in {config_file}: {exc}")]

    checks: list[CheckResult] = []
    base_dir = config_file.parent
    output = as_mapping(data.get("output"))
    configured_run_dir = run_dir if run_dir is not None else output.get("run_dir")
    if configured_run_dir:
        resolved_run_dir = resolve_path(base_dir, configured_run_dir)
        recommended_root = _recommended_results_root(config_file)
        tmp_root = Path(tempfile.gettempdir()).resolve()
        if _is_relative_to(resolved_run_dir, recommended_root):
            checks.append(CheckResult("config:run_dir", "ok", str(resolved_run_dir)))
        else:
            detail = f"{resolved_run_dir} is outside recommended root {recommended_root}"
            if _is_relative_to(resolved_run_dir, tmp_root):
                detail += "; /tmp-style paths can make Harbor/Docker artifacts harder to reproduce"
            checks.append(CheckResult("config:run_dir", "warn", detail))
    else:
        checks.append(CheckResult("config:run_dir", "warn", "output.run_dir is not set"))

    backend = as_mapping(data.get("backend"))
    if backend.get("name") == "harbor":
        backend_env = str(backend.get("env", "docker"))
        if backend_env == "docker":
            checks.extend(_inspect_harbor_docker_backend(backend))
        elif backend_env == "e2b":
            checks.extend(inspect_harbor_e2b_backend())
    baseline = as_mapping(data.get("baseline"))
    if baseline:
        checks.extend(_inspect_baseline_lifecycle(baseline))
    rollout_agent = as_mapping(data.get("rollout_agent"))
    checks.extend(_inspect_rollout_agent_lifecycle(rollout_agent))
    return checks


def inspect_harbor_e2b_backend() -> list[CheckResult]:
    checks: list[CheckResult] = []
    try:
        import dirhash  # noqa: F401
        import dockerfile_parse  # noqa: F401
        import e2b  # noqa: F401
    except ImportError as exc:
        checks.append(
            CheckResult(
                "config:harbor.e2b_extra",
                "fail",
                "missing Harbor E2B dependencies; install with: python -m pip install -e 'reference/harbor[e2b]'"
                f" ({exc})",
            )
        )
    else:
        checks.append(CheckResult("config:harbor.e2b_extra", "ok", "installed"))

    checks.append(
        CheckResult(
            "config:harbor.e2b_api_key",
            "ok" if os.environ.get("E2B_API_KEY") else "fail",
            "set" if os.environ.get("E2B_API_KEY") else "E2B_API_KEY is not set",
        )
    )
    return checks


def _inspect_harbor_docker_backend(backend: dict[str, object]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    container_env = as_mapping(backend.get("container_env"))
    verifier_env = {**container_env, **as_mapping(backend.get("verifier_env"))}
    host_has_proxy = any(os.environ.get(name) for name in DEFAULT_PROXY_ENV_NAMES)
    if not verifier_env:
        status = "warn" if host_has_proxy else "ok"
        detail = (
            "host proxy is set but backend.container_env/backend.verifier_env is empty; Terminal-Bench verifiers may fail downloading uv/packages"
            if host_has_proxy
            else "not configured"
        )
        checks.append(CheckResult("config:harbor.container_env", status, detail))
        return checks

    expanded = expand_env_templates(verifier_env)
    proxy_values = {key: value for key, value in expanded.items() if key.upper().endswith("PROXY") and value}
    localhost_values = [
        f"{key}={redact_url(value)}"
        for key, value in proxy_values.items()
        if _looks_like_localhost_proxy(value)
    ]
    if localhost_values:
        checks.append(
            CheckResult(
                "config:harbor.container_env",
                "warn",
                "container proxy points at localhost; Docker containers usually need a host gateway address: "
                + ", ".join(localhost_values),
            )
        )
    else:
        checks.append(
            CheckResult(
                "config:harbor.container_env",
                "ok" if proxy_values else "warn",
                "container proxy configured" if proxy_values else "container_env/verifier_env has no proxy values",
            )
        )

    missing_templates = [
        str(value)[2:-1]
        for value in verifier_env.values()
        if isinstance(value, str)
        and value.startswith("${")
        and value.endswith("}")
        and not os.environ.get(value[2:-1])
        and value[2:-1] in DEFAULT_CONTAINER_PROXY_ENV_NAMES
    ]
    if missing_templates:
        checks.append(
            CheckResult(
                "config:harbor.container_proxy_env",
                "warn",
                "empty env vars: " + ", ".join(sorted(missing_templates)),
            )
        )
    return checks


def _inspect_baseline_lifecycle(baseline: dict[str, object]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    class_path = baseline.get("class_path")
    if class_path:
        checks.append(CheckResult("config:baseline.class_path", "ok", str(class_path)))
        return checks
    checks.append(CheckResult("config:baseline.lifecycle", "fail", "baseline.class_path is required"))
    return checks


def _inspect_rollout_agent_lifecycle(rollout_agent: dict[str, object]) -> list[CheckResult]:
    class_path = rollout_agent.get("class_path")
    if class_path:
        return [CheckResult("config:rollout_agent.class_path", "ok", str(class_path))]
    return [CheckResult("config:rollout_agent.lifecycle", "fail", "rollout_agent.class_path is required")]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _recommended_results_root(config_file: Path) -> Path:
    for candidate in [config_file.parent, *config_file.parents, Path.cwd().resolve()]:
        if (candidate / "pyproject.toml").exists() and (candidate / "seagym").is_dir():
            return (candidate / "results" / "runs").resolve()
    return (Path.cwd().resolve() / "results" / "runs").resolve()


def _looks_like_localhost_proxy(value: str) -> bool:
    lowered = value.lower()
    return "://127.0.0.1" in lowered or "://localhost" in lowered or lowered.startswith("127.0.0.1:")
