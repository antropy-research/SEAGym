from __future__ import annotations

"""Low-level runtime dependency and connectivity checks."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Mapping

from seagym.logging import redact_url

from .env import DEFAULT_ENV_FILE, load_env_file


DEFAULT_KEY_ENV_NAMES = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")
DEFAULT_PROXY_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
DEFAULT_CONTAINER_PROXY_ENV_NAMES = (
    "SEAGYM_CONTAINER_HTTP_PROXY",
    "SEAGYM_CONTAINER_HTTPS_PROXY",
    "SEAGYM_CONTAINER_ALL_PROXY",
    "SEAGYM_CONTAINER_NO_PROXY",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class RuntimeCheckSummary:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


def run_runtime_checks(
    *,
    env_file: str | Path = DEFAULT_ENV_FILE,
    load_env: bool = True,
    config_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    key_env_names: list[str] | None = None,
    proxy_env_names: list[str] | None = None,
    harbor_bin: str | None = None,
    check_harbor: bool = True,
    check_docker: bool = False,
    check_e2b: bool = False,
    probe_urls: list[str] | None = None,
    check_container_network: bool = False,
    container_probe_urls: list[str] | None = None,
    container_probe_image: str = "python:3.12-alpine",
    container_env: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
) -> RuntimeCheckSummary:
    report = RuntimeCheckSummary()
    if load_env:
        try:
            loaded = load_env_file(env_file)
            if Path(env_file).exists():
                report.add("env_file", "ok", f"loaded {len(loaded)} values from {env_file}")
            else:
                report.add("env_file", "warn", f"{env_file} not found; using current shell environment")
        except ValueError as exc:
            report.add("env_file", "fail", str(exc))

    effective_key_env_names = list(DEFAULT_KEY_ENV_NAMES) if key_env_names is None else key_env_names
    effective_proxy_env_names = list(DEFAULT_PROXY_ENV_NAMES) if proxy_env_names is None else proxy_env_names

    for name in effective_key_env_names:
        value = os.environ.get(name)
        report.add(f"env:{name}", "ok" if value else "warn", "set" if value else "not set")

    for name in effective_proxy_env_names:
        value = os.environ.get(name)
        report.add(f"proxy:{name}", "ok" if value else "warn", redact_url(value) if value else "not set")

    if check_harbor:
        bin_name = harbor_bin or os.environ.get("SEAGYM_HARBOR_BIN", "harbor")
        report.add("harbor_bin", "ok" if shutil.which(bin_name) else "fail", bin_name)
        if shutil.which(bin_name):
            _check_command(report, "harbor_help", [bin_name, "--help"], timeout_seconds=timeout_seconds)

    if check_docker:
        report.add("docker_bin", "ok" if shutil.which("docker") else "fail", "docker")
        if shutil.which("docker"):
            _check_command(report, "docker_info", ["docker", "info"], timeout_seconds=timeout_seconds)

    if check_e2b:
        from .config import inspect_harbor_e2b_backend

        report.checks.extend(inspect_harbor_e2b_backend())

    for url in probe_urls or []:
        _probe_url(report, url, timeout_seconds=timeout_seconds)
    if check_container_network:
        urls = container_probe_urls if container_probe_urls is not None else probe_urls
        for check in run_container_network_checks(
            urls or [],
            image=container_probe_image,
            env=container_env,
            timeout_seconds=timeout_seconds,
        ):
            report.checks.append(check)
    if config_path is not None:
        from .config import inspect_experiment_config

        for check in inspect_experiment_config(config_path, run_dir=run_dir):
            report.checks.append(check)
    return report


def run_container_network_checks(
    urls: list[str],
    *,
    image: str = "python:3.12-alpine",
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
) -> list[CheckResult]:
    """Probe URL reachability from inside a Docker container."""

    if not urls:
        return []
    if not shutil.which("docker"):
        return [CheckResult("container_network:docker", "fail", "docker not found")]

    checks: list[CheckResult] = []
    effective_env = {str(key): str(value) for key, value in (env or {}).items() if value}
    for url in urls:
        command = [
            "docker",
            "run",
            "--rm",
        ]
        for key, value in effective_env.items():
            command.extend(["-e", f"{key}={value}"])
        command.extend(
            [
                image,
                "python",
                "-c",
                (
                    "import sys, urllib.error, urllib.request; "
                    "url = sys.argv[1]; "
                    f"req = urllib.request.Request(url, method='HEAD'); timeout = {timeout_seconds!r}; "
                    "\ntry:\n"
                    "    r = urllib.request.urlopen(req, timeout=timeout)\n"
                    "    print(f'HTTP {r.status}')\n"
                    "except urllib.error.HTTPError as exc:\n"
                    "    print(f'HTTP {exc.code}')\n"
                    "except Exception as exc:\n"
                    "    print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)\n"
                    "    sys.exit(1)\n"
                ),
                url,
            ]
        )
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            checks.append(CheckResult(f"container_url:{url}", "fail", f"timed out after {timeout_seconds:g}s using {image}"))
            continue
        except OSError as exc:
            checks.append(CheckResult(f"container_url:{url}", "fail", str(exc)))
            continue
        detail = (completed.stdout or completed.stderr).strip().splitlines()
        checks.append(
            CheckResult(
                f"container_url:{url}",
                "ok" if completed.returncode == 0 else "fail",
                (detail[0] if detail else "") + f" using {image}",
            )
        )
    return checks


def _check_command(report: RuntimeCheckSummary, name: str, command: list[str], *, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        report.add(name, "fail", f"timed out after {timeout_seconds:g}s")
        return
    except OSError as exc:
        report.add(name, "fail", str(exc))
        return
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    report.add(name, "ok" if completed.returncode == 0 else "fail", detail[0] if detail else "")


def _probe_url(report: RuntimeCheckSummary, url: str, *, timeout_seconds: float) -> None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            report.add(f"url:{url}", "ok", f"HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        # For connectivity runtime check, any HTTP response from the remote service
        # proves DNS/proxy/TLS routing worked. Some artifact hosts reject HEAD.
        report.add(f"url:{url}", "ok", f"HTTP {exc.code}")
    except Exception as exc:
        report.add(f"url:{url}", "fail", str(exc))
