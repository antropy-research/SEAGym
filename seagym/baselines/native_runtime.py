from __future__ import annotations

"""Method-local native runtime helpers for external baseline adapters."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback
from typing import Any, Callable

from seagym.paths import has_portable_anchor, resolve_portable_path

from .base import UpdateResult


@dataclass(frozen=True)
class NativeRuntimeConfig:
    setup_commands: list[str] = field(default_factory=list)
    run_command: str | None = None
    python_bin: str | None = None
    setup_timeout_sec: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.setup_commands or self.run_command or self.python_bin)

    @classmethod
    def from_config(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> "NativeRuntimeConfig":
        setup = data.get("setup_commands") or []
        if isinstance(setup, str):
            setup_commands = [setup]
        else:
            setup_commands = [str(item) for item in setup]
        env = data.get("runtime_env") or data.get("env") or {}
        return cls(
            setup_commands=setup_commands,
            run_command=None if data.get("run_command") in (None, "") else str(data.get("run_command")),
            python_bin=_resolve_python_bin(data.get("python_bin"), base_dir=base_dir),
            setup_timeout_sec=None
            if data.get("setup_timeout_sec") in (None, "")
            else int(data.get("setup_timeout_sec")),
            env={str(key): str(value) for key, value in dict(env).items()},
        )


def _resolve_python_bin(value: Any, *, base_dir: Path | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    path = Path(text)
    if has_portable_anchor(text):
        return str(resolve_portable_path(text, base_dir=base_dir))
    if path.is_absolute():
        return str(path.resolve())
    return text


def run_setup_commands(
    runtime: NativeRuntimeConfig,
    *,
    project_dir: Path | None,
    state_dir: Path,
) -> dict[str, Any]:
    if not runtime.setup_commands:
        return {"ran": False, "commands": []}
    marker = state_dir / ".native_runtime_setup.json"
    if marker.exists():
        return {"ran": False, "cached": True, "marker": str(marker)}
    logs: list[dict[str, Any]] = []
    env = _runtime_env(runtime)
    cwd = project_dir if project_dir is not None else Path.cwd()
    for command in runtime.setup_commands:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            capture_output=True,
            text=True,
            timeout=runtime.setup_timeout_sec,
        )
        entry = {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        logs.append(entry)
        if completed.returncode != 0:
            return {"ran": True, "ok": False, "commands": logs}
    marker.write_text(json.dumps({"ok": True, "commands": logs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ran": True, "ok": True, "marker": str(marker), "commands": logs}


def run_python_script(
    runtime: NativeRuntimeConfig,
    *,
    script_path: Path,
    args: list[str],
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    if runtime.run_command:
        command = runtime.run_command
    else:
        python_bin = runtime.python_bin or os.environ.get("PYTHON", "python")
        command = " ".join([shlex.quote(python_bin), shlex.quote(str(script_path)), *map(shlex.quote, args)])
    env = {
        **_runtime_env(runtime),
        "SEAGYM_NATIVE_SCRIPT": str(script_path),
        "SEAGYM_NATIVE_ARGS": json.dumps(args),
    }
    return subprocess.run(command, cwd=cwd, env=env, shell=True, capture_output=True, text=True)


def native_error_result(
    *,
    update_index: int,
    update_dir: Path,
    error: BaseException | str,
    stage: str,
    logs: dict[str, Any] | None = None,
) -> UpdateResult:
    if isinstance(error, BaseException):
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    else:
        error_payload = {"type": "RuntimeError", "message": error, "traceback": ""}
    summary = {
        "stage": stage,
        "error": error_payload,
        "logs": logs or {},
        "update_dir": str(update_dir),
    }
    summary_path = update_dir / "update_error.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return UpdateResult(
        update_index=update_index,
        changed=False,
        status="error",
        logs=summary,
        artifacts={"update_dir": str(update_dir), "error_summary": str(summary_path)},
    )


def safe_native_update(
    *,
    update_index: int,
    update_dir: Path,
    stage: str,
    fn: Callable[[], UpdateResult],
) -> UpdateResult:
    try:
        return fn()
    except Exception as exc:
        return native_error_result(update_index=update_index, update_dir=update_dir, error=exc, stage=stage)


@contextmanager
def patched_runtime_env(runtime: NativeRuntimeConfig):
    expanded = {key: os.path.expandvars(value) for key, value in runtime.env.items()}
    previous = {key: os.environ.get(key) for key in expanded}
    try:
        os.environ.update(expanded)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime_env(runtime: NativeRuntimeConfig) -> dict[str, str]:
    return {**os.environ, **{key: os.path.expandvars(value) for key, value in runtime.env.items()}}
