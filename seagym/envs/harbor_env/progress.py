from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time


_OUTPUT_CAPTURE_LIMIT = 20000


class _StreamCapture:
    def __init__(self, limit: int = _OUTPUT_CAPTURE_LIMIT):
        self.limit = limit
        self._parts: list[str] = []
        self._size = 0
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        if len(text) > self.limit:
            text = text[-self.limit :]
        with self._lock:
            self._parts.append(text)
            self._size += len(text)
            while self._size > self.limit and self._parts:
                removed = self._parts.pop(0)
                self._size -= len(removed)

    def text(self) -> str:
        with self._lock:
            value = "".join(self._parts)
        if len(value) > self.limit:
            return value[-self.limit :]
        return value


def run_harbor_with_progress(
    command: list[str],
    *,
    job_dir: Path,
    job_name: str,
    task_count: int,
    n_concurrent: int,
    view_name: str,
    mode: str,
    poll_interval_sec: float = 5.0,
    status_interval_sec: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    print(
        "seagym progress: harbor job started "
        f"job={job_name} view={view_name} mode={mode} tasks={task_count} n_concurrent={n_concurrent}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if hasattr(process, "stdout") and hasattr(process, "stderr"):
        stdout_capture, stdout_thread = _drain_stream(process.stdout)
        stderr_capture, stderr_thread = _drain_stream(process.stderr)
    else:
        stdout_capture = stderr_capture = None
        stdout_thread = stderr_thread = None
    last_status = ""
    last_print = 0.0
    while process.poll() is None:
        now = time.time()
        if now - last_print >= status_interval_sec:
            status = harbor_job_status(job_dir)
            if status != last_status:
                print(f"seagym progress: harbor job running job={job_name} {status}", flush=True)
                last_status = status
            last_print = now
        time.sleep(poll_interval_sec)
    if stdout_thread is not None and stderr_thread is not None:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        stdout = stdout_capture.text() if stdout_capture is not None else ""
        stderr = stderr_capture.text() if stderr_capture is not None else ""
    else:
        stdout, stderr = process.communicate()
    status = harbor_job_status(job_dir)
    print(
        "seagym progress: harbor job finished "
        f"job={job_name} returncode={process.returncode} {status}",
        flush=True,
    )
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _drain_stream(stream) -> tuple[_StreamCapture, threading.Thread]:
    capture = _StreamCapture()

    def _read() -> None:
        try:
            for chunk in iter(stream.readline, ""):
                capture.append(chunk)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return capture, thread


def harbor_job_status(job_dir: Path) -> str:
    result_path = job_dir / "result.json"
    trial_results = len(list(job_dir.glob("*/result.json"))) if job_dir.exists() else 0
    trial_exceptions = len(list(job_dir.glob("*/exception.txt"))) if job_dir.exists() else 0
    external_results = len(list(job_dir.glob("*/agent/external_project_result.json"))) if job_dir.exists() else 0
    if not result_path.exists():
        return (
            f"status=starting trial_results={trial_results} "
            f"trial_exceptions={trial_exceptions} external_results={external_results}"
        )
    try:
        data = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return (
            f"status=unreadable_result trial_results={trial_results} "
            f"trial_exceptions={trial_exceptions} external_results={external_results}"
        )
    stats = data.get("stats") or {}
    return (
        f"completed={stats.get('n_completed_trials', 0)} "
        f"running={stats.get('n_running_trials', 0)} "
        f"pending={stats.get('n_pending_trials', 0)} "
        f"errored={stats.get('n_errored_trials', 0)} "
        f"cancelled={stats.get('n_cancelled_trials', 0)} "
        f"retries={stats.get('n_retries', 0)} "
        f"trial_results={trial_results} "
        f"trial_exceptions={trial_exceptions} "
        f"external_results={external_results}"
    )


def format_harbor_error(completed: subprocess.CompletedProcess[str], job_dir: Path) -> str:
    parts = [f"harbor run failed with exit code {completed.returncode}"]
    if completed.stderr:
        parts.append(f"stderr: {completed.stderr.strip()}")
    if completed.stdout:
        parts.append(f"stdout: {completed.stdout.strip()}")
    if not job_dir.exists():
        parts.append(f"job_dir missing: {job_dir}")
    else:
        parts.append(f"job_dir: {job_dir}")
    return "\n".join(parts)
