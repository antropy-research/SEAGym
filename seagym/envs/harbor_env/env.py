from __future__ import annotations

"""Harbor environment backend."""

import json
from pathlib import Path
from typing import Any

from seagym.data.types import TaskRecord
from seagym.utils import read_json

from .commands import (
    format_agent_kwarg,
    group_tasks_by_run_path,
    harbor_batch_job_name,
    materialize_timeout_patched_dataset,
    task_local_path,
    task_run_path,
    templatize_env,
)
from .matching import find_trial_results, match_task_for_trial, task_lookup
from .progress import format_harbor_error, run_harbor_with_progress
from .results import attempt_ref, failed_harbor_result, normalize_harbor_trial_result, select_representative_attempt
from .spec import HarborAgentSpec, PROXY_ENV_KEYS
from ..results import TaskRunResult


class HarborEnv:
    """Harbor CLI environment for local Harbor-compatible task paths."""

    def __init__(
        self,
        jobs_dir: str | Path,
        *,
        harbor_bin: str = "harbor",
        n_concurrent: int = 1,
        env: str = "docker",
        environment_import_path: str | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        yes: bool = True,
        extra_args: list[str] | None = None,
        agent_spec: HarborAgentSpec | None = None,
        model_name: str | None = None,
        agent_import_path: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        agent_env: dict[str, str] | None = None,
        verifier_env: dict[str, str] | None = None,
        agent_override_timeout_sec: int | None = None,
        verifier_override_timeout_sec: int | None = None,
        preserve_task_order: bool = False,
    ):
        self.jobs_dir = Path(jobs_dir).resolve()
        self.harbor_bin = harbor_bin
        self.n_concurrent = n_concurrent
        self.env = env
        self.environment_import_path = environment_import_path
        self.environment_kwargs = dict(environment_kwargs or {})
        self.yes = yes
        self.extra_args = list(extra_args or [])
        self.model_name = model_name
        self._strip_proxy_env = env == "e2b"
        if agent_spec is not None:
            self.configure_agent_spec(agent_spec)
            self.agent_env = {**self._filtered_env(agent_env), **self.agent_env}
        else:
            self.agent_id = None
            self.agent_import_path = agent_import_path
            self.agent_kwargs = dict(agent_kwargs or {})
            self.agent_env = self._filtered_env(agent_env)
            self.n_attempts = 1
        self.verifier_env = self._filtered_env(verifier_env)
        self.agent_override_timeout_sec = agent_override_timeout_sec
        self.verifier_override_timeout_sec = verifier_override_timeout_sec
        self.preserve_task_order = preserve_task_order

    def configure_agent_spec(self, agent_spec: HarborAgentSpec) -> None:
        self.agent_id = agent_spec.agent_id
        self.agent_import_path = agent_spec.import_path
        self.agent_kwargs = dict(agent_spec.kwargs)
        self.agent_env = self._filtered_env(agent_spec.env)
        self.n_attempts = max(1, int(agent_spec.n_attempts))

    def run_tasks(
        self,
        tasks: list[TaskRecord],
        *,
        view_name: str,
        mode: str,
        agent_id: str,
    ) -> list[TaskRunResult]:
        """Run tasks in as few Harbor jobs as possible."""
        results_by_task_id: dict[str, TaskRunResult] = {}
        group_results = self._run_task_group(tasks, view_name=view_name, mode=mode, agent_id=agent_id)
        results_by_task_id.update({result.task_id: result for result in group_results})
        return [
            results_by_task_id.get(
                task.task_id,
                failed_harbor_result(
                    task,
                    view_name,
                    mode,
                    agent_id,
                    f"Harbor batch execution did not produce a result for task {task.task_id}",
                ),
            )
            for task in tasks
        ]

    def run_task_attempts(
        self,
        tasks: list[TaskRecord],
        *,
        view_name: str,
        mode: str,
        agent_id: str,
    ) -> list[TaskRunResult]:
        """Run tasks and return every Harbor attempt as a separate result."""
        results_by_task_id: dict[str, list[TaskRunResult]] = {}
        group_results = self._run_task_group(
            tasks,
            view_name=view_name,
            mode=mode,
            agent_id=agent_id,
            return_all_attempts=True,
        )
        for result in group_results:
            results_by_task_id.setdefault(result.task_id, []).append(result)
        results: list[TaskRunResult] = []
        for task in tasks:
            attempts = results_by_task_id.get(task.task_id, [])
            if attempts:
                results.extend(attempts)
                continue
            results.append(
                failed_harbor_result(
                    task,
                    view_name,
                    mode,
                    agent_id,
                    f"Harbor job completed but no trial result matched task {task.task_id}",
                )
            )
        return results

    def _run_task_group(
        self,
        tasks: list[TaskRecord],
        *,
        view_name: str,
        mode: str,
        agent_id: str,
        return_all_attempts: bool = False,
        n_concurrent: int | None = None,
    ) -> list[TaskRunResult]:
        effective_n_concurrent = max(1, int(n_concurrent or self.n_concurrent))
        try:
            command, job_name = self.build_batch_command(
                tasks,
                agent_id=agent_id,
                n_concurrent=effective_n_concurrent,
            )
        except ValueError as exc:
            return [failed_harbor_result(task, view_name, mode, agent_id, str(exc)) for task in tasks]

        job_dir = self.jobs_dir / job_name
        completed = run_harbor_with_progress(
            command,
            job_dir=job_dir,
            job_name=job_name,
            task_count=len(tasks),
            n_concurrent=effective_n_concurrent,
            view_name=view_name,
            mode=mode,
        )
        refs = {
            "env": "harbor",
            "agent_id": agent_id,
            "command": command,
            "job_dir": str(job_dir),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "harbor_returncode": completed.returncode,
        }

        result_paths = find_trial_results(job_dir)
        if completed.returncode != 0 and not result_paths:
            error = format_harbor_error(completed, job_dir)
            return [failed_harbor_result(task, view_name, mode, agent_id, error, refs=refs) for task in tasks]

        if not result_paths:
            error = format_harbor_error(completed, job_dir)
            return [failed_harbor_result(task, view_name, mode, agent_id, error, refs=refs) for task in tasks]

        harbor_warning = None
        if completed.returncode != 0:
            harbor_warning = format_harbor_error(completed, job_dir)
            refs["harbor_warning"] = harbor_warning

        task_by_key = task_lookup(tasks)
        attempts_by_task_id: dict[str, list[TaskRunResult]] = {}
        for result_path in result_paths:
            data = read_json(result_path)
            task = match_task_for_trial(data, result_path, task_by_key)
            if task is None:
                continue
            result = normalize_harbor_trial_result(
                task,
                result_path,
                view_name=view_name,
                mode=mode,
                agent_id=agent_id,
                data=data,
            )
            result.refs["command"] = command
            result.refs["job_dir"] = str(job_dir)
            result.refs["harbor_stdout"] = completed.stdout
            result.refs["harbor_stderr"] = completed.stderr
            result.refs["harbor_returncode"] = completed.returncode
            if harbor_warning is not None:
                result.refs["harbor_warning"] = harbor_warning
            attempts_by_task_id.setdefault(task.task_id, []).append(result)

        results: list[TaskRunResult] = []
        for task in tasks:
            attempts = attempts_by_task_id.get(task.task_id, [])
            if return_all_attempts:
                if attempts:
                    for attempt_index, attempt in enumerate(attempts):
                        attempt.refs["attempt_index"] = attempt_index
                    results.extend(attempts)
                    continue
                results.append(
                    failed_harbor_result(
                        task,
                        view_name,
                        mode,
                        agent_id,
                        f"Harbor job completed but no trial result matched task {task.task_id}",
                        refs=refs,
                    )
                )
                continue
            result = select_representative_attempt(attempts)
            if result is not None:
                if len(attempts) > 1:
                    result.refs["all_attempts"] = [attempt_ref(attempt) for attempt in attempts]
                results.append(result)
                continue
            results.append(
                failed_harbor_result(
                    task,
                    view_name,
                    mode,
                    agent_id,
                    f"Harbor job completed but no trial result matched task {task.task_id}",
                    refs=refs,
                )
            )
        return results

    def build_batch_command(
        self,
        tasks: list[TaskRecord],
        *,
        agent_id: str,
        n_concurrent: int | None = None,
    ) -> tuple[list[str], str]:
        if not tasks:
            raise ValueError("Cannot build Harbor command for empty task list")
        if self.preserve_task_order:
            return self._build_config_batch_command(
                tasks,
                agent_id=agent_id,
                n_concurrent=max(1, int(n_concurrent or self.n_concurrent)),
                preserve_task_order=True,
            )
        run_path = task_run_path(tasks[0])
        if run_path is None:
            raise ValueError(f"Task {tasks[0].task_id} does not define source.dataset_path or source.local_path")
        if not run_path.exists():
            raise ValueError(f"Task {tasks[0].task_id} Harbor path does not exist: {run_path}")
        mixed_run_paths = len(group_tasks_by_run_path(tasks)) > 1
        if mixed_run_paths:
            return self._build_config_batch_command(
                tasks,
                agent_id=agent_id,
                n_concurrent=max(1, int(n_concurrent or self.n_concurrent)),
            )
        for task in tasks[1:]:
            candidate_run_path = task_run_path(task)
            if candidate_run_path != run_path:
                raise ValueError("Harbor batch command requires all tasks to share one runnable path")

        has_dataset_path = bool(tasks[0].source.get("dataset_path"))
        filters = [str(task.source.get("task_name")) for task in tasks if task.source.get("task_name")]
        job_name = harbor_batch_job_name(tasks)
        if self.agent_override_timeout_sec is not None or self.verifier_override_timeout_sec is not None:
            run_path = materialize_timeout_patched_dataset(
                run_path,
                self.jobs_dir / "_patched_tasksets" / job_name,
                task_names=filters if has_dataset_path else None,
                agent_timeout_sec=None if self.agent_override_timeout_sec is None else float(self.agent_override_timeout_sec),
                verifier_timeout_sec=None if self.verifier_override_timeout_sec is None else float(self.verifier_override_timeout_sec),
            )
        if has_dataset_path and len(filters) != len(tasks):
            raise ValueError("Harbor batch command requires source.task_name for every task")
        if not has_dataset_path and len(tasks) > 1:
            raise ValueError("Harbor batch command requires source.dataset_path for multi-task execution")

        command = [
            self.harbor_bin,
            "run",
            "-p",
            str(run_path),
            "--job-name",
            job_name,
            "--jobs-dir",
            str(self.jobs_dir),
            "-n",
            str(max(1, int(n_concurrent or self.n_concurrent))),
        ]
        if self.n_attempts > 1:
            command.extend(["-k", str(self.n_attempts)])
        if self.environment_import_path:
            command.extend(["-e", self.environment_import_path])
        else:
            command.extend(["-e", self.env])
        effective_agent_id = self.agent_id or agent_id
        if self.agent_import_path:
            command.extend(["--agent-import-path", self.agent_import_path])
        else:
            command.extend(["-a", effective_agent_id])
        if self.model_name:
            command.extend(["--model", self.model_name])
        for key, value in self.agent_kwargs.items():
            command.extend(["--agent-kwarg", f"{key}={format_agent_kwarg(value)}"])
        for key, value in self.agent_env.items():
            command.extend(["--agent-env", f"{key}={value}"])
        for key, value in self.verifier_env.items():
            if value != "":
                command.extend(["--verifier-env", f"{key}={value}"])
        for key, value in self.environment_kwargs.items():
            command.extend(["--environment-kwarg", f"{key}={format_agent_kwarg(value)}"])
        if has_dataset_path:
            for task_filter in filters:
                command.extend(["-i", task_filter])
            command.extend(["-l", str(len(tasks))])
        if self.yes:
            command.append("-y")
        command.extend(self.extra_args)
        return command, job_name

    def _build_config_batch_command(
        self,
        tasks: list[TaskRecord],
        *,
        agent_id: str,
        n_concurrent: int,
        preserve_task_order: bool = False,
    ) -> tuple[list[str], str]:
        job_name = harbor_batch_job_name(tasks)
        config_dir = self.jobs_dir / "_job_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{job_name}.json"
        config = self._job_config_for_tasks(
            tasks,
            job_name=job_name,
            n_concurrent=n_concurrent,
            agent_id=agent_id,
            preserve_task_order=preserve_task_order,
        )
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            self.harbor_bin,
            "run",
            "--config",
            str(config_path),
            "--job-name",
            job_name,
            "--jobs-dir",
            str(self.jobs_dir),
            "-n",
            str(n_concurrent),
        ]
        if self.yes:
            command.append("-y")
        command.extend(self.extra_args)
        return command, job_name

    def _job_config_for_tasks(
        self,
        tasks: list[TaskRecord],
        *,
        job_name: str,
        n_concurrent: int,
        agent_id: str,
        preserve_task_order: bool = False,
    ) -> dict[str, Any]:
        datasets: list[dict[str, Any]] = []
        task_configs: list[dict[str, Any]] = []
        if preserve_task_order:
            for task in tasks:
                task_path = task_local_path(task)
                if task_path is None:
                    raise ValueError(f"Task {task.task_id} does not define source.local_path or dataset task path")
                task_configs.append({"path": str(task_path)})
        else:
            for run_path, group in group_tasks_by_run_path(tasks).items():
                has_dataset_path = bool(group[0].source.get("dataset_path"))
                if has_dataset_path:
                    task_names = [str(task.source.get("task_name")).rsplit("/", 1)[-1] for task in group]
                    datasets.append({"path": str(run_path), "task_names": task_names})
                    continue
                for task in group:
                    task_path = task_run_path(task)
                    if task_path is None:
                        raise ValueError(f"Task {task.task_id} does not define source.dataset_path or source.local_path")
                    task_configs.append({"path": str(task_path)})

        environment: dict[str, Any] = {
            "force_build": False,
            "delete": True,
            "kwargs": dict(self.environment_kwargs),
        }
        if self.environment_import_path:
            environment["import_path"] = self.environment_import_path
        else:
            environment["type"] = self.env

        effective_agent_id = self.agent_id or agent_id
        agent: dict[str, Any] = {
            "kwargs": dict(self.agent_kwargs),
            "env": templatize_env(self.agent_env),
        }
        if self.agent_import_path:
            agent["import_path"] = self.agent_import_path
        else:
            agent["name"] = effective_agent_id
        if self.model_name:
            agent["model_name"] = self.model_name
        if self.agent_override_timeout_sec is not None:
            agent["override_timeout_sec"] = float(self.agent_override_timeout_sec)

        verifier: dict[str, Any] = {"env": templatize_env(self.verifier_env)}
        if self.verifier_override_timeout_sec is not None:
            verifier["override_timeout_sec"] = float(self.verifier_override_timeout_sec)

        return {
            "job_name": job_name,
            "jobs_dir": str(self.jobs_dir),
            "n_attempts": self.n_attempts,
            "n_concurrent_trials": n_concurrent,
            "environment": environment,
            "agents": [agent],
            "verifier": verifier,
            "datasets": datasets,
            "tasks": task_configs,
        }

    def _filtered_env(self, env: dict[str, str] | None) -> dict[str, str]:
        values = dict(env or {})
        if not self._strip_proxy_env:
            return values
        return {key: value for key, value in values.items() if key not in PROXY_ENV_KEYS}
