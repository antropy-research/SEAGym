from __future__ import annotations

"""AHE NexAU rollout agent binding."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import stat
from typing import Any

from seagym.baselines.base import BaselineState
from seagym.baselines.data import TaskBatch, TrajectoryBatch
from seagym.baselines.model_mapping import UpdateModelBinding
from seagym.data.types import TaskIndex
from seagym.envs.base import TaskEnv
from seagym.envs.harbor_env import HarborAgentSpec
from .harbor import HarborRolloutAgent

try:  # pragma: no cover - optional outside Harbor runtime.
    from harbor.agents.base import BaseAgent
except ModuleNotFoundError:  # pragma: no cover
    BaseAgent = object  # type: ignore[assignment,misc]


_AHE_SETUP_UPLOAD_SEMAPHORE: asyncio.Semaphore | None = None
_AHE_SETUP_UPLOAD_LIMIT: int | None = None


@dataclass
class AHENexAURolloutAgent(HarborRolloutAgent):
    model: str = "deepseek-v4-flash"
    api_base: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_type: str = "openai_chat_completion"
    reasoning: dict[str, Any] | None = None
    max_iterations: int | None = 300

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        models: dict[str, Any],
        run_dir: Path,
        base_dir: Path | None,
    ) -> "AHENexAURolloutAgent":
        del run_dir, base_dir
        rollout_model = UpdateModelBinding.from_config(
            config,
            models,
            default_model="deepseek/deepseek-chat",
            default_model_ref="rollout_model",
            config_ref_key="model_ref",
        ).openai_compatible_settings()
        return cls(
            agent_id=str(config.get("agent", name)),
            agent_import_path="seagym.rollout_agents.ahe_nexau:AHENexAUHarborAgent",
            model=rollout_model.model,
            api_base=rollout_model.base_url,
            api_key_env=rollout_model.api_key_env,
            api_type=str(config.get("api_type", "openai_chat_completion")),
            reasoning=dict(config["reasoning"]) if isinstance(config.get("reasoning"), dict) else None,
            max_iterations=None if config.get("max_iterations") in (None, "") else int(config["max_iterations"]),
        )

    def rollout(
        self,
        batch: TaskBatch,
        *,
        env: TaskEnv,
        task_index: TaskIndex,
        baseline_state: BaselineState,
    ) -> TrajectoryBatch:
        config_path = Path(str(baseline_state.metadata["agent_config_path"]))
        _patch_code_agent_config(
            config_path,
            api_type=self.api_type,
            reasoning=self.reasoning,
            max_iterations=self.max_iterations,
        )
        return super().rollout(batch, env=env, task_index=task_index, baseline_state=baseline_state)

    def harbor_agent_spec(
        self,
        baseline_state: BaselineState | None = None,
        *,
        n_attempts: int | None = None,
    ) -> HarborAgentSpec:
        if baseline_state is None:
            raise ValueError("AHENexAURolloutAgent requires AHE BaselineState")
        return HarborAgentSpec(
            agent_id=self.agent_id,
            import_path=self.agent_import_path,
            kwargs={
                "project_dir": baseline_state.metadata["project_dir"],
                "state_dir": str(baseline_state.state_dir),
                "max_iterations": self.max_iterations,
                "sandbox_work_dir": ".",
            },
            env={
                "LLM_API_KEY": os.environ.get(self.api_key_env, ""),
                "LLM_BASE_URL": self.api_base,
                "LLM_MODEL": self.model,
                "LLM_API_TYPE": self.api_type,
            },
            n_attempts=max(1, int(n_attempts if n_attempts is not None else self.n_attempts)),
        )


class AHENexAUHarborAgent(BaseAgent):  # pragma: no cover - exercised inside Harbor trials.
    """Harbor custom agent that runs AHE's NexAU code agent in the task sandbox."""

    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        project_dir: str = "",
        state_dir: str = "",
        max_iterations: int | None = 300,
        sandbox_work_dir: str = ".",
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        if BaseAgent is object:
            self.logs_dir = logs_dir
            self.model_name = model_name
            self._extra_env = dict(extra_env or {})
        else:
            super().__init__(logs_dir=logs_dir, model_name=model_name, extra_env=extra_env)
        # Harbor versions differ on whether BaseAgent exposes this attribute.
        # Keep the adapter's configured environment inspectable in either case.
        self.extra_env = dict(extra_env or {})
        self.project_dir = Path(project_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.max_iterations = max_iterations
        self.sandbox_work_dir = sandbox_work_dir
        self._ahe_extra_env = dict(extra_env or {})
        self.extra_kwargs = dict(kwargs)

    @staticmethod
    def name() -> str:
        return "ahe-nexau"

    def version(self) -> str | None:
        return "seagym"

    async def setup(self, environment) -> None:
        await environment.exec(command="mkdir -p /installed-agent")

        workspace = _ahe_workspace_dir(self.project_dir, self.state_dir)
        install_script = _render_nexau_install_script(self.project_dir, self.logs_dir)
        async with _ahe_setup_upload_semaphore():
            await environment.upload_dir(workspace, "/nexau-workspace")
            await environment.upload_file(
                source_path=install_script,
                target_path="/installed-agent/install.sh",
            )

        result = await environment.exec(command="bash /installed-agent/install.sh")
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "return-code.txt").write_text(str(getattr(result, "return_code", "")), encoding="utf-8")
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
        if stdout:
            (setup_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        if stderr:
            (setup_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        if getattr(result, "return_code", None) != 0:
            raise RuntimeError(
                f"AHE NexAU setup failed with exit code {getattr(result, 'return_code', None)}. "
                f"See logs in {setup_dir}"
            )
        dependency_result = await environment.exec(
            command=(
                "set -e; "
                "if [ ! -x /opt/nexau-venv/bin/python ]; then "
                "echo '/opt/nexau-venv/bin/python not found' >&2; exit 127; "
                "fi; "
                "if ! /opt/nexau-venv/bin/python -c 'import e2b'; then "
                "export PATH=\"$HOME/.local/bin:$PATH\"; "
                "test -f \"$HOME/.local/bin/env\" && . \"$HOME/.local/bin/env\" || true; "
                "if command -v uv >/dev/null 2>&1; then "
                "uv pip install --python /opt/nexau-venv/bin/python e2b; "
                "else "
                "/opt/nexau-venv/bin/python -m ensurepip --upgrade "
                "&& /opt/nexau-venv/bin/python -m pip install e2b; "
                "fi; "
                "fi; "
                "/opt/nexau-venv/bin/python -c 'import e2b'"
            )
        )
        (setup_dir / "dependency-return-code.txt").write_text(
            str(getattr(dependency_result, "return_code", "")),
            encoding="utf-8",
        )
        dependency_stdout = str(getattr(dependency_result, "stdout", "") or "")
        dependency_stderr = str(getattr(dependency_result, "stderr", "") or "")
        if dependency_stdout:
            (setup_dir / "dependency-stdout.txt").write_text(dependency_stdout, encoding="utf-8")
        if dependency_stderr:
            (setup_dir / "dependency-stderr.txt").write_text(dependency_stderr, encoding="utf-8")
        if getattr(dependency_result, "return_code", None) != 0:
            raise RuntimeError(
                f"AHE NexAU dependency setup failed with exit code {getattr(dependency_result, 'return_code', None)}. "
                f"See logs in {setup_dir}"
            )

    async def run(self, instruction: str, environment, context) -> None:
        await environment.exec("mkdir -p /logs/agent", timeout_sec=60)
        env = {
            **self._ahe_extra_env,
            "SANDBOX_WORK_DIR": self.sandbox_work_dir,
        }
        command = _nexau_harbor_command(instruction)
        result = await environment.exec(
            command,
            cwd="/nexau-workspace",
            env=env,
            timeout_sec=None,
        )
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata["ahe_nexau"] = {
                "return_code": getattr(result, "return_code", None),
                "stdout": getattr(result, "stdout", ""),
                "stderr": getattr(result, "stderr", ""),
            }
        return_code = getattr(result, "return_code", None)
        if return_code not in (None, 0):
            stderr = str(getattr(result, "stderr", "") or "").strip()
            stdout = str(getattr(result, "stdout", "") or "").strip()
            detail = stderr or stdout or f"return_code={return_code}"
            raise RuntimeError(f"AHE NexAU rollout failed: {detail[:2000]}")


def _patch_code_agent_config(
    config_path: Path,
    *,
    api_type: str,
    reasoning: dict[str, Any] | None,
    max_iterations: int | None,
) -> None:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML object in {config_path}")
    llm_config = config.setdefault("llm_config", {})
    if not isinstance(llm_config, dict):
        raise ValueError(f"Expected llm_config object in {config_path}")
    llm_config["api_type"] = api_type
    if reasoning:
        llm_config["reasoning"] = dict(reasoning)
    else:
        llm_config.pop("reasoning", None)
    if max_iterations is not None:
        config["max_iterations"] = max_iterations
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _ahe_setup_upload_semaphore() -> asyncio.Semaphore:
    global _AHE_SETUP_UPLOAD_LIMIT, _AHE_SETUP_UPLOAD_SEMAPHORE
    raw_limit = os.environ.get("AHE_NEXAU_UPLOAD_CONCURRENCY", "12")
    try:
        limit = max(1, int(raw_limit))
    except ValueError:
        limit = 12
    if _AHE_SETUP_UPLOAD_SEMAPHORE is None or _AHE_SETUP_UPLOAD_LIMIT != limit:
        _AHE_SETUP_UPLOAD_SEMAPHORE = asyncio.Semaphore(limit)
        _AHE_SETUP_UPLOAD_LIMIT = limit
    return _AHE_SETUP_UPLOAD_SEMAPHORE


def _stage_ahe_workspace_bundle(project_dir: Path, state_dir: Path, staging_root: Path) -> Path:
    bundle = Path(staging_root) / "seagym_ahe_workspace_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(
        _ahe_workspace_dir(project_dir, state_dir),
        bundle,
        ignore=shutil.ignore_patterns(".git", "run_root", "__pycache__", "*.pyc"),
    )
    return bundle


def _ahe_workspace_dir(project_dir: Path, state_dir: Path) -> Path:
    project_dir = Path(project_dir).resolve()
    state_dir = Path(state_dir).resolve()
    workspace = state_dir / "workspace"
    if not workspace.exists():
        return project_dir / "agents" / "code_agent_simple"
    return workspace


def _nexau_install_template_path(project_dir: Path) -> Path:
    use_prebuilt_template = os.environ.get("SEAGYM_AHE_USE_PREBUILT_E2B_TEMPLATE", "False")
    if use_prebuilt_template == "True":
        template_name = "install-nexau.sh.j2"
    else:
        template_name = "install-nexau_saas_e2b.j2"
    return (
        project_dir
        / ".venv"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "harbor"
        / "agents"
        / "installed"
        / template_name
    )


def _render_nexau_install_script(project_dir: Path, logs_dir: Path) -> Path:
    template_path = _nexau_install_template_path(project_dir)
    if not template_path.exists():
        raise FileNotFoundError(f"AHE NexAU install template not found: {template_path}")

    rendered = template_path.read_text(encoding="utf-8").replace(
        "{{ github_token }}",
        os.environ.get("GITHUB_TOKEN", ""),
    )
    script_path = Path(logs_dir) / "install.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(rendered, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return script_path


def _nexau_harbor_command(instruction: str) -> str:
    escaped_instruction = shlex.quote(instruction)
    return (
        "set -o pipefail; "
        "("
        "if [ -x /opt/nexau-venv/bin/nexau-harbor ]; then "
        "/opt/nexau-venv/bin/nexau-harbor run "
        "--config_path /nexau-workspace/code_agent.yaml "
        "--log_dir_path /logs/agent "
        f"--query {escaped_instruction}; "
        "else echo '/opt/nexau-venv/bin/nexau-harbor not found' >&2; exit 127; fi"
        ") 2>&1 </dev/null | tee /logs/agent/nexau.txt"
    )
