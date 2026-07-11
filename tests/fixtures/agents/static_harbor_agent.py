from __future__ import annotations

"""Harbor custom rollout agent for static baseline tests."""

import json
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


STATIC_AGENT_IMPORT_PATH = "tests.fixtures.agents.static_harbor_agent:StaticHarborAgent"


class StaticHarborAgent(BaseAgent):
    """Minimal Harbor custom agent for SEAGym rollout integration tests.

    By default this agent only records the instruction and environment metadata
    in the Harbor agent log directory. A `run_command` kwarg can be supplied for
    task-specific smoke tests, e.g. Harbor hello-world can use
    `run_command="echo 'Hello, world!' > /app/hello.txt"`.
    """

    MARKER_FILENAME = "STATIC_AGENT_RAN.json"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        run_command: str | None = None,
        baseline_dir: str | None = None,
        state_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, extra_env=extra_env, **kwargs)
        self.run_command = run_command
        self.baseline_dir = baseline_dir
        self.state_dir = state_dir

    @staticmethod
    def name() -> str:
        return "seagym-static"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        marker = {
            "agent": self.name(),
            "version": self.version(),
            "model_name": self.model_name,
            "baseline_dir": self.baseline_dir,
            "state_dir": self.state_dir,
            "has_run_command": self.run_command is not None,
            "extra_env_keys": sorted(self.extra_env),
            "instruction_preview": instruction[:500],
        }
        marker_path = self.logs_dir / self.MARKER_FILENAME
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.run_command:
            result = await environment.exec(command=self.run_command, env=self.extra_env or None)
            context.metadata = {
                "static_agent": {
                    "command": self.run_command,
                    "return_code": result.return_code,
                }
            }
            if result.return_code != 0:
                raise RuntimeError(
                    f"Static baseline run_command failed with exit code {result.return_code}: "
                    f"{self.run_command}\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )
        else:
            context.metadata = {"static_agent": {"command": None, "return_code": None}}
