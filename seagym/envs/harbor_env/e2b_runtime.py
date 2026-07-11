from __future__ import annotations

"""Harbor E2B runtime customization used by SEAGym's Harbor backend."""

import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.e2b import E2BEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode


DEFAULT_E2B_SANDBOX_TIMEOUT_SEC = 2400
DEFAULT_E2B_STOP_TIMEOUT_SEC = 60


class E2BOneHourEnvironment(E2BEnvironment):
    """Harbor E2B environment with a provider-compatible sandbox timeout.

    The current local Harbor E2B implementation requests a 24 hour sandbox
    timeout. E2B currently rejects values greater than one hour, so SEAGym
    uses this thin runtime override until upstream Harbor exposes the timeout
    as config or lowers the default.
    """

    def __init__(
        self,
        *args,
        template_alias_mode: str | None = None,
        sandbox_timeout_sec: int = DEFAULT_E2B_SANDBOX_TIMEOUT_SEC,
        sandbox_stop_timeout_sec: int = DEFAULT_E2B_STOP_TIMEOUT_SEC,
        **kwargs,
    ):
        self._seagym_template_alias_mode = template_alias_mode
        self._seagym_sandbox_timeout_sec = int(sandbox_timeout_sec)
        self._seagym_sandbox_stop_timeout_sec = int(sandbox_stop_timeout_sec)
        super().__init__(*args, **kwargs)
        if template_alias_mode in {"ahe_task_name", "terminal_bench_task_name"}:
            self._template_name = _ahe_template_alias(self.environment_name)
        elif template_alias_mode == "opencode_task_name":
            self._template_name = _opencode_template_alias(self.environment_name)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.E2B

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(self):
        from e2b import AsyncSandbox

        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }

        self._sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata=metadata,
            timeout=self._seagym_sandbox_timeout_sec,
            allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
            network=self._sandbox_create_network_options(),
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _stop_sandbox(self):
        if self._sandbox:
            await asyncio.wait_for(
                self._sandbox.kill(),  # type: ignore[call-arg]
                timeout=self._seagym_sandbox_stop_timeout_sec,
            )


def _ahe_template_alias(environment_name: str) -> str:
    return environment_name.rsplit("/", 1)[-1].replace(".", "-")


def _opencode_template_alias(environment_name: str) -> str:
    return f"seagym-opencode-{_ahe_template_alias(environment_name)}"
