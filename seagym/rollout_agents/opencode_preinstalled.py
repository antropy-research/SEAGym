from __future__ import annotations

"""Harbor OpenCode agent variant that expects a prebuilt sandbox runtime."""

try:  # pragma: no cover - Harbor is available in experiment environments.
    from harbor.agents.installed.opencode import OpenCode
    from harbor.environments.base import BaseEnvironment
except ModuleNotFoundError:  # pragma: no cover
    OpenCode = object  # type: ignore[assignment,misc]
    BaseEnvironment = object  # type: ignore[assignment,misc]


class PreinstalledOpenCode(OpenCode):  # pragma: no cover - exercised inside Harbor trials.
    """Use OpenCode from the E2B template instead of installing it per trial."""

    @staticmethod
    def name() -> str:
        return "preinstalled-opencode"

    async def install(self, environment: BaseEnvironment) -> None:
        command = (
            "set -euo pipefail; "
            'if [ -f "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi; '
            "command -v node >/dev/null 2>&1 || "
            "{ echo 'node is not preinstalled in this sandbox template' >&2; exit 127; }; "
            "command -v opencode >/dev/null 2>&1 || "
            "{ echo 'opencode is not preinstalled in this sandbox template' >&2; exit 127; }; "
            'mkdir -p "$HOME/.nvm"; '
            '[ -f "$HOME/.nvm/nvm.sh" ] || : > "$HOME/.nvm/nvm.sh"; '
            "node --version; "
            "opencode --version"
        )
        await self.exec_as_agent(environment, command=command)
