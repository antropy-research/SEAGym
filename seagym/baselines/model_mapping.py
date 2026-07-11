from __future__ import annotations

"""Shared update-model mapping helpers for external baseline adapters."""

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class UpdateModelBinding:
    model: str
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        models: dict[str, Any],
        *,
        default_model: str,
        default_model_ref: str = "update_model",
        config_ref_key: str = "update_model_ref",
        config_model_keys: tuple[str, ...] = ("model",),
    ) -> "UpdateModelBinding":
        update_model = models.get(str(config.get(config_ref_key, default_model_ref))) if isinstance(models, dict) else {}
        if not isinstance(update_model, dict):
            update_model = {}
        model = _first_present(update_model, ("model",))
        if model is None:
            model = _first_present(config, config_model_keys)
        return cls(
            model=str(model or default_model),
            provider=_optional_str(update_model.get("provider") or config.get("provider")),
            api_base=_optional_str(update_model.get("api_base") or config.get("api_base")),
            api_key_env=_optional_str(update_model.get("api_key_env") or config.get("api_key_env")),
        )

    @property
    def provider_name(self) -> str:
        return (self.provider or "").strip().lower()

    def pydantic_ai_model(self) -> str:
        text = _strip_litellm_prefix(self.model)
        provider = self.provider_name
        if provider == "deepseek" or text.startswith(("deepseek/", "deepseek:")):
            return _provider_colon_model("deepseek", text, default_prefix="deepseek")
        if self._is_deepseek_openai_compatible(text):
            return _provider_colon_model("deepseek", text, default_prefix="deepseek")
        if provider in {"openai", "openai-chat", "openai_chat"} or text.startswith("openai/"):
            return _provider_colon_model("openai-chat", text, default_prefix="openai")
        if provider in {"openai_compatible", "openai-compatible"} or self.api_base:
            return _provider_colon_model("openai-chat", text, default_prefix="")
        if _looks_like_openai_model(text):
            return _provider_colon_model("openai-chat", text, default_prefix="")
        return self.model.strip()

    def pydantic_ai_env(self, model: str | None = None) -> dict[str, str]:
        model = model or self.pydantic_ai_model()
        if model.startswith("deepseek:"):
            return {"DEEPSEEK_API_KEY": _env_ref(self.api_key_env or "DEEPSEEK_API_KEY")}
        if model.startswith(("openai-chat:", "openai:")):
            env = {"OPENAI_API_KEY": _env_ref(self.api_key_env or "OPENAI_API_KEY")}
            if self.api_base:
                env["OPENAI_BASE_URL"] = self.api_base
            return env
        return {}

    def litellm_model(self) -> str:
        text = _strip_litellm_prefix(self.model)
        provider = self.provider_name
        if provider == "deepseek" or text.startswith(("deepseek/", "deepseek:")):
            return _provider_slash_model("deepseek", text, default_prefix="deepseek")
        if self._is_deepseek_openai_compatible(text):
            return _provider_slash_model("deepseek", text, default_prefix="deepseek")
        if provider in {"openai", "openai-chat", "openai_chat"} or text.startswith("openai/"):
            return _provider_slash_model("openai", text, default_prefix="openai")
        if provider in {"openai_compatible", "openai-compatible"} or self.api_base:
            return _provider_slash_model("openai", text, default_prefix="")
        if _looks_like_openai_model(text):
            return _provider_slash_model("openai", text, default_prefix="")
        return text

    def litellm_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if api_key:
                kwargs["api_key"] = api_key
        return kwargs

    def openai_compatible_env(self, *, prefix: str = "UTU_LLM") -> dict[str, str]:
        settings = self.openai_compatible_settings()
        return {
            f"{prefix}_TYPE": "chat.completions",
            f"{prefix}_MODEL": settings.model,
            f"{prefix}_BASE_URL": settings.base_url,
            f"{prefix}_API_KEY": _env_ref(settings.api_key_env),
        }

    def openai_compatible_settings(self) -> "OpenAICompatibleSettings":
        return OpenAICompatibleSettings(
            model=_strip_provider(_strip_litellm_prefix(self.model)),
            base_url=self.openai_compatible_base_url(),
            api_key_env=self.api_key_env or self.default_api_key_env(),
        )

    def openai_compatible_base_url(self) -> str:
        if self.api_base:
            return self.api_base
        provider = self.provider_name
        text = _strip_litellm_prefix(self.model)
        if provider == "deepseek" or text.startswith(("deepseek/", "deepseek:")):
            return "https://api.deepseek.com/v1"
        return "https://api.openai.com/v1"

    def default_api_key_env(self) -> str:
        provider = self.provider_name
        text = _strip_litellm_prefix(self.model)
        if provider == "deepseek" or text.startswith(("deepseek/", "deepseek:")) or self._is_deepseek_openai_compatible(text):
            return "DEEPSEEK_API_KEY"
        return "OPENAI_API_KEY"

    def _is_deepseek_openai_compatible(self, model: str) -> bool:
        if self.provider_name not in {"openai_compatible", "openai-compatible"}:
            return False
        return (
            model.startswith(("deepseek/", "deepseek:"))
            or (self.api_base is not None and "deepseek" in self.api_base.lower())
            or self.api_key_env == "DEEPSEEK_API_KEY"
        )


def _provider_colon_model(provider: str, model: str, *, default_prefix: str) -> str:
    text = model.strip()
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if prefix == provider:
            return text
        if prefix in {"deepseek", "openai", "openai-chat", "openai_chat"}:
            text = rest
    if "/" in text:
        prefix, rest = text.split("/", 1)
        if not default_prefix or prefix == default_prefix:
            text = rest
    return f"{provider}:{text}"


def _provider_slash_model(provider: str, model: str, *, default_prefix: str) -> str:
    text = model.strip()
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if prefix == provider:
            text = rest
        elif prefix in {"deepseek", "openai", "openai-chat", "openai_chat"}:
            text = rest
    if "/" in text:
        prefix, rest = text.split("/", 1)
        if prefix == provider:
            return text
        if not default_prefix or prefix == default_prefix:
            text = rest
    return f"{provider}/{text}"


def _strip_provider(model: str) -> str:
    text = model.strip()
    if ":" in text:
        return text.split(":", 1)[1]
    if "/" in text:
        return text.split("/", 1)[1]
    return text


def _strip_litellm_prefix(model: str) -> str:
    text = model.strip()
    return text.split(":", 1)[1] if text.startswith("litellm:") else text


def _looks_like_openai_model(model: str) -> bool:
    prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-", "codex-")
    return model.startswith(prefixes)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _env_ref(name: str) -> str:
    return "${" + name + "}"


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    model: str
    base_url: str
    api_key_env: str
