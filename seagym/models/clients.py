from __future__ import annotations

"""Small chat model client layer for baseline-side model calls."""

import os
from typing import Any, Protocol

from .config import ModelConfig


class ChatModelClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        ...


class LiteLLMChatModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, *, system: str, user: str) -> str:
        import litellm

        kwargs = _completion_kwargs(self.config, system=system, user=user)
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        api_key = _api_key(self.config)
        if api_key:
            kwargs["api_key"] = api_key
        return completion_text(litellm.completion(**kwargs))


class OpenAICompatibleChatModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, *, system: str, user: str) -> str:
        from openai import OpenAI

        api_key = _api_key(self.config)
        if not api_key:
            env_name = self.config.api_key_env or "OPENAI_API_KEY"
            raise ValueError(f"Missing API key environment variable: {env_name}")
        client = OpenAI(api_key=api_key, base_url=self.config.api_base)
        response = client.chat.completions.create(**_completion_kwargs(self.config, system=system, user=user))
        return completion_text(response)


def build_chat_model_client(config: ModelConfig) -> ChatModelClient:
    if config.provider == "openai_compatible" or config.api_base:
        return OpenAICompatibleChatModelClient(config)
    if config.provider == "litellm":
        return LiteLLMChatModelClient(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _completion_kwargs(config: ModelConfig, *, system: str, user: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.extra_body:
        kwargs["extra_body"] = dict(config.extra_body)
    return kwargs


def _api_key(config: ModelConfig) -> str | None:
    if not config.api_key_env:
        return None
    return os.environ.get(config.api_key_env)


def completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return str(content or "")
