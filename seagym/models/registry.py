from __future__ import annotations

"""Model client registry."""

from .clients import ChatModelClient, build_chat_model_client
from .config import ModelConfig


class ModelRegistry:
    def build_chat_client(self, config: ModelConfig) -> ChatModelClient:
        return build_chat_model_client(config)

