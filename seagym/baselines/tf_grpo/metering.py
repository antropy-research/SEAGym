from __future__ import annotations

import threading
import time
from typing import Any


class _TFGRPOMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0.0
        self.output_tokens = 0.0
        self.cache_tokens = 0.0
        self.total_tokens = 0.0

    def add_usage(self, usage: Any) -> None:
        cost = _openai_usage_cost(usage)
        if not cost:
            return
        with self._lock:
            self.input_tokens += cost.get("input_tokens", 0.0)
            self.output_tokens += cost.get("output_tokens", 0.0)
            self.cache_tokens += cost.get("cache_tokens", 0.0)
            self.total_tokens += cost.get("total_tokens", 0.0)

    def cost(self) -> dict[str, float]:
        if self.total_tokens <= 0:
            return {}
        cost = {"total_tokens": self.total_tokens}
        if self.input_tokens:
            cost["input_tokens"] = self.input_tokens
        if self.output_tokens:
            cost["output_tokens"] = self.output_tokens
        if self.cache_tokens:
            cost["cache_tokens"] = self.cache_tokens
        return cost


def _patch_experience_updater_llm(experience_module: Any, meter: _TFGRPOMeter) -> Any:
    original_llm = getattr(experience_module, "LLM", None)
    if original_llm is None:
        return getattr(experience_module, "ExperienceUpdater")

    class MeteredLLM(original_llm):  # type: ignore[misc, valid-type]
        def chat(self, messages_or_prompt, max_tokens=16384, temperature=0, max_retries=3, return_reasoning=False):
            for _ in range(max_retries):
                try:
                    if isinstance(messages_or_prompt, str):
                        messages = [{"role": "user", "content": messages_or_prompt}]
                    elif isinstance(messages_or_prompt, list):
                        messages = messages_or_prompt
                    else:
                        raise ValueError("messages_or_prompt must be a string or a list of messages.")

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    meter.add_usage(getattr(response, "usage", None))
                    response_text = response.choices[0].message.content.strip()

                    if return_reasoning:
                        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                        return response_text, reasoning
                    return response_text
                except Exception as exc:
                    print(f"An unexpected error occurred: {exc}")
                time.sleep(10)

    setattr(experience_module, "LLM", MeteredLLM)
    return getattr(experience_module, "ExperienceUpdater")


def _openai_usage_cost(usage: Any) -> dict[str, float]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        input_tokens = _usage_number(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_number(usage, "completion_tokens", "output_tokens")
        total_tokens = _usage_number(usage, "total_tokens")
        cache_tokens = _usage_number(usage.get("prompt_tokens_details") or {}, "cached_tokens")
    else:
        input_tokens = _usage_attr_number(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_attr_number(usage, "completion_tokens", "output_tokens")
        total_tokens = _usage_attr_number(usage, "total_tokens")
        details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
        cache_tokens = _usage_attr_number(details, "cached_tokens") if details is not None else 0.0
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return {}
    cost = {"total_tokens": total_tokens}
    if input_tokens:
        cost["input_tokens"] = input_tokens
    if output_tokens:
        cost["output_tokens"] = output_tokens
    if cache_tokens:
        cost["cache_tokens"] = cache_tokens
    return cost


def _usage_number(value: dict[str, Any], *keys: str) -> float:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, int | float):
            return float(raw)
    return 0.0


def _usage_attr_number(value: Any, *keys: str) -> float:
    for key in keys:
        raw = getattr(value, key, None)
        if isinstance(raw, int | float):
            return float(raw)
    return 0.0


