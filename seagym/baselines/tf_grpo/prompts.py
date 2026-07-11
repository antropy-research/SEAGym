from __future__ import annotations

import importlib
import json
from typing import Any


def _patch_tf_grpo_dotenv_lookup() -> None:
    try:
        dotenv = importlib.import_module("dotenv")
        dotenv_main = importlib.import_module("dotenv.main")
    except ImportError:
        return

    def _empty_dotenv(*args: Any, **kwargs: Any) -> str:
        return ""

    setattr(dotenv, "find_dotenv", _empty_dotenv)
    setattr(dotenv_main, "find_dotenv", _empty_dotenv)


def _apply_update_prompt_profile(experience_module: Any, profile: str) -> None:
    prompt_module_name = experience_module.__name__.rsplit(".", 1)[0] + ".prompts"
    try:
        prompts = importlib.import_module(prompt_module_name)
    except ImportError:
        if profile in ("", "native", "default"):
            return
        raise
    _snapshot_native_tf_grpo_prompts(experience_module, prompts)
    if profile in ("", "native", "default"):
        _restore_native_tf_grpo_prompts(experience_module, prompts)
        return
    if profile != "meta":
        raise ValueError(f"Unsupported TF-GRPO update_prompt_profile: {profile}")
    if hasattr(prompts, "SINGLE_ROLLOUT_SUMMARY_TEMPLATE"):
        prompts.SINGLE_ROLLOUT_SUMMARY_TEMPLATE = _META_SINGLE_ROLLOUT_SUMMARY_TEMPLATE
        prompts.SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = _META_SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE
        prompts.SINGLE_QUERY_CRITIQUE_TEMPLATE = _META_SINGLE_QUERY_CRITIQUE_TEMPLATE
        prompts.SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = _META_SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE
        prompts.BATCH_EXPERIENCE_UPDATE_TEMPLATE = _META_BATCH_EXPERIENCE_UPDATE_TEMPLATE
        for name in (
            "SINGLE_ROLLOUT_SUMMARY_TEMPLATE",
            "SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE",
            "SINGLE_QUERY_CRITIQUE_TEMPLATE",
            "SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE",
            "BATCH_EXPERIENCE_UPDATE_TEMPLATE",
        ):
            if hasattr(experience_module, name):
                setattr(experience_module, name, getattr(prompts, name))


_TF_GRPO_PROMPT_NAMES = (
    "SINGLE_ROLLOUT_SUMMARY_TEMPLATE",
    "SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE",
    "SINGLE_QUERY_CRITIQUE_TEMPLATE",
    "SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE",
    "BATCH_EXPERIENCE_UPDATE_TEMPLATE",
)


def _snapshot_native_tf_grpo_prompts(experience_module: Any, prompts_module: Any) -> None:
    if hasattr(prompts_module, "_seagym_native_prompt_snapshot"):
        return
    snapshot: dict[str, dict[str, Any]] = {"prompts": {}, "experience": {}}
    for name in _TF_GRPO_PROMPT_NAMES:
        if hasattr(prompts_module, name):
            snapshot["prompts"][name] = getattr(prompts_module, name)
        if hasattr(experience_module, name):
            snapshot["experience"][name] = getattr(experience_module, name)
    setattr(prompts_module, "_seagym_native_prompt_snapshot", snapshot)


def _restore_native_tf_grpo_prompts(experience_module: Any, prompts_module: Any) -> None:
    snapshot = getattr(prompts_module, "_seagym_native_prompt_snapshot", None)
    if not isinstance(snapshot, dict):
        return
    for name, value in (snapshot.get("prompts") or {}).items():
        setattr(prompts_module, name, value)
    for name, value in (snapshot.get("experience") or {}).items():
        setattr(experience_module, name, value)


_META_SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system may be provided with learned experiences, then attempts a task. Summarize the trajectory step-by-step.

For each step, extract the action, tool or file operation, observation, reasoning, outcome, and any experience that appears to be used. Given the grading and correct answer, identify detours, errors, missing checks, or effective decisions.

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{grade}
</evaluation>

<groundtruth>
{answer}
</groundtruth>

Return only a concise step-by-step trajectory summary."""

_META_SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = """An agent system may be provided with learned experiences, then attempts a task. Summarize the trajectory step-by-step.

For each step, extract the action, tool or file operation, observation, reasoning, outcome, and any experience that appears to be used. Identify likely detours, errors, missing checks, or effective decisions from the trajectory and verifier outcome.

<trajectory>
{trajectory}
</trajectory>

Return only a concise step-by-step trajectory summary."""

_META_SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the same task multiple times with both successful and wrong solutions. Review these attempts and extract generalizable experiences.

Focus on reusable task-solving behavior, not benchmark plumbing. Avoid advice about missing logs, missing trajectories, Harbor, file paths, result metadata, or evaluation infrastructure unless that behavior directly affected task success.

You have two options: [modify, add]. Produce at most {max_operations} clear, generalizable lessons.

After reasoning, return JSON:
```json
[
  {{"option": "modify", "experience": "the modified experience", "modified_from": "G17"}},
  {{"option": "add", "experience": "the added experience"}}
]
```

<problem>
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<groundtruth>
{answer}
</groundtruth>

<experience>
{experiences}
</experience>"""

_META_SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the same task multiple times. Review these attempts and extract generalizable experiences.

Use verifier success/failure labels as weak feedback. Focus on reusable task-solving behavior, not benchmark plumbing. Avoid advice about missing logs, missing trajectories, Harbor, file paths, result metadata, or evaluation infrastructure unless that behavior directly affected task success.

You have two options: [modify, add]. Produce at most {max_operations} clear, generalizable lessons.

After reasoning, return JSON:
```json
[
  {{"option": "modify", "experience": "the modified experience", "modified_from": "G17"}},
  {{"option": "add", "experience": "the added experience"}}
]
```

<problem>
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<experience>
{experiences}
</experience>"""

_META_BATCH_EXPERIENCE_UPDATE_TEMPLATE = """An agent system has proposed updates to a set of learned experiences. Produce a concise final experience revision plan.

Each final experience must be a reusable behavioral guideline, no more than 32 words. Avoid benchmark plumbing, result metadata, task IDs, local paths, and logging advice unless it directly helps solve future tasks.

<existing_experiences>
{experiences}
</existing_experiences>

<suggested_updates>
{updates}
</suggested_updates>

You have two update options: [modify, merge].

Return JSON:
```json
[
  {{"option": "modify", "experience": "the modified experience", "modified_from": "C1"}},
  {{"option": "merge", "experience": "the merged experience", "merged_from": ["C1", "C3"]}}
]
```"""


def _format_experiences_prompt(experiences: dict[str, Any]) -> str:
    if not experiences:
        return ""
    lines = ["Use the following learned experiences when solving future tasks:"]
    for key, value in experiences.items():
        lines.append(f"[{key}] {value}")
    return "\n".join(lines) + "\n"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


