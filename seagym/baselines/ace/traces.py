from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from ..data import TrajectoryBatch


def _materialize_ace_traces(
    trajectories: TrajectoryBatch,
    *,
    max_records: int | None = None,
    feedback_mode: str = "reward_only",
    max_reasoning_chars: int = 20000,
) -> list[dict[str, Any]]:
    records = trajectories.trajectories
    if max_records is not None:
        records = records[:max_records]
    return [
        _materialize_ace_trace(
            trajectory,
            feedback_mode=feedback_mode,
            max_reasoning_chars=max_reasoning_chars,
        )
        for trajectory in records
    ]


def _materialize_ace_trace(
    trajectory: Any,
    *,
    feedback_mode: str,
    max_reasoning_chars: int,
) -> dict[str, Any]:
    trial_dir = _trial_dir_from_refs(trajectory.refs)
    agent_trace = _read_json(trial_dir / "agent" / "trajectory.json") if trial_dir else None
    result = _read_json(trial_dir / "result.json") if trial_dir else None

    question = _extract_question(agent_trace) or _extract_task_instruction(result) or str(trajectory.task_id)
    reasoning = _extract_reasoning(agent_trace, max_chars=max_reasoning_chars)
    answer = _extract_answer(agent_trace)
    feedback = _build_ace_feedback(trajectory, result=result, feedback_mode=feedback_mode)
    ground_truth = _extract_ground_truth(result) if feedback_mode == "gt_labels" else None

    return {
        "question": question,
        "context": _build_ace_context(trajectory, result=result, trial_dir=trial_dir),
        "reasoning": reasoning,
        "answer": answer,
        "feedback": feedback,
        "ground_truth": ground_truth,
        "skill_ids": [],
        "trace_id": _trace_id(trajectory),
        "source_system": "harbor" if trial_dir else "seagym",
        "task_id": str(trajectory.task_id),
        "metadata": {
            "attempt_id": trajectory.attempt_id,
            "view_name": trajectory.view_name,
            "mode": trajectory.mode,
            "success": trajectory.success,
            "reward": trajectory.reward,
            "score": trajectory.score,
            "rewards": trajectory.rewards,
            "result_path": None if trial_dir is None else str(trial_dir / "result.json"),
            "trial_dir": None if trial_dir is None else str(trial_dir),
            "feedback_mode": feedback_mode,
        },
    }


def _trial_dir_from_refs(refs: dict[str, Any]) -> Path | None:
    for key in ("result_path", "trial_path", "trial_dir"):
        value = refs.get(key)
        if isinstance(value, str) and value:
            path = _path_from_ref(value)
            return path.parent if path.name == "result.json" else path
    trial_uri = refs.get("trial_uri")
    if isinstance(trial_uri, str) and trial_uri:
        return _path_from_ref(trial_uri)
    return None


def _path_from_ref(value: str) -> Path:
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path))
    return Path(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_question(agent_trace: Any) -> str:
    if not isinstance(agent_trace, dict):
        return ""
    for step in agent_trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for content in _observation_contents(step):
            if "/app/instruction.md" not in content and "instruction.md" not in content:
                continue
            extracted = _extract_xmlish_content(content)
            if extracted:
                return extracted
    return ""


def _extract_task_instruction(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    task = ((result.get("config") or {}).get("task") or {}) if isinstance(result.get("config"), dict) else {}
    path = task.get("path") if isinstance(task, dict) else None
    if not isinstance(path, str) or not path:
        task_id = result.get("task_id")
        if isinstance(task_id, dict):
            path = task_id.get("path")
    if not isinstance(path, str) or not path:
        return ""
    instruction_path = Path(path) / "instruction.md"
    try:
        return instruction_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _extract_reasoning(agent_trace: Any, *, max_chars: int) -> str:
    if not isinstance(agent_trace, dict):
        return ""
    chunks: list[str] = []
    for step in agent_trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            chunks.append(f"Reasoning:\n{reasoning.strip()}")
        message = step.get("message")
        if isinstance(message, str) and message.strip() and message.strip() != "(tool use)":
            chunks.append(f"Message:\n{message.strip()}")
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            summaries = []
            for call in tool_calls[:8]:
                if isinstance(call, dict):
                    name = call.get("function_name") or call.get("name") or "tool"
                    summaries.append(str(name))
            if summaries:
                chunks.append("Tool calls: " + ", ".join(summaries))
        observations = [_compact_text(content, 1000) for content in _observation_contents(step)]
        if observations:
            chunks.append("Observations:\n" + "\n\n".join(observations[:4]))
    text = "\n\n".join(chunks).strip()
    return _truncate_middle(text, max_chars)


def _extract_answer(agent_trace: Any) -> str:
    if not isinstance(agent_trace, dict):
        return ""
    for step in reversed(agent_trace.get("steps") or []):
        if not isinstance(step, dict):
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip() and message.strip() != "(tool use)":
            return message.strip()
    return ""


def _observation_contents(step: dict[str, Any]) -> list[str]:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return []
    contents: list[str] = []
    for result in observation.get("results") or []:
        if isinstance(result, dict) and isinstance(result.get("content"), str):
            contents.append(result["content"])
    return contents


def _extract_xmlish_content(text: str) -> str:
    match = re.search(r"<content>\s*(.*?)\s*</content>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_ace_feedback(trajectory: Any, *, result: Any, feedback_mode: str) -> str:
    if feedback_mode not in {"reward_only", "public_feedback", "gt_labels"}:
        raise ValueError(f"Unsupported ACE feedback_mode: {feedback_mode}")
    parts = [
        f"success={bool(trajectory.success)}",
        f"reward={float(trajectory.reward):g}",
        f"score={float(trajectory.score):g}",
    ]
    if trajectory.rewards:
        parts.append("rewards=" + json.dumps(trajectory.rewards, sort_keys=True))
    if trajectory.error:
        parts.append("error=" + _compact_text(str(trajectory.error), 500))
    if feedback_mode == "public_feedback":
        verifier_summary = _public_verifier_summary(result)
        if verifier_summary:
            parts.append("verifier_summary=" + verifier_summary)
    return "; ".join(parts)


def _public_verifier_summary(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    exception_info = result.get("exception_info")
    if exception_info:
        return _compact_text(json.dumps(exception_info, sort_keys=True), 500)
    verifier_result = result.get("verifier_result")
    if isinstance(verifier_result, dict) and isinstance(verifier_result.get("rewards"), dict):
        return _compact_text("verifier_rewards=" + json.dumps(verifier_result["rewards"], sort_keys=True), 500)
    return ""


def _extract_ground_truth(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    verifier_result = result.get("verifier_result")
    if not isinstance(verifier_result, dict):
        return None
    value = verifier_result.get("ground_truth")
    return str(value) if value is not None else None


def _build_ace_context(trajectory: Any, *, result: Any, trial_dir: Path | None) -> str:
    source = ""
    if isinstance(result, dict):
        source = str(result.get("source") or "")
    parts = [
        f"task_id={trajectory.task_id}",
        f"view={trajectory.view_name}",
        f"mode={trajectory.mode}",
    ]
    if source:
        parts.append(f"source={source}")
    if trial_dir:
        parts.append(f"trial={trial_dir.name}")
    return "\n".join(parts)


def _trace_id(trajectory: Any) -> str:
    attempt = "" if trajectory.attempt_id is None else str(trajectory.attempt_id)
    return f"{trajectory.task_id}:{attempt}"


def _compact_text(text: str, max_chars: int) -> str:
    return _truncate_middle(" ".join(text.split()), max_chars)


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[truncated]...\n" + text[-(max_chars - half) :].lstrip()


