from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_tf_grpo_rollout(record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record.get("task_id", "unknown-task"))
    reward = float(record.get("reward", record.get("score", 0.0)) or 0.0)
    problem = _extract_problem(record) or task_id
    trajectory_text, artifact_flags = _extract_trajectory_text(record)
    return {
        "problem": problem,
        "groundtruth": "",
        "reward": 1 if reward > 0 else 0,
        "score": reward,
        "task_id": task_id,
        "seagym_artifacts": artifact_flags,
        "trajectories": [
            {
                "trajectory": trajectory_text,
            }
        ],
    }


def _extract_problem(record: dict[str, Any]) -> str:
    result_path = _record_result_path(record)
    if result_path is not None:
        data = _read_json_path(result_path)
        task_path = (((data or {}).get("config") or {}).get("task") or {}).get("path")
        if task_path:
            instruction = Path(str(task_path)) / "instruction.md"
            if instruction.exists():
                return instruction.read_text(encoding="utf-8", errors="replace").strip()
    return str(record.get("instruction") or record.get("problem") or "")


def _extract_trajectory_text(record: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    result_path = _record_result_path(record)
    flags = {"has_harbor_trajectory": False, "has_response": False, "used_metadata_fallback": False}
    parts: list[str] = []
    if result_path is not None:
        trial_dir = result_path.parent
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        response_path = trial_dir / "agent" / "response.txt"
        trace = _read_json_path(trajectory_path)
        if trace is not None:
            flags["has_harbor_trajectory"] = True
            parts.append(_format_harbor_trajectory(trace))
        if response_path.exists():
            flags["has_response"] = True
            parts.append("Final response file:\n" + response_path.read_text(encoding="utf-8", errors="replace").strip())
    if not parts:
        flags["used_metadata_fallback"] = True
        parts.append("SEAGym normalized result metadata:\n" + json.dumps(record, indent=2, sort_keys=True))
    return "\n\n".join(part for part in parts if part.strip()), flags


def _format_harbor_trajectory(trace: Any) -> str:
    if not isinstance(trace, dict):
        return json.dumps(trace, indent=2, sort_keys=True)
    lines: list[str] = []
    agent = trace.get("agent") or {}
    if isinstance(agent, dict):
        lines.append(f"Agent: {agent.get('name', 'unknown')} model={agent.get('model_name', 'unknown')}")
    for step in trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id", "?")
        source = step.get("source", "unknown")
        lines.append(f"\nStep {step_id} [{source}]")
        reasoning = step.get("reasoning_content")
        if reasoning:
            lines.append(f"Reasoning: {reasoning}")
        message = step.get("message")
        if message and message != "(tool use)":
            lines.append(f"Message: {message}")
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                lines.append(
                    "Tool call: "
                    + str(call.get("function_name") or call.get("name") or "unknown")
                    + " args="
                    + json.dumps(call.get("arguments") or {}, sort_keys=True)
                )
        observation = step.get("observation")
        if observation:
            lines.append("Observation: " + _compact_json(observation, max_chars=6000))
    final_metrics = trace.get("final_metrics")
    if final_metrics:
        lines.append("\nFinal metrics: " + _compact_json(final_metrics, max_chars=2000))
    return "\n".join(lines).strip() or json.dumps(trace, indent=2, sort_keys=True)


def _record_result_path(record: dict[str, Any]) -> Path | None:
    refs = record.get("refs")
    if not isinstance(refs, dict):
        return None
    raw = refs.get("result_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def _read_json_path(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _compact_json(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _rollout_diagnostics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    by_problem: dict[str, list[int]] = {}
    has_real = 0
    fallback = 0
    for rollout in rollouts:
        by_problem.setdefault(str(rollout.get("task_id") or rollout.get("problem")), []).append(int(rollout.get("reward", 0)))
        artifacts = rollout.get("seagym_artifacts") or {}
        if artifacts.get("has_harbor_trajectory"):
            has_real += 1
        if artifacts.get("used_metadata_fallback"):
            fallback += 1
    mixed = sum(1 for rewards in by_problem.values() if min(rewards) < max(rewards))
    return {
        "num_rollout_groups": len(by_problem),
        "mixed_reward_groups": mixed,
        "real_trajectory_records": has_real,
        "metadata_fallback_records": fallback,
    }


def _filter_update_rollouts(rollouts: list[dict[str, Any]], *, skip_metadata_fallback: bool) -> list[dict[str, Any]]:
    if not skip_metadata_fallback:
        return list(rollouts)
    return [
        rollout
        for rollout in rollouts
        if (rollout.get("seagym_artifacts") or {}).get("has_harbor_trajectory")
    ]


