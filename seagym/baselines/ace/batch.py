from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..adapter_utils import jsonable
from .traces import _compact_text, _truncate_middle


def _batch_reflect_then_update(agent: Any, records: list[dict[str, Any]], *, epochs: int, update_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        reflections: list[Any] = []
        trace_results: list[dict[str, Any]] = []
        injected_skill_ids: set[str] = set()
        for index, record in enumerate(records, start=1):
            skill_ids = record.get("skill_ids")
            if isinstance(skill_ids, list):
                injected_skill_ids.update(str(skill_id) for skill_id in skill_ids)
            reflection = agent.reflector.reflect(
                question=str(record.get("question") or ""),
                agent_output=SimpleNamespace(
                    reasoning=str(record.get("reasoning") or ""),
                    final_answer=str(record.get("answer") or ""),
                ),
                skillbook=agent.skillbook,
                ground_truth=record.get("ground_truth"),
                feedback=record.get("feedback"),
                injected_skill_ids=tuple(str(skill_id) for skill_id in skill_ids) if isinstance(skill_ids, list) else (),
                mode=record.get("mode", "online"),
            )
            reflections.append(reflection)
            trace_results.append(
                {
                    "epoch": epoch,
                    "index": index,
                    "trace_id": record.get("trace_id"),
                    "task_id": record.get("task_id"),
                    "reflection": jsonable(reflection),
                }
            )
        update_output = agent.skill_manager.update_skills(
            reflections=tuple(reflections),
            skillbook=agent.skillbook,
            question_context=_batch_question_context(records),
            progress=f"Epoch {epoch}/{epochs}, batch_size {len(records)}",
            source=None,
            injected_skill_ids=tuple(sorted(injected_skill_ids)),
        )
        results.append(
            {
                "epoch": epoch,
                "num_traces": len(records),
                "trace_results": trace_results,
                "skill_manager_output": jsonable(update_output),
            }
        )
        checkpoint_dir = update_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_dir / f"batch_checkpoint_{epoch:04d}.json"))
    return results


def _batch_question_context(records: list[dict[str, Any]], *, max_chars: int = 12000) -> str:
    parts = []
    for index, record in enumerate(records, start=1):
        question = _compact_text(str(record.get("question") or ""), 600)
        context = _compact_text(str(record.get("context") or ""), 300)
        feedback = _compact_text(str(record.get("feedback") or ""), 300)
        parts.append(f"[{index}] task_id={record.get('task_id')}\nquestion={question}\ncontext={context}\nfeedback={feedback}")
    return _truncate_middle("\n\n".join(parts), max_chars)


