from __future__ import annotations

_ACE_RUNTIME_RUNNER = r'''
from pathlib import Path
import hashlib
import json
import sys

def extract_usage_cost(value):
    if not isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
        usage_fn = getattr(value, "usage", None)
        if callable(usage_fn):
            try:
                usage = usage_fn()
                total = getattr(usage, "total_tokens", 0) or 0
                if total > 0:
                    return {"total_tokens": float(total)}
            except Exception:
                pass
        for attr in ("raw", "output", "result"):
            child_cost = extract_usage_cost(getattr(value, attr, None))
            if child_cost:
                return child_cost
        object_dict = getattr(value, "__dict__", None)
        if isinstance(object_dict, dict):
            return extract_usage_cost(object_dict)
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens") or usage.get("totalTokens")
            if isinstance(total, (int, float)) and total > 0:
                return {"total_tokens": float(total)}
        total = 0.0
        for child in value.values():
            child_cost = extract_usage_cost(child)
            total += child_cost.get("total_tokens", 0.0)
        return {"total_tokens": total} if total > 0 else {}
    if isinstance(value, list):
        total = 0.0
        for child in value:
            child_cost = extract_usage_cost(child)
            total += child_cost.get("total_tokens", 0.0)
        return {"total_tokens": total} if total > 0 else {}
    return {}

records_path, state_dir_raw, update_dir_raw, project_dir_raw, model, epochs_raw, wait_raw, skillbook_filename, prompt_filename, update_prompt_variant, reflector_prompt_path_raw, skill_manager_prompt_path_raw, skill_manager_system_prompt_path_raw, update_mode, output_raw = sys.argv[1:16]
state_dir = Path(state_dir_raw)
update_dir = Path(update_dir_raw)
if project_dir_raw:
    sys.path.insert(0, project_dir_raw)
records = json.loads(Path(records_path).read_text(encoding="utf-8"))["records"]
from ace import ACELiteLLM  # type: ignore
from types import SimpleNamespace

def read_optional(path_raw):
    return Path(path_raw).read_text(encoding="utf-8") if path_raw else None

def compact_text(text, max_chars):
    text = " ".join(str(text).split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[truncated]...\n" + text[-(max_chars - half):].lstrip()

def batch_question_context(batch_records, max_chars=12000):
    parts = []
    for index, record in enumerate(batch_records, start=1):
        parts.append(
            f"[{index}] task_id={record.get('task_id')}\n"
            f"question={compact_text(record.get('question') or '', 600)}\n"
            f"context={compact_text(record.get('context') or '', 300)}\n"
            f"feedback={compact_text(record.get('feedback') or '', 300)}"
        )
    return compact_text("\n\n".join(parts), max_chars)

def batch_reflect_then_update(agent, batch_records, epochs, update_dir):
    results = []
    for epoch in range(1, epochs + 1):
        reflections = []
        trace_results = []
        injected_skill_ids = set()
        for index, record in enumerate(batch_records, start=1):
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
                    "reflection": repr(reflection),
                }
            )
        update_output = agent.skill_manager.update_skills(
            reflections=tuple(reflections),
            skillbook=agent.skillbook,
            question_context=batch_question_context(batch_records),
            progress=f"Epoch {epoch}/{epochs}, batch_size {len(batch_records)}",
            source=None,
            injected_skill_ids=tuple(sorted(injected_skill_ids)),
        )
        results.append(
            {
                "epoch": epoch,
                "num_traces": len(batch_records),
                "trace_results": trace_results,
                "skill_manager_output": repr(update_output),
            }
        )
        checkpoint_dir = update_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_dir / f"batch_checkpoint_{epoch:04d}.json"))
    return results

def custom_update_role_kwargs():
    reflector_prompt = read_optional(reflector_prompt_path_raw)
    skill_manager_prompt = read_optional(skill_manager_prompt_path_raw)
    skill_manager_system_prompt = read_optional(skill_manager_system_prompt_path_raw)
    if reflector_prompt is None and skill_manager_prompt is None and skill_manager_system_prompt is None:
        return {}
    from ace.implementations import Reflector, SkillManager  # type: ignore

    role_kwargs = {}
    if reflector_prompt is not None:
        role_kwargs["reflector"] = Reflector(model, prompt_template=reflector_prompt)
    if skill_manager_prompt is not None or skill_manager_system_prompt is not None:
        sm_kwargs = {}
        if skill_manager_prompt is not None:
            sm_kwargs["prompt_template"] = skill_manager_prompt
        if skill_manager_system_prompt is not None:
            sm_kwargs["system_prompt"] = skill_manager_system_prompt
        role_kwargs["skill_manager"] = SkillManager(model, **sm_kwargs)
    return role_kwargs

skillbook_path = state_dir / skillbook_filename
before_hash = hashlib.sha256(skillbook_path.read_bytes()).hexdigest() if skillbook_path.exists() else None
kwargs = {"checkpoint_dir": update_dir / "checkpoints"}
if skillbook_path.exists():
    kwargs["skillbook_path"] = str(skillbook_path)
kwargs.update(custom_update_role_kwargs())
agent = ACELiteLLM(model, **kwargs)
if update_mode == "native_trace_analyser":
    results = agent.learn_from_traces(records, epochs=int(epochs_raw), wait=wait_raw == "1")
elif update_mode == "batch_reflect_then_update":
    results = batch_reflect_then_update(agent, records, int(epochs_raw), update_dir)
else:
    raise ValueError(f"Unsupported ACE update_mode: {update_mode}")
agent.save(str(skillbook_path))
strategies = agent.get_strategies()
(update_dir / "strategies.md").write_text(str(strategies), encoding="utf-8")
prompt_path = state_dir / prompt_filename
prompt_path.write_text(str(strategies), encoding="utf-8")
(update_dir / "ace_results.json").write_text(json.dumps(results, default=repr, indent=2, sort_keys=True) + "\n", encoding="utf-8")
cost = extract_usage_cost(results)
result_errors = [repr(getattr(item, "error", None)) for item in results if getattr(item, "error", None) is not None]
all_results_failed = bool(results) and len(result_errors) == len(results)
try:
    skillbook_data = json.loads(skillbook_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    skillbook_data = {}
skills = skillbook_data.get("skills") if isinstance(skillbook_data, dict) else {}
num_skills = len(skills) if isinstance(skills, dict) else 0
after_hash = hashlib.sha256(skillbook_path.read_bytes()).hexdigest() if skillbook_path.exists() else None
changed = before_hash != after_hash and (bool(str(strategies).strip()) or num_skills > 0)
Path(output_raw).write_text(json.dumps({
    "changed": changed,
    "status": "error" if all_results_failed else ("updated" if changed else "unchanged"),
    "metrics": {
        "num_trajectories": len(records),
        "num_results": len(results),
        "num_result_errors": len(result_errors),
        "num_skills": num_skills,
    },
    "error": (
        {
            "type": "ACEAllSamplesFailed",
            "message": "ACE native update produced only failed SampleResult objects.",
            "samples": result_errors[:5],
        }
        if all_results_failed
        else None
    ),
    "cost": cost,
    "cost_source": "ace_results_usage" if cost else None,
    "artifacts": {"skillbook_path": str(skillbook_path), "update_dir": str(update_dir)},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'''
