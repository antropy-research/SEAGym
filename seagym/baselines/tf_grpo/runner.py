from __future__ import annotations

_TF_GRPO_RUNTIME_RUNNER = r'''
from pathlib import Path
import json
import importlib
import sys
import threading
import time

def extract_usage_cost(value):
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

class Meter:
    def __init__(self):
        self.lock = threading.Lock()
        self.input_tokens = 0.0
        self.output_tokens = 0.0
        self.cache_tokens = 0.0
        self.total_tokens = 0.0

    def add_usage(self, usage):
        cost = openai_usage_cost(usage)
        if not cost:
            return
        with self.lock:
            self.input_tokens += cost.get("input_tokens", 0.0)
            self.output_tokens += cost.get("output_tokens", 0.0)
            self.cache_tokens += cost.get("cache_tokens", 0.0)
            self.total_tokens += cost.get("total_tokens", 0.0)

    def cost(self):
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

def openai_usage_cost(usage):
    if usage is None:
        return {}
    if isinstance(usage, dict):
        input_tokens = usage_number(usage, "prompt_tokens", "input_tokens")
        output_tokens = usage_number(usage, "completion_tokens", "output_tokens")
        total_tokens = usage_number(usage, "total_tokens")
        cache_tokens = usage_number(usage.get("prompt_tokens_details") or {}, "cached_tokens")
    else:
        input_tokens = usage_attr_number(usage, "prompt_tokens", "input_tokens")
        output_tokens = usage_attr_number(usage, "completion_tokens", "output_tokens")
        total_tokens = usage_attr_number(usage, "total_tokens")
        details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
        cache_tokens = usage_attr_number(details, "cached_tokens") if details is not None else 0.0
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

def usage_number(value, *keys):
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return 0.0

def usage_attr_number(value, *keys):
    for key in keys:
        raw = getattr(value, key, None)
        if isinstance(raw, (int, float)):
            return float(raw)
    return 0.0

def patch_experience_updater_llm(experience_module, meter):
    original_llm = getattr(experience_module, "LLM", None)
    if original_llm is None:
        return getattr(experience_module, "ExperienceUpdater")

    class MeteredLLM(original_llm):
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

def read_json_path(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

def record_result_path(record):
    refs = record.get("refs")
    if not isinstance(refs, dict):
        return None
    raw = refs.get("result_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None

def compact_json(value, max_chars):
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"

def format_harbor_trajectory(trace):
    if not isinstance(trace, dict):
        return json.dumps(trace, indent=2, sort_keys=True)
    lines = []
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
            lines.append("Observation: " + compact_json(observation, 6000))
    final_metrics = trace.get("final_metrics")
    if final_metrics:
        lines.append("\nFinal metrics: " + compact_json(final_metrics, 2000))
    return "\n".join(lines).strip() or json.dumps(trace, indent=2, sort_keys=True)

def extract_problem(record):
    result_path = record_result_path(record)
    if result_path is not None:
        data = read_json_path(result_path)
        task_path = (((data or {}).get("config") or {}).get("task") or {}).get("path")
        if task_path:
            instruction = Path(str(task_path)) / "instruction.md"
            if instruction.exists():
                return instruction.read_text(encoding="utf-8", errors="replace").strip()
    return str(record.get("instruction") or record.get("problem") or "")

def extract_trajectory_text(record):
    result_path = record_result_path(record)
    flags = {"has_harbor_trajectory": False, "has_response": False, "used_metadata_fallback": False}
    parts = []
    if result_path is not None:
        trial_dir = result_path.parent
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        response_path = trial_dir / "agent" / "response.txt"
        trace = read_json_path(trajectory_path)
        if trace is not None:
            flags["has_harbor_trajectory"] = True
            parts.append(format_harbor_trajectory(trace))
        if response_path.exists():
            flags["has_response"] = True
            parts.append("Final response file:\n" + response_path.read_text(encoding="utf-8", errors="replace").strip())
    if not parts:
        flags["used_metadata_fallback"] = True
        parts.append("SEAGym normalized result metadata:\n" + json.dumps(record, indent=2, sort_keys=True))
    return "\n\n".join(part for part in parts if part.strip()), flags

def rollout_diagnostics(rollouts):
    by_problem = {}
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

def filter_update_rollouts(rollouts, skip_metadata_fallback):
    if not skip_metadata_fallback:
        return list(rollouts)
    return [
        rollout
        for rollout in rollouts
        if (rollout.get("seagym_artifacts") or {}).get("has_harbor_trajectory")
    ]

META_SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system may be provided with learned experiences, then attempts a task. Summarize the trajectory step-by-step.

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

META_SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = """An agent system may be provided with learned experiences, then attempts a task. Summarize the trajectory step-by-step.

For each step, extract the action, tool or file operation, observation, reasoning, outcome, and any experience that appears to be used. Identify likely detours, errors, missing checks, or effective decisions from the trajectory and verifier outcome.

<trajectory>
{trajectory}
</trajectory>

Return only a concise step-by-step trajectory summary."""

META_SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the same task multiple times with both successful and wrong solutions. Review these attempts and extract generalizable experiences.

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

META_SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the same task multiple times. Review these attempts and extract generalizable experiences.

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

META_BATCH_EXPERIENCE_UPDATE_TEMPLATE = """An agent system has proposed updates to a set of learned experiences. Produce a concise final experience revision plan.

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

TF_GRPO_PROMPT_NAMES = (
    "SINGLE_ROLLOUT_SUMMARY_TEMPLATE",
    "SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE",
    "SINGLE_QUERY_CRITIQUE_TEMPLATE",
    "SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE",
    "BATCH_EXPERIENCE_UPDATE_TEMPLATE",
)

def snapshot_native_tf_grpo_prompts(experience_module, prompts_module):
    if hasattr(prompts_module, "_seagym_native_prompt_snapshot"):
        return
    snapshot = {"prompts": {}, "experience": {}}
    for name in TF_GRPO_PROMPT_NAMES:
        if hasattr(prompts_module, name):
            snapshot["prompts"][name] = getattr(prompts_module, name)
        if hasattr(experience_module, name):
            snapshot["experience"][name] = getattr(experience_module, name)
    setattr(prompts_module, "_seagym_native_prompt_snapshot", snapshot)

def restore_native_tf_grpo_prompts(experience_module, prompts_module):
    snapshot = getattr(prompts_module, "_seagym_native_prompt_snapshot", None)
    if not isinstance(snapshot, dict):
        return
    for name, value in (snapshot.get("prompts") or {}).items():
        setattr(prompts_module, name, value)
    for name, value in (snapshot.get("experience") or {}).items():
        setattr(experience_module, name, value)

def apply_update_prompt_profile(experience_module, profile):
    prompt_module_name = experience_module.__name__.rsplit(".", 1)[0] + ".prompts"
    try:
        prompts = importlib.import_module(prompt_module_name)
    except ImportError:
        if profile in ("", "native", "default"):
            return
        raise
    snapshot_native_tf_grpo_prompts(experience_module, prompts)
    if profile in ("", "native", "default"):
        restore_native_tf_grpo_prompts(experience_module, prompts)
        return
    if profile != "meta":
        raise ValueError(f"Unsupported TF-GRPO update_prompt_profile: {profile}")
    replacements = {
        "SINGLE_ROLLOUT_SUMMARY_TEMPLATE": META_SINGLE_ROLLOUT_SUMMARY_TEMPLATE,
        "SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE": META_SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE,
        "SINGLE_QUERY_CRITIQUE_TEMPLATE": META_SINGLE_QUERY_CRITIQUE_TEMPLATE,
        "SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE": META_SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE,
        "BATCH_EXPERIENCE_UPDATE_TEMPLATE": META_BATCH_EXPERIENCE_UPDATE_TEMPLATE,
    }
    for name, value in replacements.items():
        if hasattr(prompts, name):
            setattr(prompts, name, value)
        if hasattr(experience_module, name):
            setattr(experience_module, name, value)

def patch_tf_grpo_dotenv_lookup():
    try:
        dotenv = importlib.import_module("dotenv")
        dotenv_main = importlib.import_module("dotenv.main")
    except ImportError:
        return

    def empty_dotenv(*args, **kwargs):
        return ""

    setattr(dotenv, "find_dotenv", empty_dotenv)
    setattr(dotenv_main, "find_dotenv", empty_dotenv)

(
    records_path,
    state_dir_raw,
    update_dir_raw,
    project_dir_raw,
    domain,
    experiences_filename,
    prompt_filename,
    max_workers_raw,
    given_ground_truth_raw,
    only_partial_correct_raw,
    update_prompt_profile_raw,
    skip_metadata_fallback_raw,
    output_raw,
) = sys.argv[1:14]
state_dir = Path(state_dir_raw)
update_dir = Path(update_dir_raw)
if project_dir_raw:
    sys.path.insert(0, project_dir_raw)
records = json.loads(Path(records_path).read_text(encoding="utf-8"))["records"]
experience_path = state_dir / experiences_filename
experiences = json.loads(experience_path.read_text(encoding="utf-8")) if experience_path.exists() else {}

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)

def to_rollout(record):
    task_id = str(record.get("task_id", "unknown-task"))
    reward = float(record.get("reward", record.get("score", 0.0)) or 0.0)
    problem = extract_problem(record) or task_id
    trajectory_text, artifact_flags = extract_trajectory_text(record)
    return {
        "problem": problem,
        "groundtruth": "",
        "reward": 1 if reward > 0 else 0,
        "score": reward,
        "task_id": task_id,
        "seagym_artifacts": artifact_flags,
        "trajectories": [{"trajectory": trajectory_text}],
    }

rollouts = [to_rollout(record) for record in records]
diagnostics = rollout_diagnostics(rollouts)
update_rollouts = filter_update_rollouts(rollouts, skip_metadata_fallback_raw == "1")
skipped_rollouts = [rollout for rollout in rollouts if rollout not in update_rollouts]
diagnostics["skipped_metadata_fallback_records"] = sum(
    1 for rollout in skipped_rollouts if (rollout.get("seagym_artifacts") or {}).get("used_metadata_fallback")
)
diagnostics["skipped_non_trajectory_records"] = sum(
    1
    for rollout in skipped_rollouts
    if not (rollout.get("seagym_artifacts") or {}).get("used_metadata_fallback")
)
meter = Meter()
patch_tf_grpo_dotenv_lookup()
if rollouts and not update_rollouts:
    prompt_path = state_dir / prompt_filename
    if experiences:
        lines = ["Use the following learned experiences when solving future tasks:"]
        for key, value in experiences.items():
            lines.append(f"[{key}] {value}")
        prompt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        prompt_path.write_text("", encoding="utf-8")
    Path(output_raw).write_text(json.dumps({
        "changed": False,
        "status": "unchanged",
        "metrics": {"num_trajectories": len(records), "num_experiences": len(experiences), **diagnostics},
        "cost": {},
        "cost_source": None,
        "artifacts": {"experience_path": str(experience_path), "update_dir": str(update_dir)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.exit(0)
if domain == "math":
    experience_module = importlib.import_module("training_free_grpo.math.experience")
    apply_update_prompt_profile(experience_module, update_prompt_profile_raw)
    ExperienceUpdater = patch_experience_updater_llm(experience_module, meter)
    new_experiences = ExperienceUpdater().run(
        rollouts=update_rollouts,
        experiences=experiences,
        save_dir=str(update_dir),
        max_workers=int(max_workers_raw),
        given_ground_truth=given_ground_truth_raw == "1",
        only_partial_correct=only_partial_correct_raw == "1",
    )
elif domain == "web":
    experience_module = importlib.import_module("training_free_grpo.web.experience")
    apply_update_prompt_profile(experience_module, update_prompt_profile_raw)
    ExperienceUpdater = patch_experience_updater_llm(experience_module, meter)
    new_experiences = ExperienceUpdater().run(
        rollouts=update_rollouts,
        experiences=experiences,
        save_dir=str(update_dir),
        max_workers=int(max_workers_raw),
        given_ground_truth=given_ground_truth_raw == "1",
    )
else:
    raise ValueError(f"Unsupported TF-GRPO domain: {domain}")

changed = canonical_json(new_experiences) != canonical_json(experiences)
experience_path.write_text(json.dumps(new_experiences, indent=2, sort_keys=True) + "\n", encoding="utf-8")
prompt_path = state_dir / prompt_filename
if new_experiences:
    lines = ["Use the following learned experiences when solving future tasks:"]
    for key, value in new_experiences.items():
        lines.append(f"[{key}] {value}")
    prompt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
else:
    prompt_path.write_text("", encoding="utf-8")
cost = meter.cost() or extract_usage_cost(new_experiences)
Path(output_raw).write_text(json.dumps({
    "changed": changed,
    "status": "updated" if changed else "unchanged",
    "metrics": {"num_trajectories": len(records), "num_experiences": len(new_experiences), **diagnostics},
    "cost": cost,
    "cost_source": "tf_grpo_llm_usage" if meter.cost() else ("tf_grpo_update_outputs" if cost else None),
    "artifacts": {"experience_path": str(experience_path), "update_dir": str(update_dir)},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'''
