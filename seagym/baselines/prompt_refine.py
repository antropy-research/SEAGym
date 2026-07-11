from __future__ import annotations

"""Prompt-refinement baseline."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
from typing import Any

from seagym.models import ChatModelClient, ModelConfig, build_chat_model_client

from .base import BaseBaseline, BaselineState, Checkpoint, UpdateResult, _resolve_checkpoint_ref
from .data import TrajectoryBatch


DEFAULT_PROMPT = """Use the task instruction carefully.
Before acting, identify the expected artifact and verifier-visible success condition.
Prefer simple, direct edits or commands over broad exploration.
"""


@dataclass
class PromptRefineBaseline(BaseBaseline):
    refiner_model: ModelConfig | None = None
    model_client: ChatModelClient | None = None
    prompt: str = DEFAULT_PROMPT
    max_batch_records: int = 12
    fail_on_refine_error: bool = False

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        models: dict[str, Any],
        state_dir: Path,
        run_dir: Path,
        base_dir: Path | None,
    ) -> "PromptRefineBaseline":
        del run_dir, base_dir
        update_ref = str(config.get("update_model_ref", "update_model"))
        return cls(
            baseline_id=name,
            state_dir=state_dir,
            refiner_model=_model_config(models.get(update_ref)),
            max_batch_records=int(config.get("max_batch_records", 12)),
            fail_on_refine_error=bool(config.get("fail_on_error", False)),
        )

    @property
    def prompt_template_path(self) -> Path:
        return self.state_dir / "prompt_template.md"

    @property
    def state_path(self) -> Path:
        return self.state_dir / "prompt_state.json"

    @property
    def history_path(self) -> Path:
        return self.state_dir / "prompt_updates.jsonl"

    def initialize(self, run_dir: Path) -> BaselineState:
        del run_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self._read_state()
        else:
            self._write_state()
        return BaselineState(
            state_dir=self.state_dir,
            metadata={
                "baseline_id": self.baseline_id,
                "prompt_template_path": str(self.prompt_template_path),
            },
        )

    def update(self, trajectories: TrajectoryBatch, state: BaselineState) -> UpdateResult:
        del state
        self.update_index += 1
        before = self.prompt
        records = [trajectory.to_dict() for trajectory in trajectories.trajectories[: self.max_batch_records]]
        try:
            refined = self._refine_prompt(_build_refinement_prompt(self.prompt, records))
        except Exception as exc:
            if self.fail_on_refine_error:
                raise
            summary = {
                "type": "llm_prompt_refine_update",
                "status": "failed",
                "changed": False,
                "num_records": len(records),
                "error": str(exc),
            }
            self._append_history(summary)
            return UpdateResult(self.update_index, False, "failed", logs=summary)
        cleaned = _clean_prompt(refined)
        changed = bool(cleaned and cleaned != before)
        if changed:
            self.prompt = cleaned
            self._write_state()
        summary = {
            "type": "llm_prompt_refine_update",
            "status": "updated" if changed else "unchanged",
            "changed": changed,
            "num_records": len(records),
            "prompt_chars_before": len(before),
            "prompt_chars_after": len(self.prompt),
        }
        self._append_history(summary)
        return UpdateResult(self.update_index, changed, summary["status"], logs=summary)

    def save_checkpoint(self, state: BaselineState, path: Path) -> Checkpoint:
        del state
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        destination = path / "prompt_state"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.state_dir, destination)
        manifest = {
            "type": "prompt_refine_checkpoint",
            "baseline_id": self.baseline_id,
            "state_ref": destination.name,
            "prompt_template_path": str(destination / "prompt_template.md"),
        }
        (path / "checkpoint.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return Checkpoint(checkpoint_dir=path, state_ref=str(destination), metadata=manifest)

    def load_checkpoint(self, checkpoint: Checkpoint) -> BaselineState:
        manifest_path = checkpoint.checkpoint_dir / "checkpoint.json"
        if not manifest_path.exists():
            return BaselineState(self.state_dir, {"loaded": False, "reason": f"checkpoint manifest not found: {manifest_path}"})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest.get("baseline"), dict):
            manifest = manifest["baseline"]
        source = manifest.get("state_ref")
        if source:
            source_path = _resolve_checkpoint_ref(Path(str(source)), checkpoint.checkpoint_dir)
            if self.state_dir.exists():
                shutil.rmtree(self.state_dir)
            shutil.copytree(source_path, self.state_dir)
            self._read_state()
        return BaselineState(
            self.state_dir,
            {
                "loaded": True,
                "manifest": manifest,
                "prompt_template_path": str(self.prompt_template_path),
            },
        )

    def _read_state(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.prompt = str(data.get("prompt", self.prompt))
        self._write_prompt_template()

    def _write_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "baseline_id": self.baseline_id,
            "prompt": self.prompt,
            "prompt_template_path": str(self.prompt_template_path),
        }
        self.state_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_prompt_template()

    def _write_prompt_template(self) -> None:
        self.prompt_template_path.write_text(
            "# Learned SEAGym Prompt\n\n"
            f"{self.prompt.strip()}\n\n"
            "# Task Instruction\n\n"
            "{{ instruction }}\n",
            encoding="utf-8",
        )

    def _append_history(self, summary: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")

    def _refine_prompt(self, prompt_input: str) -> str:
        if self.refiner_model is None:
            raise ValueError("PromptRefineBaseline requires an update_model binding")
        client = self.model_client or build_chat_model_client(self.refiner_model)
        return client.complete(
            system="You refine concise operating prompts for LLM agents.",
            user=prompt_input,
        )


def _model_config(raw: Any) -> ModelConfig | None:
    if not isinstance(raw, dict):
        return None
    return ModelConfig(
        name=str(raw.get("model", "")),
        provider=str(raw.get("provider", "openai_compatible")),
        api_base=None if raw.get("api_base") in (None, "") else str(raw.get("api_base")),
        api_key_env=None if raw.get("api_key_env") in (None, "") else str(raw.get("api_key_env")),
        reasoning_effort=None if raw.get("reasoning_effort") in (None, "") else str(raw.get("reasoning_effort")),
        extra_body=dict(raw.get("extra_body") or {}),
    )


def _build_refinement_prompt(prompt: str, records: list[dict[str, Any]]) -> str:
    compact = [
        {
            "task_id": row.get("task_id"),
            "score": row.get("score"),
            "success": row.get("success"),
            "error": row.get("error"),
            "rewards": row.get("rewards"),
        }
        for row in records
    ]
    return (
        "You are maintaining a prompt for an LLM coding/tool-use agent in SEAGym.\n"
        "Revise the prompt to improve future train-batch task performance while avoiding overfitting to task IDs.\n"
        "Return only the next prompt text. Do not include markdown fences or explanations.\n\n"
        f"Current prompt:\n{prompt.strip()}\n\n"
        "Recent trajectories:\n"
        f"{json.dumps(compact, indent=2, sort_keys=True)}\n"
    )


def _clean_prompt(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("text"):
            cleaned = cleaned[4:].strip()
    return cleaned
