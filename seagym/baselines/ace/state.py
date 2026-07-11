from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _skillbook_num_skills(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    skills = data.get("skills") if isinstance(data, dict) else None
    return len(skills) if isinstance(skills, dict) else 0


def _skillbook_has_entries(path: Path) -> bool:
    return _skillbook_num_skills(path) > 0


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_state_metadata(state_dir: Path, metadata: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in metadata.items() if key != "manifest"}
    (state_dir / "baseline_state.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


