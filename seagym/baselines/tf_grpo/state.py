from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_state_metadata(state_dir: Path, metadata: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in metadata.items() if key != "manifest"}
    (state_dir / "baseline_state.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
