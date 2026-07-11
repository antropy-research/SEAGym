from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_ahe_state_metadata(state_dir: Path) -> dict[str, Any]:
    metadata_path = state_dir / "baseline_state.json"
    if not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
