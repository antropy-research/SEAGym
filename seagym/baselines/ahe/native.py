from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_ahe_evolve(project_dir: Path) -> Any:
    evolve_path = project_dir / "evolve.py"
    project_path = str(project_dir)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    spec = importlib.util.spec_from_file_location("seagym_ahe_evolve", evolve_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AHE evolve module from {evolve_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_ahe_post_update_hooks(
    *,
    evolve: Any,
    exp_dir: Path,
    iteration: int,
    iteration_dir: Path,
    result: str,
) -> dict[str, Any]:
    """Run reference AHE post-evolve artifact hooks when available."""

    artifacts: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    save_summary = getattr(evolve, "save_evolve_summary", None)
    if callable(save_summary):
        try:
            save_summary(iteration_dir, iteration, result)
            summary_path = iteration_dir / "evolve" / "evolve_summary.md"
            if summary_path.exists():
                artifacts["evolve_summary"] = str(summary_path)
        except Exception as exc:  # pragma: no cover - defensive native hook wrapper.
            errors.append({"hook": "save_evolve_summary", "type": type(exc).__name__, "message": str(exc)})

    archive_manifest = getattr(evolve, "archive_change_manifest", None)
    if callable(archive_manifest):
        try:
            archive_manifest(exp_dir, iteration)
            manifest_path = iteration_dir / "evolve" / "change_manifest.json"
            if manifest_path.exists():
                artifacts["change_manifest"] = str(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive native hook wrapper.
            errors.append({"hook": "archive_change_manifest", "type": type(exc).__name__, "message": str(exc)})

    if errors:
        artifacts["errors"] = errors
    return artifacts
