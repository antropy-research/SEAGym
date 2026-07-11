from __future__ import annotations

"""Command-line entrypoint for SEAGym training and inspection workflows."""

import argparse
import json
from pathlib import Path

from seagym.config import load_experiment_context
from seagym.logging import ArtifactLayout, write_run_reports
from seagym.paths import REPO_ROOT, resolve_portable_path
from seagym.runtime import RuntimeCheckOptions, inspect_runtime, load_env_file
from seagym.trainers import SEAGymTrainer
from seagym.trainers.checkpoint import resolve_checkpoint
from seagym.trainers.run import RunOptions
from seagym.utils import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SEAGym training, evaluation, and inspection workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_inspect_parser(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


def _add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Train a SEAGym experiment.")
    parser.add_argument("config")
    _add_run_dir_args(parser)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the run directory.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(func=_train)


def _add_eval_parser(subparsers) -> None:
    parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint on final views.")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    _add_run_dir_args(parser)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(func=_eval)


def _add_inspect_parser(subparsers) -> None:
    parser = subparsers.add_parser("inspect", help="Inspect config, run artifacts, or runtime environment.")
    inspect_subparsers = parser.add_subparsers(dest="inspect_command", required=True)
    config = inspect_subparsers.add_parser("config", help="Load and validate a config.")
    config.add_argument("config")
    config.set_defaults(func=_inspect_config)
    run = inspect_subparsers.add_parser("run", help="Summarize a run directory.")
    run.add_argument("run_dir")
    run.set_defaults(func=_inspect_run)
    env = inspect_subparsers.add_parser("env", help="Print basic runtime paths.")
    env.set_defaults(func=_inspect_env)
    runtime = inspect_subparsers.add_parser("runtime", help="Inspect runtime dependencies for a config.")
    runtime.add_argument("config")
    runtime.add_argument("--run-dir", default=None)
    runtime.add_argument("--env-file", default=".env")
    runtime.add_argument("--load-env", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--host-probe-url", action="append", default=[])
    runtime.add_argument("--container-probe-url", action="append", default=[])
    runtime.add_argument("--container-probe-image", default="python:3.12-alpine")
    runtime.add_argument("--canary", action="store_true")
    runtime.add_argument("--canary-task-limit", type=int, default=1)
    runtime.add_argument("--canary-agent", default="oracle")
    runtime.add_argument("--timeout-seconds", type=float, default=20.0)
    runtime.set_defaults(func=_inspect_runtime)


def _add_run_dir_args(parser) -> None:
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)


def _train(args) -> None:
    _load_default_env()
    checkpoint = args.resume_from_checkpoint
    if args.resume:
        if checkpoint is not None:
            raise SystemExit("--resume and --resume-from-checkpoint are mutually exclusive")
        checkpoint = "latest"
    trainer = SEAGymTrainer.from_config(
        _user_path(args.config),
        run_options=RunOptions(
            output_dir=_user_path(args.output_dir),
            run_dir=_user_path(args.run_dir),
            run_name=args.run_name,
            resume=args.resume,
            resume_from_checkpoint=checkpoint,
            overwrite=args.overwrite,
        ),
    )
    run_dir = trainer.fit(resume_from_checkpoint=checkpoint)
    print(f"Wrote run artifacts to {run_dir}")


def _eval(args) -> None:
    _load_default_env()
    trainer = SEAGymTrainer.from_config(
        _user_path(args.config),
        run_options=RunOptions(
            output_dir=_user_path(args.output_dir),
            run_dir=_user_path(args.run_dir),
            run_name=args.run_name,
            overwrite=args.overwrite,
        ),
    )
    engine = trainer.prepare(reset_run_dir=not (args.checkpoint == "latest" and args.run_dir))
    checkpoint_dir = resolve_checkpoint(trainer.layout.run_dir, _checkpoint_value(args.checkpoint))
    load_ref = engine.load_checkpoint(checkpoint_dir)
    point = engine.record_evaluation_point(
        point_type="checkpoint_eval",
        train_batch_index=0,
        num_train_tasks_seen=0,
    )
    evaluations = {}
    for view_name, task_ids in engine.final_views().items():
        results = engine.run_tasks(
            task_ids,
            view_name=view_name,
            mode="checkpoint_eval",
            evaluation_point=point,
            agent_checkpoint_id=checkpoint_dir.name,
            baseline_role="checkpoint",
        )
        score = 0.0 if not results else sum(result.score for result in results) / len(results)
        evaluations[view_name] = {
            "view_ref": view_name,
            "score": score,
            "num_tasks": len(task_ids),
            "checkpoint": str(checkpoint_dir),
        }
    engine.write_evaluation_point(
        point,
        evaluations=evaluations,
        refs={"checkpoint_load": load_ref, "metric_inputs": str(engine.metric_input_path)},
    )
    trainer.compute_metrics()
    write_run_reports(trainer.layout.run_dir)
    print(f"Wrote eval artifacts to {trainer.layout.run_dir}")


def _inspect_config(args) -> None:
    _load_default_env()
    context = load_experiment_context(_user_path(args.config))
    print(
        json.dumps(
            {
                "experiment_id": context.config.experiment_id,
                "run_dir": str(context.config.run_dir),
                "tasks": len(context.task_index.tasks),
                "train": len(context.split.train),
                "val": len(context.split.val),
                "test": len(context.split.test),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _inspect_run(args) -> None:
    _load_default_env()
    layout = ArtifactLayout.from_run_dir(_user_path(args.run_dir))
    summary = {
        "run_dir": str(layout.run_dir),
        "metrics": layout.metrics_path.exists(),
        "task_results": len(read_jsonl(layout.task_results_path)) if layout.task_results_path.exists() else 0,
        "evaluation_points": len(read_jsonl(layout.evaluation_points_path)) if layout.evaluation_points_path.exists() else 0,
        "checkpoints_dir": str(layout.checkpoints_dir),
        "latest_checkpoint": str(layout.checkpoints_dir / "latest_checkpoint.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _inspect_env(args) -> None:
    del args
    _load_default_env()
    print(json.dumps({"repo_root": str(REPO_ROOT), "cwd": str(Path.cwd())}, indent=2, sort_keys=True))


def _inspect_runtime(args) -> None:
    report = inspect_runtime(
        RuntimeCheckOptions(
            config_path=_required_path(args.config),
            run_dir=_user_path(args.run_dir),
            env_file=_required_path(args.env_file),
            load_env=args.load_env,
            host_probe_urls=list(args.host_probe_url or []),
            container_probe_urls=list(args.container_probe_url or []),
            container_probe_image=args.container_probe_image,
            canary=args.canary,
            canary_task_limit=args.canary_task_limit,
            canary_agent=args.canary_agent,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def _user_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_portable_path(value, base_dir=Path.cwd(), repo_root=REPO_ROOT)


def _required_path(value: str | Path) -> Path:
    resolved = _user_path(value)
    assert resolved is not None
    return resolved


def _checkpoint_value(value: str) -> str | Path:
    if value == "latest":
        return value
    path = Path(value)
    if path.is_absolute() or len(path.parts) > 1 or value.startswith(("repo://", "data://", "results://")):
        resolved = _user_path(value)
        assert resolved is not None
        return resolved
    return value


def _load_default_env() -> None:
    load_env_file(Path.cwd() / ".env")


if __name__ == "__main__":
    main()
