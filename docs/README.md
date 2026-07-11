# SEAGym Documentation

This directory contains release-facing documentation for installing, running,
and extending SEAGym. Internal development plans, manuscript notes, experiment
schedules, and local result analysis are intentionally excluded from the
release branch.

## Start Here

1. `getting_started.md`: installation, deterministic example, and common CLI
   commands.
2. `concepts.md`: SEAGym lifecycle, splits, evaluation points, and passive
   evaluation.
3. `configuration.md`: JSON config structure, schedules, dataloader settings,
   class paths, outputs, and portable path anchors.
4. `data.md`: task indexes, split manifests, and batch plans.
5. `training.md`: standard training lifecycle, checkpoints, resume, run
   directories, and Python usage.
6. `evaluation.md`: checkpoint evaluation, views, records, scoring, and
   reports.
7. `metrics.md`: default metrics, update-validation labels, replay diagnostics,
   and custom metric hooks.
8. `harbor.md`: Harbor submodule setup, runtime environment, E2B, and Harbor
   integration boundary.
9. `extending.md`: how to add datasets, rollout agents, baselines, metrics, and
   runtime checks.
10. `api_reference.md`: main public imports and CLI entrypoints.

## Architecture Notes

`decisions.md` records the release-level design decisions that define public
SEAGym boundaries. It is not required reading for normal usage.

## Run Configs

Runnable examples live under `examples/`. Reusable run configs live under
`runs/`. Run outputs are written under `results/runs/`, which is intentionally
ignored by git.

The public entrypoint is the `seagym` console script:

```bash
seagym train CONFIG
seagym eval CONFIG --checkpoint CHECKPOINT
seagym inspect config CONFIG
seagym inspect runtime CONFIG
```

`scripts/seagym.py` is only a thin source-tree wrapper for the same CLI.
