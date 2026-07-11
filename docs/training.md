# Training

The public training entrypoint is:

```bash
seagym train CONFIG
```

`scripts/seagym.py` is only a source-tree wrapper for environments where the
console script is not installed.

## Standard Lifecycle

```text
load config
load task index and split manifest
materialize BatchPlan
build env, rollout agent, baseline, metrics
save A_0 checkpoint
for each epoch:
  for each train batch:
    rollout train tasks
    update baseline
  save epoch checkpoint
  run frozen update-validation
run final evaluation views
write metrics and reports
```

## Checkpoints

SEAGym uses checkpoint terminology consistently. Standard checkpoint
directories include:

```text
checkpoints/A_0/
checkpoints/epoch_0001/
checkpoints/final/
```

Each checkpoint contains baseline-owned state plus SEAGym metadata needed to
resume or evaluate the state.

## Resume

Resume the same run directory:

```bash
seagym train CONFIG --run-dir results/runs/smoke --resume
```

Continue from a checkpoint into a new run:

```bash
seagym train CONFIG --resume-from-checkpoint results/runs/old/checkpoints/epoch_0001
```

These two modes share the same underlying semantics: load checkpointed state,
restore trainer bookkeeping when available, and continue from the requested
point.

## Run Directory

If no run directory is supplied, SEAGym creates a timestamped directory under
`results/runs/`.

Run directories are self-contained audit packages:

```text
results/runs/<run_id>/
  inputs/
  records/
  runtime/
  checkpoints/
  reports/summary.md
  harbor/jobs/
```

## Python Usage

Researchers can call the trainer directly when they need custom orchestration:

```python
from seagym.trainers import SEAGymTrainer

trainer = SEAGymTrainer.from_config("examples/deterministic/config.json")
trainer.fit()
```

For lower-level workflows, use `ExecutionEngine` primitives directly.
