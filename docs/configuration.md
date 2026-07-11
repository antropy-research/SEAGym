# Configuration

SEAGym runs are configured by JSON files. A config describes what to run; local
machine details such as API keys, proxy settings, and runtime credentials live
in `.env`.

## Top-Level Shape

```json
{
  "experiment_id": "example",
  "seed": 42,
  "task_dataset": {"path": "runs/example/tasks/task_index.json"},
  "split_manifest": {"path": "runs/example/splits/split.json"},
  "schedule": {
    "train_size": 20,
    "val_size": 10,
    "test_size": 10,
    "batch_size": 5,
    "num_epochs": 2,
    "num_updates_per_batch": 1
  },
  "dataloader": {
    "shuffle_train": true,
    "seed": 42
  },
  "backend": {
    "name": "deterministic"
  },
  "rollout_agent": {
    "name": "deterministic",
    "class_path": "seagym.rollout_agents.harbor:HarborRolloutAgent",
    "config": {
      "agent": "deterministic"
    },
    "models": {}
  },
  "baseline": {
    "name": "static",
    "class_path": "seagym.baselines.static:StaticBaseline",
    "config": {},
    "state": {}
  },
  "metrics": {
    "default": true
  },
  "output": {
    "run_dir": "results/runs/example"
  }
}
```

## Schedule

- `train_size`: number of training tasks to materialize from `train`.
- `val_size`: number of validation tasks for update-validation.
- `test_size`: number of held-out test tasks for final evaluation.
- `batch_size`: training tasks per batch.
- `num_epochs`: number of passes over the materialized train set.
- `num_updates_per_batch`: rollout-update repeats for each train batch.

The derived update count is:

```text
num_train_batches * num_epochs * num_updates_per_batch
```

## Dataloader

The dataloader controls deterministic sampling and batch construction:

- `seed`: dataloader seed; defaults to the top-level `seed`.
- `shuffle_train`: whether to shuffle the materialized train set.
- `drop_last`: whether to drop incomplete train batches.
- `batching_strategy`: `shuffle` by default, or `stratified`.
- `stratify_by`: task attribute names used by stratified batching.
- `batch_plan_path`: optional frozen `BatchPlan` for paired comparisons.

## Runtime Scheduling

`runtime_scheduling` optionally reorders task dispatch *within* a materialized
batch. It never changes the task membership or batch boundaries in the
`BatchPlan`, so it can be enabled for paired scheduling comparisons.

```json
{
  "runtime_scheduling": {
    "enabled": true,
    "apply_to": ["train"],
    "policy": "lpt",
    "estimator": {
      "kind": "ema",
      "k": 5,
      "cold_start": "none"
    },
    "runtime_field": "runtime_seconds",
    "random_seed": 20260710,
    "diagnostics": {
      "record_hindsight_lpt": true
    }
  }
}
```

- `apply_to`: execution modes to reorder. Use `["train"]` for the main
  training protocol; the default is no scheduling.
- `policy`: `fixed` preserves the batch order, `random` uses a seeded shuffle,
  and `lpt` dispatches tasks with the largest predicted runtimes first.
- `estimator`: the supported estimator is EMA. With window `k`,
  `alpha = 2 / (k + 1)`. Each task prediction uses only completed earlier
  batches. `cold_start: "none"` leaves a batch in its original order unless
  every task has an observed runtime history.
- `runtime_field`: must be `runtime_seconds`, the elapsed Harbor trial time
  from `started_at` to `finished_at`, including sandbox occupation across setup,
  agent, and verifier phases.
- `random_seed`: makes the Random baseline reproducible across runs with the
  same batch plan.

For Harbor backends, enabling this section submits the batch as an ordered list
of individual task configs, rather than an unordered dataset-name filter. This
preserves the chosen dispatch order at Harbor's concurrency queue. Sandbox
timeouts remain a Harbor/backend setting; the scheduler does not clip runtimes
or change timeout policy.

## Class Paths

Runtime components are loaded from explicit Python class paths:

- `backend.name`
- `rollout_agent.class_path`
- `baseline.class_path`

`backend.name` selects the execution substrate. Use `deterministic` for local
dry runs and `harbor` for Harbor-backed benchmark tasks. Harbor runtime options
such as `env`, `n_concurrent`, `container_env`, `extra_args`, and timeout
overrides live under `backend`.

This keeps method-specific logic in Python classes while preserving a stable
JSON run surface.

## Output

`output.run_dir` is optional. If omitted, `seagym train` writes to a timestamped
directory under `results/runs/`. Use `--run-dir`, `--run-name`, `--resume`, and
`--resume-from-checkpoint` for common ML/RL checkpoint workflows.

## Portable Path Anchors

Release configs may use portable path anchors:

```text
repo://reference/ace        # path inside this repository
data://hle                  # $SEAGYM_DATA_ROOT/hle
results://my_run            # $SEAGYM_RESULTS_ROOT/my_run
```

Set `SEAGYM_DATA_ROOT` and `SEAGYM_RESULTS_ROOT` in `.env` when using
`data://` or `results://`.
