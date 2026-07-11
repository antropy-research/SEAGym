# Run Configs

This directory stores reusable SEAGym run configurations.

Run outputs go under `results/runs/`, which is ignored by git. Files here are
small examples or starting points, not result artifacts.

## Included Configs

- `local_harbor/`: local Harbor-backed examples over small task indexes and
  splits. Use these as starting points for real Harbor baseline runs.
- `paper_reproduction/`: release configs and runbook for reproducing the paper
  main AHE, TF-GRPO, and ACE experiments. These are full paper-scale runs and
  require Harbor, E2B, model API keys, method submodules, and released task
  data under `SEAGYM_DATA_ROOT`.

## Config Boundary

Each experiment defines method state/update behavior under one `baseline`
object and task execution under one `rollout_agent` object.

Stable entry points:

- `baseline.class_path`
- `baseline.config`
- `baseline.models`
- `baseline.state`
- `rollout_agent.class_path`
- `rollout_agent.config`
- `rollout_agent.models`

Schedules use explicit ML/RL-style fields:

- `train_size`
- `val_size`
- `test_size`
- `batch_size`
- `num_epochs`
- `num_updates_per_batch`

Each train batch can run multiple `rollout -> update` attempts, so derived
`num_updates` is the number of materialized train batches times
`num_updates_per_batch`. Standard update-validation runs at epoch end.

Deterministic toy data used by unit tests lives under `tests/fixtures/`; the
public quickstart example lives under `examples/deterministic/`.
