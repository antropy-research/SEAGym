# Extending SEAGym

Prefer config-only extensions when possible. Add Python classes when a new
method needs new lifecycle behavior.

## Add A Dataset Or Split

Create a task index and split manifest:

```text
runs/<experiment>/
  tasks/task_index.json
  splits/split.json
  configs/run.json
```

Point the config at those files:

```json
{
  "task_dataset": {"path": "../tasks/task_index.json"},
  "split_manifest": {"path": "../splits/split.json"}
}
```

For Harbor-compatible tasks, task records should reference Harbor dataset/task
ids rather than copying task content.

## Add A Rollout Agent

Implement `RolloutAgent` when task execution behavior changes:

```python
from seagym.rollout_agents import RolloutAgent

class MyRolloutAgent(RolloutAgent):
    def rollout(self, tasks, *, env, state, context):
        ...
```

Then configure:

```json
{
  "rollout_agent": {
    "class_path": "my_package.my_agent:MyRolloutAgent",
    "config": {}
  }
}
```

## Add A Baseline

Implement `BaseBaseline` when update/checkpoint behavior changes:

```python
from seagym.baselines import BaseBaseline, UpdateResult

class MyBaseline(BaseBaseline):
    def update(self, trajectories, state):
        return UpdateResult(update_index=1, changed=True, status="updated")
```

Then configure:

```json
{
  "baseline": {
    "class_path": "my_package.my_baseline:MyBaseline",
    "config": {},
    "state": {}
  }
}
```

## Add A Metric

Register a metric function that consumes normalized records or metric inputs.
Avoid reading raw provider logs unless the metric explicitly documents that
dependency.

## Validation Checklist

Before launching a large run:

```bash
seagym inspect config CONFIG
seagym inspect env
seagym inspect runtime CONFIG
python -m unittest discover -s tests
```

For Harbor-backed extensions:

```bash
git submodule update --init reference/harbor
python -m pip install -e reference/harbor
```

For model clients or cloud runtimes, install only the extras you need:

```bash
python -m pip install -e ".[models]"
python -m pip install -e ".[external]"
python -m pip install -e ".[e2b]"
```
