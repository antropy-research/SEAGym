# API Reference

This page lists the main public import paths for release users.

## Config

```python
from seagym.config import ExperimentConfig, load_experiment_config
```

## Data

```python
from seagym.data import SEAGymDataModule
from seagym.data import BatchPlan, TaskIndex, SplitManifest
```

## Environments

```python
from seagym.envs import TaskEnv, DeterministicEnv, HarborEnv
```

## Rollout Agents

```python
from seagym.rollout_agents import RolloutAgent
from seagym.rollout_agents.harbor import HarborRolloutAgent
```

## Baselines

```python
from seagym.baselines import BaseBaseline, StaticBaseline, UpdateResult
```

Selected built-in and adapter baselines:

```python
from seagym.baselines.prompt_refine import PromptRefineBaseline
from seagym.baselines.ace import ACEBaseline
from seagym.baselines.ahe import AHEBaseline
from seagym.baselines.gepa import GEPABaseline
from seagym.baselines.tf_grpo import TFGRPOBaseline
```

## Trainers

```python
from seagym.trainers import SEAGymTrainer
from seagym.trainers.engine import ExecutionEngine
```

## Metrics

```python
from seagym.metrics import MetricRegistry
```

## Runtime

```python
from seagym.runtime import inspect_experiment_config, inspect_runtime, load_env_file
```

## CLI

```bash
seagym train CONFIG
seagym eval CONFIG --checkpoint CHECKPOINT
seagym inspect config CONFIG
seagym inspect env
seagym inspect runtime CONFIG
```
