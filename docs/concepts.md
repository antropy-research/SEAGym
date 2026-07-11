# Concepts

SEAGym is an evaluation framework for self-evolving LLM agents. It is not a
self-evolution algorithm and does not prescribe how an agent updates prompts,
memory, tools, planners, verifiers, or other internal state.

## Evaluation Lifecycle

SEAGym treats self-evolution as an ML-style training and evaluation process:

```text
train batch exposure
  -> baseline update
  -> checkpoint
  -> frozen update-validation assessment
  -> final held-out evaluation
```

The benchmark controls task exposure, checkpoint accounting, frozen assessment
views, metrics, and artifacts. The evaluated method controls its own update
mechanism through the `Baseline` interface.

## Core Objects

- `RolloutAgent`: runs tasks and returns trajectories.
- `Baseline`: owns cross-batch update state and checkpoints.
- `TaskEnv`: executes task batches through deterministic logic or Harbor.
- `SEAGymDataModule`: materializes train batches and evaluation views.
- `ExecutionEngine`: runs task batches, baseline updates, checkpoints, and logs.
- `MetricRegistry`: computes metrics from normalized run records.

## Data Splits And Views

Base split manifests use conventional ML names:

- `train`: tasks exposed during update.
- `val`: pool for frozen update-validation assessment.
- `test`: held-out pool for final evaluation.

Additional views such as replay, OOD, transfer, and negative-transfer probes
are materialized by config or strategy code. They are not extra base split
names.

## Evaluation Points

An evaluation point is a checkpointed agent state plus the assessments attached
to that state. Standard runs record:

- `A_0`: initial state before train-batch updates.
- `epoch_000N`: epoch-end checkpoints.
- `final`: final state after the training lifecycle.

The same checkpoint may be evaluated on update-validation, replay, ID test,
OOD test, or custom views. Metrics keep those views separate.

## Passive Evaluation

The default protocol does not accept, reject, or roll back updates. It records
observed behavior and labels update-validation changes as beneficial, neutral,
or harmful for analysis. Any active gate or rollback policy is method logic or
an explicit experiment variant, not the default SEAGym protocol.
