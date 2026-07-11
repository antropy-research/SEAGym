# Evaluation

Use `seagym eval` to evaluate an existing checkpoint without running new
training updates.

```bash
seagym eval CONFIG --checkpoint results/runs/old/checkpoints/epoch_0001
```

## Evaluation Views

Config and strategy code define the views to materialize. Common views are:

- `update_validation`: frozen validation view for intermediate checkpoints.
- `final.id_test`: held-out in-distribution test view.
- `final.ood_test`: held-out OOD or transfer view.
- `replay`: tasks already seen during training, used for forgetting analysis.
- `negative_transfer_probe`: tasks selected to probe harmful transfer.

Views are task-id lists, not new split names.

## Records

Evaluation writes normalized JSONL records:

```text
records/task_results.jsonl
records/verifier_results.jsonl
records/metric_inputs.jsonl
records/evaluation_points.jsonl
```

When `runtime_scheduling.enabled` is set, runs additionally write:

```text
records/scheduling_decisions.jsonl
records/task_runtimes.jsonl
runtime/scheduling_history.json
reports/scheduling_summary.json
```

`scheduling_decisions.jsonl` records the original and dispatched task orders,
policy, predictions, worker count, and cold-start state for every scheduled
batch. `task_runtimes.jsonl` records the observed `runtime_seconds` used for
future predictions. The history file makes resumed runs use the same runtime
observations accumulated before the checkpoint boundary. The summary includes
per-batch observed-order makespan, a workload lower bound, and hindsight LPT
when all batch runtimes are available. These diagnostics describe scheduling
quality; hindsight LPT is a heuristic and is not an exact optimum certificate.

Harbor-backed runs also keep raw Harbor job references under:

```text
harbor/jobs/
```

## Scoring

Task scores are normalized from verifier rewards according to the task index
and metric config. A task is successful when its normalized score meets the
configured success threshold.

If a task execution errors, the task result records the error and contributes a
failed score unless a custom metric explicitly chooses different semantics.

## Reports

Human-readable summaries are written under:

```text
reports/summary.md
```

CSV audit views may also be written for task-level and checkpoint-level
inspection.
