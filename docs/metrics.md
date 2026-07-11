# Metrics

SEAGym computes metrics from normalized run records rather than from raw Harbor
artifacts directly.

Default inputs:

```text
records/evaluation_points.jsonl
records/task_results.jsonl
records/verifier_results.jsonl
records/metric_inputs.jsonl
```

## Default Metrics

### Success Rate

```text
SR(A, D) = successful tasks / total tasks
```

### Cost

SEAGym records rollout, update, and overall cost when providers expose it:

- input tokens;
- cache tokens;
- output tokens;
- total tokens;
- tool calls;
- wall-clock time;
- optional `cost_usd`.

Missing provider usage remains missing rather than being guessed.

### Validation-Supported Update Rate

Validation-Supported Update Rate reports the fraction of update-validation
assessments labeled beneficial under the configured threshold policy.

Harmful and neutral labels remain available as diagnostics.

## Update-Validation Labels

Standard labels are:

- `beneficial`: update-validation score improved by at least the configured
  threshold.
- `neutral`: change is within the neutral band.
- `harmful`: score degraded by at least the configured threshold.

These labels describe observed behavior. They do not trigger automatic rollback
in the default protocol.

## Replay And Forgetting

Replay metrics compare checkpoint behavior on already seen tasks. They are
useful for diagnosing regressions and forgetting but should be reported
separately from held-out final performance.

## Custom Metrics

Custom metrics can be registered in Python:

```python
from seagym.metrics import MetricRegistry

registry = MetricRegistry.default()
registry.register("my_metric", my_metric_fn)
```

Metrics should consume normalized records or metric inputs so they remain
reproducible without rerunning task environments.
