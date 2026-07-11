# Deterministic SEAGym Example

This example runs the full SEAGym trainer lifecycle without Harbor, Docker, or
API keys. It uses fixture scores from `tasks/task_index.json` through
`DeterministicEnv`.

Run from the repository root:

```bash
conda activate seagym
seagym train examples/deterministic/config.json
```

If you prefer to run the source-tree wrapper directly:

```bash
python scripts/seagym.py train examples/deterministic/config.json
```

The run writes artifacts to a timestamped directory under:

```text
results/runs/
```

Use this example to verify installation, inspect the run artifact layout, and
test custom metric changes before launching Harbor-backed runs.
