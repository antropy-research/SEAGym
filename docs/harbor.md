# Harbor Integration

SEAGym uses Harbor as the default execution substrate for benchmark tasks.
Harbor owns dataset integration, task execution, environments, verifiers,
parallel jobs, and raw task artifacts. SEAGym adds the outer trainer lifecycle,
checkpoint accounting, normalized records, metrics, and reports.

## Install Harbor

Initialize the Harbor submodule:

```bash
git submodule update --init reference/harbor
```

Install it locally:

```bash
python -m pip install -e reference/harbor
```

For E2B-backed Harbor runs:

```bash
python -m pip install -e "reference/harbor[e2b]"
python -m pip install -e ".[e2b]"
```

Initialize only the submodules required by the run you are executing. The paper
reproduction path does not require recursive submodule initialization.

## Runtime Environment

Local machine configuration belongs in `.env`, not in experiment configs.
Start from `.env.example`.

Common variables:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`
- `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`
- `SEAGYM_CONTAINER_HTTP_PROXY`, `SEAGYM_CONTAINER_HTTPS_PROXY`
- `SEAGYM_HARBOR_BIN`, `SEAGYM_HARBOR_ENV`, `SEAGYM_HARBOR_N_CONCURRENT`
- `E2B_API_KEY`, `DAYTONA_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
- `SEAGYM_DATA_ROOT`, `SEAGYM_RESULTS_ROOT`

Experiment configs should reference environment variable names or portable path
anchors, not secret values.

## Inspect Runtime

Run read-only checks:

```bash
seagym inspect env
seagym inspect runtime CONFIG
```

Run a small canary only when you want to execute Harbor:

```bash
seagym inspect runtime CONFIG --canary --canary-task-limit 1
```

## Harbor Agent Boundary

SEAGym does not modify Harbor source code or duplicate Harbor runner logic.
`HarborEnv` invokes Harbor through supported CLI/config/result surfaces and
normalizes job, trial, reward, error, and artifact references into SEAGym
records.

Rollout agent settings live under `rollout_agent.config`; baseline update
settings live under `baseline.config` and `baseline.state`.
