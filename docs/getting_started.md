# Getting Started

This guide covers the fastest local path from a fresh checkout to a runnable
SEAGym command.

## Install

Clone the repository:

```bash
git clone <repo-url> seagym
cd seagym
```

For deterministic examples only, the Harbor submodule is not required:

```bash
python -m pip install -e .
```

For Harbor-backed runs, initialize only the submodules required by the config
you plan to run. The local Harbor examples need Harbor:

```bash
git submodule update --init reference/harbor
```

For development:

```bash
conda env create -f environment.yml
conda activate seagym
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
```

Optional extras are feature-scoped:

```bash
python -m pip install -e ".[models]"    # OpenAI / LiteLLM clients
python -m pip install -e ".[external]"  # native external baseline helpers
python -m pip install -e ".[e2b]"       # E2B runtime helper
python -m pip install -e ".[all]"       # all optional SEAGym extras
```

## Run The Deterministic Example

The deterministic example requires no Harbor install, Docker, API key, or
external model:

```bash
seagym train examples/deterministic/config.json
```

The same command can be launched through the source-tree wrapper:

```bash
python scripts/seagym.py train examples/deterministic/config.json
```

Run artifacts are written to a timestamped directory:

```text
results/runs/<YYYYMMDD-HHMMSS>_examples_deterministic/
```

## Inspect A Config

Use read-only checks before launching larger runs:

```bash
seagym inspect config examples/deterministic/config.json
seagym inspect env
```

For Harbor-backed runs:

```bash
seagym inspect runtime runs/local_harbor/configs/oracle_no_update.json
```

Add a canary only when you explicitly want to execute a tiny Harbor task:

```bash
seagym inspect runtime CONFIG --canary --canary-task-limit 1
```

## Common Commands

```bash
seagym train CONFIG
seagym train CONFIG --run-name smoke
seagym train CONFIG --run-dir results/runs/smoke --resume
seagym train CONFIG --resume-from-checkpoint results/runs/old/checkpoints/epoch_0001
seagym eval CONFIG --checkpoint results/runs/old/checkpoints/epoch_0001
```

See `docs/training.md` and `docs/evaluation.md` for lifecycle details.
