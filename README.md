# SEAGym

SEAGym is an evaluation environment for measuring self-evolving LLM agents as
controlled agent-harness update processes. It evaluates how changes to prompts,
memory, tools, middleware, runtime state, and model-tool interaction loops affect
train, frozen update-validation, held-out test, replay, and cost views.

This repository contains the release code for the arXiv paper:

```text
SEAGym: An Evaluation Environment for Self-Evolving LLM Agents
Congjie Zheng, Chuanyi Xue, Bin Liang, Jun Yang, Changshui Zhang
arXiv:2606.17546
```

Paper: https://arxiv.org/abs/2606.17546

SEAGym is an evaluation framework, not a self-evolution algorithm. It does not
accept, reject, or roll back agent updates. Instead, it records checkpointed
agent states and evaluates them through complementary views that expose reusable
improvement, overfitting, forgetting, cost changes, and reliability shifts.

## Minimal Quickstart

The deterministic example runs without Harbor, Docker, E2B, external datasets,
or model API keys.

```bash
git clone <repo-url> seagym
cd seagym

conda env create -f environment.yml
conda activate seagym
python -m pip install -e ".[dev]"

seagym train examples/deterministic/config.json
```

The same command can be launched through the source-tree wrapper when the
console script is not installed:

```bash
python scripts/seagym.py train examples/deterministic/config.json
```

Run artifacts are written to timestamped directories under:

```text
results/runs/<YYYYMMDD-HHMMSS>_<experiment_id>/
```

Run the test suite with:

```bash
python -m unittest discover -s tests
```

## Paper Main Reproduction

The paper experiments instantiate SEAGym on Terminal-Bench 2.0 and HLE, and
compare ACE, TF-GRPO, and AHE under a shared epoch/batch protocol. The release
configs live under `runs/paper_reproduction/`.

Initialize the required submodules:

```bash
git submodule update --init reference/harbor
git submodule update --init reference/agentic-harness-engineering
git submodule update --init reference/ace
git submodule update --init reference/tf-grpo
```

Install SEAGym and Harbor with the features used by the reproduction configs:

```bash
python -m pip install -e ".[models,external,e2b]"
python -m pip install -e "reference/harbor[e2b]"
```

Create `.env` from `.env.example` and set at least:

```bash
DEEPSEEK_API_KEY=...
E2B_API_KEY=...
SEAGYM_DATA_ROOT=/path/to/seagym-data
SEAGYM_RESULTS_ROOT=/path/to/seagym-results
```

The task data root should contain:

```text
$SEAGYM_DATA_ROOT/hle
$SEAGYM_DATA_ROOT/terminal-bench-2
```

Run preflight checks before launching expensive jobs:

```bash
seagym inspect config runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
seagym inspect runtime runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
```

Launch the full paper-scale runs:

```bash
seagym train runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
seagym train runs/paper_reproduction/configs/main_tf_grpo_terminal_hle_deepseek.json
seagym train runs/paper_reproduction/configs/main_ace_terminal_hle_deepseek.json
```

These runs require E2B capacity, model API budget, released task data, Harbor,
and the method submodules above. The checked-in configs use paper-scale
concurrency settings; reduce `backend.n_concurrent` if your E2B account or pool
has a lower concurrent sandbox limit.

See `runs/paper_reproduction/README.md` for the complete reproduction notes,
including canary checks, runtime gates, method-specific setup, and optional E2B
prebuilt template guidance.

## Core Workflow

SEAGym follows an ML/RL-style lifecycle:

```text
load config
materialize train batches and evaluation views
save A_0 checkpoint
for each epoch:
  for each train batch:
    rollout tasks
    update baseline
  save epoch checkpoint
  run frozen update-validation assessment
run held-out final evaluation views
write normalized records, metrics, reports, and Harbor refs
```

The main evaluation views are:

- train batches for controlled task exposure;
- frozen update-validation for intermediate checkpoint assessment;
- held-out ID and OOD tests for final generalization;
- replay diagnostics for forgetting;
- cost and process diagnostics for update reliability.

## Commands

```bash
seagym train CONFIG
seagym train CONFIG --run-name smoke
seagym train CONFIG --run-dir results/runs/smoke --resume
seagym train CONFIG --resume-from-checkpoint results/runs/old/checkpoints/epoch_0001
seagym eval CONFIG --checkpoint results/runs/old/checkpoints/epoch_0001
seagym inspect config CONFIG
seagym inspect env
seagym inspect runtime CONFIG
```

## Configuration And Artifacts

Release configs may use portable path anchors:

```text
repo://reference/ace        # path inside this repository
data://hle                  # $SEAGYM_DATA_ROOT/hle
results://my_run            # $SEAGYM_RESULTS_ROOT/my_run
```

Each run is a self-contained audit package:

```text
results/runs/<run_id>/
  inputs/
  records/
  checkpoints/
  reports/summary.md
  harbor/jobs/
```

Run outputs, local `.env` files, Python caches, and editable-install metadata are
intentionally ignored by git.

## Documentation

- `docs/getting_started.md`: installation, deterministic example, and common
  commands.
- `docs/concepts.md`: lifecycle, evaluation points, passive evaluation, and
  SEAGym's framework boundary.
- `docs/configuration.md`: JSON config structure and portable paths.
- `docs/data.md`: task indexes, split manifests, and batch plans.
- `docs/training.md`: training lifecycle, checkpoints, resume, and Python usage.
- `docs/evaluation.md`: checkpoint evaluation, views, records, reports.
- `docs/metrics.md`: default metrics and update-validation labels.
- `docs/harbor.md`: Harbor setup and runtime boundary.
- `docs/extending.md`: extension points for datasets, rollout agents, baselines,
  metrics, and runtime checks.
- `docs/api_reference.md`: public imports and CLI entrypoints.

## Optional Installs

Install only the extras required by the workflow you run:

```bash
python -m pip install -e ".[models]"    # OpenAI / LiteLLM model clients
python -m pip install -e ".[external]"  # native external baseline helpers
python -m pip install -e ".[e2b]"       # SEAGym E2B runtime override helper
python -m pip install -e ".[all]"       # all SEAGym optional extras
```

For Harbor-backed runs:

```bash
git submodule update --init reference/harbor
python -m pip install -e "reference/harbor[e2b]"
```

## Citation

```bibtex
@misc{zheng2026seagym,
  title = {SEAGym: An Evaluation Environment for Self-Evolving LLM Agents},
  author = {Zheng, Congjie and Xue, Chuanyi and Liang, Bin and Yang, Jun and Zhang, Changshui},
  year = {2026},
  eprint = {2606.17546},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2606.17546}
}
```
