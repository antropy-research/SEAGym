# Paper Main Reproduction

This directory contains the release configs for reproducing the paper's main
AHE, TF-GRPO, and ACE experiments on the Terminal-Bench 2.0 plus HLE
Math/Physics source split.

The configs use the same main schedule:

- train tasks: 80
- frozen update-validation tasks: 35
- held-out ID test tasks: 55
- batch size: 20
- epochs: 5
- updates per batch: 1

## Required Submodules

Initialize Harbor and the method reference projects:

```bash
git submodule update --init reference/harbor
git submodule update --init reference/agentic-harness-engineering
git submodule update --init reference/ace
git submodule update --init reference/tf-grpo
```

Install SEAGym and Harbor:

```bash
python -m pip install -e ".[models,external,e2b]"
python -m pip install -e "reference/harbor[e2b]"
```

## Environment

Start from `.env.example` and set at least:

```bash
DEEPSEEK_API_KEY=...
E2B_API_KEY=...
SEAGYM_DATA_ROOT=/path/to/seagym-data
SEAGYM_RESULTS_ROOT=/path/to/seagym-results
```

The `seagym` CLI loads `.env` from the current working directory. If you copy an
existing `.env` from another checkout, verify that these `SEAGYM_*` path anchors
are present; older local files may contain only API keys.

The task index expects the following Harbor-compatible task trees under
`SEAGYM_DATA_ROOT`:

```text
$SEAGYM_DATA_ROOT/hle
$SEAGYM_DATA_ROOT/terminal-bench-2
```

The configs write run artifacts under:

```text
$SEAGYM_RESULTS_ROOT/runs/paper_reproduction/
```

## Preflight

Run config and runtime checks before launching expensive jobs:

```bash
seagym inspect config runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
seagym inspect config runs/paper_reproduction/configs/main_tf_grpo_terminal_hle_deepseek.json
seagym inspect config runs/paper_reproduction/configs/main_ace_terminal_hle_deepseek.json

seagym inspect runtime runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
seagym inspect runtime runs/paper_reproduction/configs/main_tf_grpo_terminal_hle_deepseek.json
seagym inspect runtime runs/paper_reproduction/configs/main_ace_terminal_hle_deepseek.json
```

Optionally run a one-task Harbor/E2B canary for each method config:

```bash
seagym inspect runtime runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json --canary --canary-task-limit 1
seagym inspect runtime runs/paper_reproduction/configs/main_tf_grpo_terminal_hle_deepseek.json --canary --canary-task-limit 1
seagym inspect runtime runs/paper_reproduction/configs/main_ace_terminal_hle_deepseek.json --canary --canary-task-limit 1
```

The canary uses Harbor's oracle agent on one selected task. It is a substrate
smoke test for task materialization, Harbor command construction, E2B sandbox
startup, and result normalization. It does not exercise the AHE, TF-GRPO, or
ACE update lifecycle and is not a substitute for a real training run.

The runtime check should pass the following gates before a full reproduction.
Passing these gates is necessary, but not sufficient, for completing the full
paper-scale runs:

- `harbor_bin`: `harbor` is on `PATH`, or `SEAGYM_HARBOR_BIN` points to it.
- `config:harbor.e2b_extra`: Harbor was installed with `reference/harbor[e2b]`.
- `config:harbor.e2b_api_key`: `E2B_API_KEY` is set.
- `env:DEEPSEEK_API_KEY`: the DeepSeek update / rollout model key is set.
- `task_path:hle`: `$SEAGYM_DATA_ROOT/hle` exists.
- `task_path:terminal-bench-2`: `$SEAGYM_DATA_ROOT/terminal-bench-2` exists.

## Full Runs

```bash
seagym train runs/paper_reproduction/configs/main_ahe_terminal_hle_deepseek.json
seagym train runs/paper_reproduction/configs/main_tf_grpo_terminal_hle_deepseek.json
seagym train runs/paper_reproduction/configs/main_ace_terminal_hle_deepseek.json
```

These are full paper-scale runs. They require working E2B capacity, Harbor
runtime setup, model API budget, and the local method submodules above. The
checked-in configs use the paper-scale concurrency settings (`backend.n_concurrent`
is 16 for AHE and 20 for TF-GRPO / ACE). Reduce `backend.n_concurrent` before
launching if your E2B account or pool has a lower concurrent sandbox limit.

## Optional E2B Prebuilt Templates

The release configs use ordinary E2B sandboxes by default so that reproduction
does not depend on maintainer-private templates. This is slower, because AHE
installs NexAU inside the sandbox and TF-GRPO / OpenCode may need method setup
during the first tasks.

If you run many full-scale experiments, you can prebuild E2B templates that
already contain method dependencies:

- AHE: NexAU plus the Harbor-facing `nexau-harbor` entrypoint.
- TF-GRPO / OpenCode: the OpenCode runtime and Python dependencies required by
  the configured rollout/update path.

For AHE, set `SEAGYM_AHE_USE_PREBUILT_E2B_TEMPLATE=True` only after you have
created a compatible template. Without that variable, SEAGym uses the portable
install script and the GitHub host allowlist in the config.

## Method Notes

- AHE uses `reference/agentic-harness-engineering` and the custom
  `AHENexAURolloutAgent`. The release default installs NexAU inside each
  ordinary E2B sandbox. The AHE configs explicitly allow GitHub-related hosts
  during sandbox setup because NexAU is installed from GitHub. Internal
  prebuilt AHE templates can be selected with
  `SEAGYM_AHE_USE_PREBUILT_E2B_TEMPLATE=True`, but they are not required for
  the public reproduction path.
- TF-GRPO uses the adapter-fixed meta prompt profile with train-only
  `n_attempts = 2` and skips metadata-only fallback update evidence.
- ACE uses the repaired batch-update trace-learning path with reward-only
  public feedback and ACE-standard trace materialization.

## Known Runtime Notes

- E2B-backed Harbor runs require the Harbor E2B extra and `E2B_API_KEY`.
- Harbor must be installed and discoverable as `harbor`, or `SEAGYM_HARBOR_BIN`
  must point to the Harbor CLI.
- Terminal-Bench tasks may depend on verifier-time network access inside the
  sandbox. Use `seagym inspect runtime ... --canary --canary-task-limit 1` if
  local networking or template setup is uncertain.
- TF-GRPO bootstraps a method-local Python environment with `uv` during native
  setup. The setup step needs network access for `uv`, Python 3.12, and Python
  packages listed in the config.
- ACE bootstraps `reference/ace/.venv/bin/python` with `uv` from the run
  config before its first native update. The setup step needs network access
  for `uv`, Python 3.12, the ACE Python package dependencies, `boto3`
  because ACE imports the Bedrock model integration through `pydantic-ai`,
  `httpx[socks]` / `socksio` when the host environment uses a SOCKS proxy, and
  `pydantic-ai-slim==1.101.0`, matching the verified ACE batch-update
  environment used for the paper results.
- AHE bootstraps NexAU in the Harbor agent setup step. If you customize the
  AHE config, preserve the `--allow-environment-host` entries for GitHub
  domains unless you use a prebuilt E2B template that already contains NexAU.
  The AHE baseline config also installs `nexau`, `httpx[socks]`,
  `httpcore[socks]`, and `socksio` into the local Python environment before the
  first native update, because the update step calls the reference AHE evolve
  agent on the host process.
- Do not initialize ACE recursively for the SEAGym paper reproduction path; the
  top-level ACE project is sufficient for SEAGym's adapter.
