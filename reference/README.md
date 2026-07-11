# Reference Code

This directory contains upstream code references used by SEAGym integrations.
The SEAGym core package and deterministic example do not require any reference
submodule.

## Paper Main Reproduction

The paper main reproduction configs under `runs/paper_reproduction/` require:

- `harbor/`: execution substrate for Harbor-backed tasks.
- `agentic-harness-engineering/`: AHE reference implementation.
- `ace/`: Agentic Context Engineering reference implementation.
- `tf-grpo/`: Training-Free GRPO reference branch from
  `TencentCloudADP/youtu-agent`.

Initialize only the required submodules:

```bash
git submodule update --init reference/harbor
git submodule update --init reference/agentic-harness-engineering
git submodule update --init reference/ace
git submodule update --init reference/tf-grpo
```

Install Harbor when running Harbor-backed configs:

```bash
python -m pip install -e reference/harbor
```

For E2B-backed Harbor runs:

```bash
python -m pip install -e "reference/harbor[e2b]"
python -m pip install -e ".[e2b]"
```

Prefer initializing the minimum submodule set for the run you are executing.
Do not use recursive submodule initialization for the paper reproduction path;
some upstream projects contain optional nested submodules that are not required
by SEAGym adapters.
