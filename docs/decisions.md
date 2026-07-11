# Decisions

This release branch keeps only decisions that define the public SEAGym
interface, protocol, and repository boundaries.

## Decision 001: SEAGym is an evaluation framework

Status: accepted

SEAGym evaluates observable effects of self-evolving agents. It is not a
self-evolution algorithm and does not prescribe how an agent updates memory,
prompts, tools, planners, verifiers, or other internal state.

## Decision 002: Harbor-first execution substrate

Status: accepted

The first public implementation reuses Harbor for task execution,
environments, verifiers, dataset adapters, parallel trials, and trial result
artifacts. SEAGym adds trainer / ExecutionEngine bookkeeping, update
lifecycle orchestration, result normalization, and metrics.

## Decision 003: Conventional base splits

Status: accepted

Split manifests use conventional `train` / `val` / `test` pools.
Update-validation, replay, OOD, transfer, and negative-transfer views are
materialized by experiment config and trainer strategy rather than stored as
extra base split fields.

## Decision 004: Frozen update-validation assessment

Status: accepted

The main protocol is frozen update-validation assessment. The standard loop
records an initial evaluation point and then runs update-validation at epoch
boundaries. Accept/reject gates and rollback policies are optional diagnostics,
not the main protocol.

## Decision 005: Baseline and RolloutAgent lifecycles are separate

Status: accepted

`Baseline` owns cross-batch update, state, checkpoint, load, and report
lifecycle. `RolloutAgent` owns task rollout and returns normalized
trajectories. External methods must adapt to these two surfaces instead of
mixing task rollout and method update into one opaque entrypoint.

## Decision 006: Release uses checkpoint wording

Status: accepted

Public code and docs use checkpoint/resume terminology consistently:
`checkpoints/`, `records/agent_checkpoints.jsonl`, `train --resume`,
`train --resume-from-checkpoint`, and `eval --checkpoint`. Older pre-release
wording is not part of the release API.

## Decision 007: Release branch keeps a minimal public surface

Status: accepted

The release branch excludes manuscript drafts, publication-specific run
configs, result analysis, plotting scripts, local worktree scripts, and
internal agent progress docs. The public entrypoint is the `seagym` console
command with `train`, `eval`, and `inspect` commands. Runnable examples live under
`examples/`; minimal Harbor configs live under `runs/local_harbor/`.

## Decision 008: Runtime checks live under inspect runtime

Status: accepted

Release-facing runtime validation is exposed as `seagym inspect runtime`.
Legacy standalone setup checks and all-in-one lifecycle scripts are not part
of the public release API. Runtime checks may materialize task plans, inspect
local dependencies, and optionally run Harbor canaries, but training and
checkpoint lifecycle management remains under `seagym train` and `seagym eval`.
