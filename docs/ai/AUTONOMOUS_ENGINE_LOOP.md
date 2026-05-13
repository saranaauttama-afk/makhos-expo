# Autonomous Engine Loop

## Purpose

This document defines the default autonomous loop for future engine experiments on branch `tuneCodex`.

Scope:

- docs/tooling and isolated experiment execution workflow
- no automatic promotion
- no broad search/eval rewrites

## Current State

- Phase AL.2 near-horizon capture extension was tested and rejected
- runtime code was reverted after AL.2
- `npm.cmd run test:perft` passes after revert
- do not retry AL.2 near-horizon capture extension on the current architecture
- stop aggressive tiny eval tuning
- prefer isolated search/gameplay-strength experiments

## Source Of Truth

Read these before each loop run:

- `ENGINE_STATUS.md`
- `PROJECT_HANDOFF_ENGINE_STATUS.md`
- `CURRENT_TASK.md`
- `docs/ai/EXPERIMENT_QUEUE.md`
- `docs/ai/EXPERIMENT_RESULTS.md`

## Protected Cases

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`

Secondary monitored cases:

- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- `quiet-hanging-piece-p1`
- `small-piece-king-vs-men`
  - informational only
  - noisy / oracle-probe suspect

## No-Go Rules

- no global depth increase
- no broad search rewrite
- no eval rewrite
- no mass revert
- no automatic promotion
- no retry of AL.2 near-horizon capture extension

## Per-Loop Rules

- pick one experiment from `docs/ai/EXPERIMENT_QUEUE.md`
- implement one isolated `OFF`-by-default experiment only
- keep default runtime behavior unchanged when the flag is `OFF`
- run baseline gates before experiment-on gates
- compare protected cases first
- if a protected case regresses, revert only that experiment
- record the outcome in `docs/ai/EXPERIMENT_RESULTS.md`
- continue automatically to the next queued experiment
- do not ask the user between experiments

## Stop Conditions

Stop the autonomous loop immediately when:

- queue is exhausted
- `test:perft` fails after revert
- runtime code remains dirty after failed revert
- clean-OFF protected case worsens
- 3 experiments have been attempted in one loop run

## Standard Gate Command

Use this gate runner for both baseline and experiment-on checks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ai\run-engine-gate.ps1
```

## Standard Loop

For each selected experiment:

1. Confirm it is still allowed under the no-go rules.
2. Make one isolated `OFF`-by-default change only.
3. Run clean-OFF gates:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\ai\run-engine-gate.ps1`
4. Run experiment-ON gates with the explicit env flag for that experiment.
5. Compare:
   - `sac-two-win-three-p1`
   - `sac-two-win-three-p2`
   - then secondary monitored cases
6. Classify:
   - rejected
   - promising
   - inconclusive
7. If protected cases regress:
   - revert only that experiment
   - rerun `npm.cmd run test:perft`
   - record rejection
8. If baseline itself worsens:
   - stop the loop
9. Move to the next experiment unless a stop condition is hit.

## Result Recording

Each experiment result should record:

- experiment name
- isolated file scope
- explicit flag name
- baseline gate summary
- experiment-on gate summary
- protected case outcome
- decision
- whether revert was required

## Recommendation

Use this loop for future search/gameplay-strength experiments first.

Do not use it to:

- re-open AL.2
- perform broad tuning sweeps
- auto-promote runtime behavior
