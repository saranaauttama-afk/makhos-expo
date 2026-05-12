# Phase AL.2 Near-Horizon Capture Extension Result

## Status

`REJECTED`

## Scope

AL.2 tested the distinct near-horizon capture-extension slice only.

Implementation scope during the test:

- narrow experiment in `src/coreClaude/search/alphabeta.ts`
- `OFF` by default
- explicit flag:
  - `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`
- no eval rewrite
- no global depth increase
- no promotion

## What Was Tested

AL.2 implemented a near-horizon tactical extension intended to affect only the distinct slice that current tactical extension logic does not already cover:

- `depth == 1`
- capture-sensitive tactical line
- max `+1` ply
- bounded so it would not become a broad search rewrite

## Benchmark Process

Both baseline and experiment-on checks were run:

- clean `OFF` benchmarks
- experiment `ON` benchmarks under the explicit env flag

Required checks were executed:

- `npm.cmd run test:perft`
- `npm.cmd run gate:ai:report`
- `npm.cmd run gate:ai:repeat`

## Result

Stop condition was hit.

Protected tactical case regression:

- `sac-two-win-three-p1` became more unstable under the flag

Observed comparison:

- clean-OFF: `1/3` catastrophic repeat
- experiment-ON: `2/3` catastrophic repeat

Additional protected-case note:

- `sac-two-win-three-p2` did not become the primary failure signal
- the rejection reason is specifically the increased instability of `sac-two-win-three-p1`

## Decision

AL.2 is rejected.

Exact reason:

- protected tactical case `sac-two-win-three-p1` became more unstable under the near-horizon capture-extension experiment

This is sufficient to reject the experiment.

Do not tune around this result.

## Revert / Post-Test State

After the rejection:

- AL.2 code was reverted only in `src/coreClaude/search/alphabeta.ts`
- `npm.cmd run test:perft` passes again after revert
- no promotion
- no tuning
- no commit yet

## Rule Going Forward

Do not retry near-horizon capture extension unless the search architecture changes significantly.

Examples of significant change:

- materially different extension-budget architecture
- materially different quiescence / horizon architecture
- materially different root tactical override architecture

Do not revisit this as another small capture-extension tweak on the current architecture.

## Recommended Next Direction

Prefer one of:

- docs/tooling only
- a different isolated search idea

Do not continue with another capture-extension tweak in the current architecture.
