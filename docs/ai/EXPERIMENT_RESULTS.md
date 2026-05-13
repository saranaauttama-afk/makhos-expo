# Experiment Results

## Result Template

### YYYY-MM-DD - Experiment Name

- Status:
- Scope:
- Flag:
- Baseline gates:
- Experiment-on gates:
- Protected cases:
- Decision:
- Revert:
- Next:

## Recorded Results

### 2026-05-12 - AL.2 near-horizon capture extension

- Status: `rejected`
- Scope: narrow search experiment in `src/coreClaude/search/alphabeta.ts`
- Flag: `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`
- Baseline gates:
  - `npm.cmd run test:perft` passed
  - clean-OFF repeat showed `sac-two-win-three-p1` still unstable
- Experiment-on gates:
  - `npm.cmd run test:perft` passed
  - experiment-ON repeat made the protected tactical case worse
- Protected cases:
  - `sac-two-win-three-p1`: clean-OFF `1/3` catastrophic repeat, experiment-ON `2/3` catastrophic repeat
  - `sac-two-win-three-p2`: remained clean
- Decision: reject
- Revert: reverted only the AL.2 code in `src/coreClaude/search/alphabeta.ts`
- Next: do not retry near-horizon capture extension unless search architecture changes significantly

### 2026-05-12 - Root tactical move ordering only

- Status: `aborted`
- Scope: root-only move ordering experiment in `src/coreClaude/search/alphabeta.ts`
- Flag: `MAKHOS_ENABLE_ROOT_TACTICAL_MOVE_ORDERING`
- Baseline gates:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\ai\run-engine-gate.ps1`
  - `npm.cmd run test:perft` passed
  - clean-OFF repeat worsened the protected baseline on `sac-two-win-three-p1`
- Experiment-on gates:
  - not run
  - autonomous loop stopped before experiment-ON because the clean-OFF protected baseline worsened
- Protected cases:
  - `sac-two-win-three-p1`: catastrophic in `2/3` clean-OFF repeats
  - `sac-two-win-three-p2`: clean in all `3/3` clean-OFF repeats
- Decision: abort this experiment attempt and stop the autonomous run
- Revert: reverted only the root tactical move ordering code in `src/coreClaude/search/alphabeta.ts`; `npm.cmd run test:perft` passed after revert
- Next: do not continue queue execution until the clean-OFF protected baseline is stable enough to compare experiments safely

## Baseline Stability Diagnostics

### 2026-05-13 - Clean-OFF protected-case stability check

- Status: `diagnostic`
- Scope: clean-OFF repeated gate only, no experiment flags
- Repeat count: `6` runs total
- Commands:
  - `npm.cmd run gate:ai:repeat`
  - `npm.cmd run gate:ai:repeat`

- Protected-case outcomes by run:
  - `batch1-run1`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch1-run2`
    - `sac-two-win-three-p1`: catastrophic
    - `sac-two-win-three-p2`: clean
  - `batch1-run3`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run1`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run2`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run3`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean

- Ratios:
  - `sac-two-win-three-p1`: clean `5/6`, catastrophic `1/6`
  - `sac-two-win-three-p2`: clean `6/6`, catastrophic `0/6`

- Key diagnostic observation:
  - on the catastrophic `p1` run, the benchmark `oracleMove` itself changed materially versus the clean runs
  - clean runs saw `oracleMove` variants such as `7->2K` or `8->4K`
  - the catastrophic run switched to `10->28` with near-`999k` score drops

- Suspected instability cause:
  - primary: oracle/search time-budget instability in the benchmark path
  - secondary: nondeterministic search/order behavior inside the oracle/engine interaction
  - not the main suspect:
    - stale benchmark artifact reuse
    - pure `small-piece-king-vs-men`-style oracle noise on the protected case

- Supporting evidence:
  - all six runs were freshly generated and saved with distinct timestamps
  - `sac-two-win-three-p2` remained stable across all six runs
  - `sac-two-win-three-p1` flipped only when the benchmark oracle target changed
  - `scripts/aiBenchmark.ts` uses a time-limited oracle search (`ORACLE_MS=1500`, `ORACLE_DEPTH=11` in quick mode), so oracle output can vary by run

- Worktree note:
  - `src/coreClaude/search/alphabeta.ts` still shows modified in `git status`
  - `git diff -- src/coreClaude/search/alphabeta.ts` is empty
  - `git ls-files --eol` reports `i/lf` and `w/mixed`
  - with `core.autocrlf=true`, this strongly suggests line-ending normalization / mixed-EOL worktree state rather than a live content diff

- Decision:
  - keep the autonomous queue paused
  - do not resume item 2 yet

- Next:
  - prefer diagnostics/tooling to stabilize or explain the protected-case benchmark oracle path before resuming autonomous experiments
