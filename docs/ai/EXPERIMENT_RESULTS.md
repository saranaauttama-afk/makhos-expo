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
