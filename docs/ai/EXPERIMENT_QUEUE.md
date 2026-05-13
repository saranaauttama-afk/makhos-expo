# Experiment Queue

Status values:

- `pending`
- `in_progress`
- `rejected`
- `done`
- `research_only`

Rules:

- attempt only one isolated experiment at a time
- maximum 3 experiment attempts per autonomous loop run
- do not retry rejected AL.2 near-horizon capture extension

Autonomous run state:

- `paused`
- reason: clean-OFF protected baseline worsened during the first autonomous experiment attempt on 2026-05-12

## Queue

| Priority | Experiment | Status | Type | Notes |
| --- | --- | --- | --- | --- |
| 1 | Root tactical move ordering only | `pending` | isolated search experiment | `OFF` by default; no depth increase; 2026-05-12 attempt aborted before experiment-ON because clean-OFF `sac-two-win-three-p1` worsened |
| 2 | Capture-only quiescence ordering only | `pending` | isolated search experiment | ordering only; no quiescence semantic rewrite |
| 3 | Recapture-priority ordering only | `pending` | isolated search experiment | ordering only; protect `sac-two-win-three-*` first |
| 4 | Endgame probe consistency diagnostics only | `pending` | diagnostics only | no runtime strength promotion |
| 5 | Opening/tactical pattern book docs-only research | `pending` | docs only | no runtime engine behavior change |

## Rejected / Do Not Retry

| Experiment | Status | Reason |
| --- | --- | --- |
| AL.2 near-horizon capture extension | `rejected` | Protected tactical case `sac-two-win-three-p1` became more unstable; do not retry unless search architecture changes significantly |
