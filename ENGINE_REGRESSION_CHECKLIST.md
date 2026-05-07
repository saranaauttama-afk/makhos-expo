# Engine Regression Checklist

Use this checklist for all future engine work after the baseline is frozen.

## Required Workflow

- Always clear `.tmp` before any critical benchmark run
- Run perft before engine changes
- Run perft after engine changes
- Run the quick tactical benchmark after every behavior change
- Keep all future changes small and benchmarked

## Auto-Reject Conditions

- Reject if perft fails
- Reject if `hard` blunder is greater than `0` for a release-quality target
- Reject if `expert` blunder is greater than `3` in the quick test
- Reject if any `hard` or `expert` drop is greater than `100k`
- Reject if catastrophic trap drops around `500k-999k` appear
- Reject if `sac-two-win-three-p1` fails catastrophically
- Reject if `small-piece-king-vs-men` fails with a huge drop

## Safety Rules

- Avoid root overrides unless proven safe
- Do not fabricate root candidate scores
- Do not use the opening book in tactical benchmarks
- Do not pack TT generation into the existing packed data word
- Do not integrate incremental hash into `orderMoves` yet

## Minimal Command Set

```bash
Remove-Item -Recurse -Force .tmp\*   # when preparing a critical benchmark
npm run test:perft
npm run bench:ai:fresh
```

## Working Rule

Do not optimize first and validate later. Freeze the baseline, change one thing at a time, and benchmark every behavior change.
