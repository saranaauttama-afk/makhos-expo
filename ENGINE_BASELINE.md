# Engine Baseline

This document freezes the current engine baseline for the `tuneCodex` branch before any further tuning work.

## Baseline Identity

- Branch name: `tuneCodex`
- Base commit: `28ec66e`
- Rebuild reason: later `tuneGPT` experiments caused baseline drift, so this branch was rebuilt from the last known good baseline commit
- Current rule: no optimization before this baseline is frozen

## Current Results

- Perft: `8/8 passed`
- Quick tactical benchmark:
  - `easy`: solve `90%`, blunder `5%`
  - `normal`: solve `95%`, blunder `5%`
  - `hard`: solve `95%`, blunder `5%`
  - `expert`: solve `95%`, blunder `3%`
- `gate:ai` status: missing on this branch as an `npm` script
- Gate note: `npm run gate:ai -- --mode quick` failed only because the script is missing, not because the engine failed

## Commands Used

```bash
npm run test:perft
npm run bench:ai:fresh
npm run gate:ai -- --mode quick
```

## Available Benchmark And Test Scripts

From `package.json`, the current branch exposes these relevant scripts:

- Tests:
  - `npm run test:rules`
  - `npm run test:perft`
  - `npm run test:tactical`
- Benchmarks:
  - `npm run bench:ai`
  - `npm run bench:ai:fresh`
  - `npm run bench:ai:full`
  - `npm run bench:ai:full:fresh`
  - `npm run bench:ai:full:tactical`
  - `npm run bench:ai:teacher`
  - `npm run bench:ai:teacher:fresh`
  - `npm run bench:ai:analyze`
  - `npm run profile:ai`

There is currently no `gate:ai` script in `package.json`.

## Future Script Suggestion

If a gate script is restored later, the preferred script name should be:

- `npm run gate:ai`

## Verification Snapshot

- `npm run test:perft`: passed locally on this branch with summary `8/8 passed`
- No long benchmark was run as part of this documentation task
