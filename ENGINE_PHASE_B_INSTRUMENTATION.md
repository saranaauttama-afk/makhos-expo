# Engine Phase B Instrumentation

## What Was Instrumented

Passive in-memory counters were added around the root override path in `alphabeta.ts` to measure:

- low-mobility recapture override attempts
- low-mobility recapture override accepted
- trap override attempts
- trap override accepted
- opening diversification attempts and accepted
- promotion override attempts and accepted
- root tactical safety override attempts and accepted
- anti-hang safety override attempts and accepted
- final selected move source counts

## Files Changed

- `src/coreClaude/search/alphabeta.ts`
- `ENGINE_PHASE_B_INSTRUMENTATION.md`

## Validation Snapshot

Instrumentation validation benchmark passed with:

- `easy`: `95/3`
- `normal`: `100/0`
- `hard`: `100/0`
- `expert`: `97/3`

## How To Read Or Reset Stats

The search module now exports two helper functions:

- `getRootOverrideStats()`
- `resetRootOverrideStats()`

Recommended benchmark or debug flow:

```ts
import { getRootOverrideStats, resetRootOverrideStats } from './src/coreClaude/search/alphabeta';

// 1. Clear counters before the run
resetRootOverrideStats();

// 2. Run searches or a benchmark

// 3. Read a snapshot after the run
const stats = getRootOverrideStats();
```

The returned snapshot includes:

- total `searches`
- per-override `attempts`
- per-override `accepted`
- `finalMoveSource` counts

The stats object is in-memory only and does not print to the console during normal play.

## Behavior Impact

- No selected moves should change
- No scores should change
- No override was disabled
- No benchmark script behavior was changed
- The instrumentation is passive and intended only for debug or benchmark reads

## Next Suggested Validation

- Run `npm run test:perft`
- Then, when requested, use the existing tactical benchmark flow and read the override stats before and after the run
