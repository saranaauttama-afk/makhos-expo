#!/usr/bin/env tsx
/**
 * Historical entry point retained so old automation fails safely rather than
 * silently running the former one-sided, single-opening, wall-clock prototype.
 * Use `npm run tournament:smoke` (or import tournamentHarness.ts) instead.
 */
throw new Error(
  'DEPRECATED: headToHeadTest.ts was not a valid A/B measurement harness. ' +
  'Use `npm run tournament:smoke`; see docs/TOURNAMENT_PROTOCOL.md.',
);
