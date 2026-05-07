# ENGINE_PHASE_D_FRESH_BOOK_DESIGN

## Purpose

Design a fresh opening book system from scratch while treating the current legacy opening book as reference-only.

This document is design-only. It does not change engine behavior, benchmark behavior, or legacy book wiring.

## 1. Why Legacy Book Is Reference-Only

The legacy opening book is useful for understanding:

- current opening coverage size
- current position hashing approach
- current legal move revalidation pattern
- current UI and script wiring

It should remain reference-only because:

- the data is source-embedded instead of externally versioned
- generation requires manual paste-back into code
- runtime policy is mixed across direct-book and guided-search paths
- metadata is minimal
- it was built around older assumptions and should not define the new system architecture

The fresh system should be designed as a new subsystem, not as a direct extension of the legacy `BOOK` array.

## 2. Proposed New Book Data Format

Recommended top-level structure:

```ts
type FreshBookFile = {
  version: 1;
  format: 'makhos-opening-book';
  generatedAt: string;
  generator: {
    name: string;
    settings: {
      maxPly: number;
      topN: number;
      thinkMs: number;
      sidePolicy: 'p1' | 'p2' | 'both';
    };
  };
  entries: FreshBookEntry[];
};

type FreshBookEntry = {
  key: number;
  verify?: number;
  ply: number;
  side: 1 | -1;
  moves: FreshBookMove[];
};

type FreshBookMove = {
  from: number;
  to: number;
  weight: number;
  scoreCp?: number;
  rank?: number;
  note?: string;
};
```

Design goals:

- external structured data rather than a giant pasted constant
- explicit versioning
- explicit generator metadata
- per-entry multi-move support
- room for future verification fields without changing runtime API

Recommended storage options later:

- checked-in JSON for readability
- optionally precompiled TS or compact JSON for production loading

## 3. How Positions Are Keyed / Hash Verified

Recommended keying model:

1. Primary key:
   - use the existing zobrist/hash position key as the fast lookup key

2. Verification key:
   - add a second verification value for each entry when feasible
   - this can be:
     - a second independent hash, or
     - a compact canonical position fingerprint derived from board occupancy + side + promotion state inputs used by movegen

3. Runtime check:
   - lookup by primary key
   - if multiple entries match or a verify field exists, confirm the verify value before accepting the entry

Why:

- keeps lookup fast
- lowers silent collision risk
- avoids overloading the fresh system with TT policy concerns

Recommendation:

- reuse the same position hashing family already trusted by the engine
- keep fresh book verification separate from TT storage and TT replacement policy

## 4. How Legal Move Validation Works

Legal move validation should remain mandatory before any book move is accepted.

Recommended runtime flow:

1. generate current legal moves
2. lookup matching fresh book entry
3. map stored `from/to` book moves to actual legal moves
4. reject any stored move not present in the legal list
5. accept only validated moves

Recommended result classification:

- `hashMiss`
- `verifyMiss`
- `entryFoundButNoLegalMove`
- `entryFoundAndValidated`

This preserves the strongest safety property from the legacy system:

- no book move is trusted until it is revalidated against current legal move generation

## 5. Deterministic Default Selection Policy

Default selection policy should be deterministic.

Recommended default:

1. choose the validated move with highest `weight`
2. tie-break by highest `scoreCp` when present
3. tie-break by lowest `rank` when present
4. tie-break by lowest `from`
5. tie-break by lowest `to`

Why deterministic by default:

- reproducible tests
- easier debugging
- easier benchmark interpretation
- smaller risk of mistaking variance for engine improvement

Important:

- deterministic selection should be the production baseline unless an explicit experiment flag says otherwise

## 6. Optional Randomness Behind Flag Only

Randomness should be opt-in and fully flag-gated.

Recommended future flags:

- `ENABLE_FRESH_BOOK_RANDOMNESS = false`
- `FRESH_BOOK_RANDOMNESS_MODE = 'weighted' | 'top2' | 'topK'`

Rules:

- default must remain `false`
- tactical benchmark must never use random book selection
- randomness should only apply after legal validation
- randomness should not be mixed with unrelated search experiments

If randomness is ever enabled later:

- log it clearly in book stats / report metadata
- keep deterministic mode as the baseline fallback

## 7. How Tactical Benchmark Bypasses Book

The tactical benchmark should stay permanently book-free.

Recommended policy:

1. `scripts/aiBenchmark.ts` must not import the fresh opening book module
2. benchmark report should continue to carry an explicit field such as:
   - `openingBookBypassed: true`
3. any future benchmark helper must be prevented from calling book lookup accidentally

Recommended architecture:

- put fresh book access behind a dedicated module boundary
- keep tactical benchmark on pure search + level policy only

That keeps tactical benchmark results attributable to search behavior rather than opening policy.

## 8. How Book Stats Should Work

Fresh book stats should be lightweight, explicit, and separated from search stats.

Recommended counters:

- `lookupAttempts`
- `successfulHits`
- `disabledRejects`
- `benchmarkBypassRejects`
- `hashMisses`
- `verifyMisses`
- `illegalMoves`
- `selectedDeterministic`
- `selectedRandomized`
- `candidateCountTotal`

Optional derived metrics:

- average validated candidates per hit
- validation failure rate
- direct-book selection rate
- guided-book usage rate

Recommended API:

```ts
resetFreshOpeningBookStats(): void
getFreshOpeningBookStats(): FreshOpeningBookStats
```

Rules:

- no console spam during normal play
- no hot-path string building
- benchmark and debug code should read stats explicitly

## 9. Migration Plan From Legacy Book

Migration should be optional and one-way, not automatic.

Recommended path:

1. keep legacy book untouched
2. write a one-off conversion tool later:
   - read legacy rows
   - group by hash
   - map into `FreshBookEntry`
   - populate version metadata
3. mark migrated data as:
   - `source: legacy-conversion`
4. review converted output before using it
5. allow the fresh system to start with:
   - no entries, or
   - a reviewed converted seed set

Important:

- migration is a convenience path, not a trust path
- converted legacy data should still go through the fresh validation and selection rules

## 10. Rollback Plan

Rollback should be simple and low-risk.

Recommended rollback strategy later:

1. keep legacy book module intact during rollout
2. gate fresh book usage at the caller boundary
3. if fresh system misbehaves:
   - disable fresh book flag
   - return to legacy behavior or book-off behavior immediately
4. preserve separate stats for:
   - fresh book
   - legacy book

This allows rollback without touching:

- search
- TT
- eval
- repetition
- move ordering
- tactical benchmark

## 11. Recommended Next Implementation Steps

1. Add a new fresh-book design namespace and docs-only API sketch before writing runtime code.

2. Create a separate runtime module later, for example:
   - `src/coreClaude/search/openingBookFresh.ts`

3. Start with:
   - data types
   - disabled-by-default load path
   - lookup + verify + legal validation
   - deterministic selection only

4. Keep the first integration caller narrow:
   - direct script path first, or
   - isolated UI gate first

5. Keep tactical benchmark explicitly bypassed and reported as bypassed.

6. Add passive fresh-book stats before any randomness or strength tuning.

7. Only after deterministic behavior is stable:
   - consider optional randomness behind flags
   - consider legacy conversion tooling

## Recommended Architecture Summary

Fresh book system principles:

- separate module
- external structured data
- explicit versioning
- hash + verify lookup
- mandatory legal validation
- deterministic default selection
- optional randomness behind flag only
- benchmark bypass by policy and by module boundary
- rollback by caller gate, not by search surgery

## Final Recommendation

The fresh opening book should be built as a new, isolated subsystem with deterministic default behavior and explicit benchmark bypass guarantees.

The legacy book should remain:

- available
- readable
- convertible

but not trusted as the architectural foundation for the replacement system.
