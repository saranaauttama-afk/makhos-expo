# ENGINE_PHASE_F_EVAL_INSTRUMENTATION_DESIGN

## Scope

Phase F.1 is design only. No evaluation, search, or benchmark behavior is changed here.

The goal is to expose evaluation term contributions safely before any tuning work begins.

## Scaffold Status

Phase F.2 scaffold implementation is now present in `src/coreClaude/eval.ts`.

Implemented scaffold pieces:

- `EvalBreakdown` interface
- `createEmptyEvalBreakdown()`
- `fillEvalBreakdown(p, out)`
- `evaluateWithBreakdown(p)`

Current implementation approach:

- normal `evaluate(p)` remains unchanged
- normal `handEvaluate(p)` remains the hot-path evaluator
- breakdown collection is only available through explicit debug helpers
- callers can reuse a provided sink object through `fillEvalBreakdown(...)`
- `evaluateWithBreakdown(...)` allocates only when explicitly called

## 1. Eval Terms To Expose As Breakdown Fields

The first instrumentation pass should expose the terms that already exist in `src/coreClaude/eval.ts`.

Recommended primary fields:

- `material`
- `psqt`
- `mobility`
- `promotionThreat`
- `hangingPieces`
- `backRankGuard`
- `simplification`
- `kingEndgame`
- `allKingsEndgame`

Recommended supporting fields:

- `totalPieces`
- `endgameFactor`
- `kingValue`
- `finalScore`

Optional later-only fields:

- `protectedMen`
  - only if the term is revived
- `trappedPieces`
  - only after a real trapped-piece term exists
- `promotionRace`
  - only if a separate promotion-race eval term is added

Why keep the first pass narrow:

- matches the current evaluator exactly
- avoids inventing synthetic terms that do not yet exist
- makes comparisons against current runtime scores straightforward

## 2. Proposed `EvalBreakdown` Type / Interface

Recommended shape:

```ts
export interface EvalBreakdown {
  material: number;
  psqt: number;
  mobility: number;
  promotionThreat: number;
  hangingPieces: number;
  backRankGuard: number;
  simplification: number;
  kingEndgame: number;
  allKingsEndgame: number;
  totalPieces: number;
  endgameFactor: number;
  kingValue: number;
  finalScore: number;
}
```

Optional debug wrapper:

```ts
export interface EvalDebugSnapshot {
  breakdown: EvalBreakdown;
  side: 1 | -1;
  hash?: number;
}
```

Important design choice:

- keep the breakdown numeric and flat
- avoid nested objects in the first pass
- make `finalScore` equal the actual returned eval score so consistency is easy to verify

## 3. How To Keep Normal Eval Hot Path Unchanged

Recommended implementation pattern:

1. keep the existing `evaluate(p)` entry as the hot-path function
2. keep `handEvaluate(p)` fast and allocation-free
3. add a separate debug-only function such as:
   - `evaluateWithBreakdown(p): EvalBreakdown`
4. optionally refactor shared math into a private helper that can write into a caller-provided sink

Preferred internal pattern:

```ts
function computeEvalTerms(p: Position, out?: EvalBreakdown): number
```

Rules:

- if `out` is omitted, compute score only
- if `out` is provided, populate fields
- do not allocate the breakdown object in the normal search path

This preserves the current fast path while enabling explicit debug/test inspection.

## 4. How To Expose Breakdown Only In Debug / Test Path

Recommended exposure points:

- standalone script for eval breakdown
- tactical miss review helper
- A/B eval inspection helper

Recommended non-exposure points:

- do not call breakdown from normal `alphabeta` node evaluation
- do not print breakdowns from normal gameplay
- do not push breakdowns into benchmark output by default

Safe first API set:

- `evaluate(p): number`
- `evaluateWithBreakdown(p): EvalBreakdown`

Current scaffold status:

- this API shape now exists
- there is still no benchmark integration
- there is still no search-path usage

Optional later:

- `formatEvalBreakdown(...)` helper for scripts only

## 5. How To Avoid Allocations During Normal Search

Key principle:

- no new object allocation inside the normal leaf evaluation path

Recommended safeguards:

1. no default creation of `EvalBreakdown` in `evaluate(p)`
2. no arrays or string formatting in core eval
3. no closure-heavy helper structure in hot path
4. no `generateMoves()`-based debug hooks inside normal eval

Preferred options:

- explicit debug-only function that allocates once per inspected position
- or caller-provided mutable sink object for breakdown capture

If a sink pattern is used later:

```ts
function fillEvalBreakdown(p: Position, out: EvalBreakdown): number
```

This allows scripts/tests to reuse a single object if needed.

Current scaffold status:

- both modes now exist
- `evaluateWithBreakdown(...)` is the allocating convenience path
- `fillEvalBreakdown(...)` is the reusable-sink path

## 6. How To Test Symmetry / Determinism

Recommended checks:

### Determinism

For the same `Position`:

- `evaluate(p)` should return the same number on repeated calls
- `evaluateWithBreakdown(p).finalScore` should match `evaluate(p)`
- repeated breakdown calls should yield identical field values

### Side-Relative Symmetry

Construct mirrored positions and verify:

- score signs behave as expected when swapping side/material perspective
- breakdown fields also flip consistently where appropriate

Important caution:

- the eval is side-relative, not absolute board-relative
- “symmetric” tests should compare logically mirrored states, not only identical bitboards with side flipped

Recommended first script-level assertions:

- repeated eval equality
- repeated breakdown equality
- `sum(terms) === finalScore`
- selected mirrored test cases with expected sign inversion

## 7. How To Inspect Benchmark Miss Cases Safely

Recommended workflow:

1. keep tactical benchmark itself unchanged
2. use benchmark results only to identify miss-case positions
3. feed those positions into a separate eval-inspection script
4. print:
   - eval score
   - breakdown terms
   - chosen move / oracle move context if available externally

Why separate it:

- keeps benchmark timing stable
- avoids contaminating performance-sensitive runs
- lets us inspect only a small number of miss cases

Recommended helper later:

- `scripts/evalBreakdownCases.ts`

Inputs could be:

- hard-coded case list
- selected tactical IDs copied from benchmark miss output

Current scaffold status:

- no miss-case helper script exists yet
- this remains intentionally disabled until the breakdown API is reviewed

## 8. Recommended First Tiny Eval Experiment After Instrumentation

Best first experiment:

- mobility term inspection only

Reason:

- mobility already exists
- low-mobility positions are showing up in tactical suites/benchmarks
- it is a self-contained term with low conceptual spillover

Recommended first experiment shape:

1. instrument current term contributions
2. inspect low-mobility miss cases
3. decide whether the issue is:
   - weight
   - approximation shape
   - missing trapped-piece term instead

Why not start with trapped-piece penalties immediately:

- trapped-piece logic does not exist as a standalone eval term yet
- it is easy to double-count with mobility or hanging-piece penalties

## Recommended Implementation Boundary

When Phase F.1 implementation starts later, keep changes limited to:

- `src/coreClaude/eval.ts`
- optional new debug script under `scripts/`

Do not touch:

- `alphabeta.ts`
- TT
- opening book
- repetition
- move ordering
- root overrides

## What Remains Intentionally Disabled

The current scaffold intentionally does not include:

- any `alphabeta.ts` integration
- any benchmark output integration
- any runtime logging
- any automatic object allocation in the normal search path
- any eval constant tuning
- any new eval terms
- any symmetry/determinism test script yet

## Known Risks

1. Breakdown drift risk.
   - if `handEvaluate(...)` changes later without updating `fillEvalBreakdown(...)`, debug output can go stale

2. Fractional contribution nuance.
   - `backRankGuard` is scaled by `(1 - eg)`, so its field may be fractional before final score truncation

3. Misuse risk.
   - if callers start using `evaluateWithBreakdown(...)` in hot paths, allocations will increase unnecessarily

4. Interpretation risk.
   - breakdown values remain side-relative, just like the evaluator itself

## Summary

The safest eval instrumentation design is:

- expose existing eval terms only
- keep the normal `evaluate(p)` hot path allocation-free
- add a separate debug/test-only breakdown path
- inspect miss cases outside the benchmark runner
- start with mobility analysis as the first post-instrumentation experiment

This gives visibility into current eval behavior without changing runtime search behavior or benchmark stability.
