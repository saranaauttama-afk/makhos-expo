# ENGINE_PHASE_F_EVAL_SIGNAL_GAPS

## Scope

Phase F.4 is analysis only. No eval behavior was changed.

This note explains why the current breakdown output can miss important benchmark patterns even when the evaluator is working as designed.

## Current Breakdown Output

From `scripts/evalBreakdownDebug.ts`:

### quiet-hanging-piece-p1

```text
score=-32 sum=-32 consistency=PASS
mat=0 psqt=-36 mob=0 promo=0 hang=0 back=4 simp=0 kEg=0 kkEg=0 eg=0.25 kv=305 final=-32
mirrorScore=-32 mirrorFinal=-32 symmetry=PASS
```

### low-mobility-squeeze

```text
score=3 sum=3 consistency=PASS
mat=0 psqt=0 mob=0 promo=0 hang=0 back=4 simp=0 kEg=0 kkEg=0 eg=0.25 kv=305 final=3
mirrorScore=-7 mirrorFinal=-7 symmetry=WARN
```

### small-endgame

```text
score=3 sum=3 consistency=PASS
mat=0 psqt=4 mob=-1 promo=0 hang=0 back=0 simp=0 kEg=0 kkEg=0 eg=0.50 kv=330 final=3
mirrorScore=-5 mirrorFinal=-5 symmetry=WARN
```

## Case 1: Why `hang=0` May Miss `quiet-hanging-piece-p1`

Observed result:

- total eval is negative mainly because `psqt=-36`
- `hang=0` even though the case is meant to represent a quiet tactical hanging-piece pattern

Most likely reason:

- the current hanging detector is local and static
- it only checks whether a piece is currently:
  - threatened by opponent
  - not defended

Relevant source function:

- `hangingPiecesPenalty(p)` in `src/coreClaude/eval.ts`

Why it can miss this pattern:

1. It checks immediate adjacency-style threat/defense only.
   - It does not model a quiet positional squeeze where a piece becomes tactically loose after one preparatory move.

2. It does not simulate actual capture legality deeply.
   - It uses `STEPS`-based local logic, not full capture generation or exchange resolution.

3. It does not account for tempo-sensitive threats.
   - A piece may be “quietly hanging” in the benchmark sense even if it is not statically attacked on the current board.

Interpretation:

- this is not necessarily a bug in the code
- it is a signal-gap problem
- the current term is detecting “currently undefended under immediate local threat,” not “quietly tactically doomed”

## Case 2: Why `mob=0` May Happen In `low-mobility-squeeze`

Observed result:

- `mob=0`
- total eval is almost flat
- the benchmark label suggests a squeeze / low-liberty pattern

Relevant source function:

- `mobilityScore(p)` in `src/coreClaude/eval.ts`

Why it can return zero here:

1. Mobility is measured as a cheap move-space approximation, not a squeeze detector.
   - men: counts available step squares
   - kings: approximate ray mobility

2. The metric is difference-based.
   - if both sides have equally limited approximated mobility, the net term can be `0`

3. It does not measure quality of mobility.
   - no distinction between:
     - safe useful mobility
     - cramped but technically legal mobility
     - mobility that walks into tactical loss

4. It does not detect trapped or funnelled formations explicitly.
   - there is no separate trapped-piece / low-liberty term yet

Interpretation:

- `mob=0` here is plausible under the current definition
- the benchmark pattern likely needs:
  - a richer mobility shape term, or
  - a separate trapped-piece / squeeze penalty

## Case 3: Why `symmetry=WARN` May Happen

Observed:

- `quiet-hanging-piece-p1` gave `PASS`
- `low-mobility-squeeze` and `small-endgame` gave `WARN`

Important context:

- the debug script compares a “mirrored equivalent” position, not the exact same position
- the evaluator is side-relative

Relevant source pieces:

- `evaluate(p)` in `src/coreClaude/eval.ts`
- `evaluateWithBreakdown(p)` in `src/coreClaude/eval.ts`
- mirror helper in `scripts/evalBreakdownDebug.ts`

Most likely reasons:

1. The comparison is not a pure algebraic symmetry check.
   - it constructs a rotated/swapped equivalent, which may preserve broad structure but not guarantee exact term equality

2. Some eval terms are geometry-sensitive in ways that are not perfectly preserved by the current mirror helper.
   - PSQT
   - back-rank guard
   - promotion threat lanes
   - king endgame distance structure

3. Side-relative evaluation plus phase/endgame shaping can expose small asymmetries.
   - especially when a position is sparse or structurally directional

4. The current mirror helper is useful for warning-level inspection, not proof of evaluator defect.

Interpretation:

- `WARN` does not automatically mean the evaluator is incorrect
- it means the mirrored-equivalent check is surfacing non-trivial directional structure
- this should be investigated before using symmetry checks as a hard invariant

## Exact Source Functions Involved

Main eval entry:

- `evaluate(p)` in `src/coreClaude/eval.ts`
- `handEvaluate(p)` in `src/coreClaude/eval.ts`

Breakdown/debug path:

- `fillEvalBreakdown(p, out)` in `src/coreClaude/eval.ts`
- `evaluateWithBreakdown(p)` in `src/coreClaude/eval.ts`
- `scripts/evalBreakdownDebug.ts`

Terms directly implicated by current gaps:

- `hangingPiecesPenalty(p)`
- `mobilityScore(p)`
- `psqtScore(p)`
- `promotionThreatScore(p)`
- `backRankGuard(p)`
- `kingEndgameScore(p)`
- `allKingsEndgameScore(p)`

## What These Gaps Suggest

### quiet-hanging-piece-p1

Current eval is missing:

- a quiet tactical vulnerability signal
- or a stronger notion of “attacked soon / loose soon” beyond immediate local attack status

### low-mobility-squeeze

Current eval is missing:

- a trapped-piece or low-liberty signal
- or a mobility-quality signal rather than only raw mobility difference

### small-endgame

Current eval may need:

- better endgame-specific positional discrimination
- or a clearer rule for what symmetry tests are expected to preserve

## Recommended First Tiny Eval Experiment

Best first experiment:

- add a distinct trapped-piece / low-liberty inspection term behind instrumentation first, not scoring

Why this first:

- it directly addresses the `low-mobility-squeeze` gap
- it is easier to keep separate from current hanging-piece logic
- it helps decide whether mobility needs retuning at all

Alternative close second:

- improve/debug the hanging-piece inspection path for quiet threat patterns before changing any score weight

Recommended order:

1. inspect trapped / low-liberty candidates without scoring them
2. compare against `mobilityScore`
3. inspect whether quiet-hanging patterns correlate with current `hangingPiecesPenalty`
4. only then choose the first scored experiment

## Summary

The current eval breakdown is internally consistent, but it is revealing signal gaps:

- the hanging-piece term is too immediate/local for some quiet tactical losses
- the mobility term is too coarse to represent squeeze quality
- mirrored-equivalent warnings are likely exposing directional/structural asymmetry rather than a simple arithmetic bug

The safest next step is still small-scope signal inspection first, with trapped-piece / low-liberty analysis as the strongest candidate for the first tiny follow-up experiment.
