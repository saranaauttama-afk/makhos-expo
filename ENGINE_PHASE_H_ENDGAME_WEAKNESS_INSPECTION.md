# ENGINE_PHASE_H_ENDGAME_WEAKNESS_INSPECTION

## Scope

Phase H.0 is inspection only.

No engine behavior, eval values, search policy, or benchmark scoring is changed here.

This note summarizes the current weakness profile for a small set of endgame / tactical cases that have repeatedly shown up in benchmark review and regression-harness output.

## Evidence Sources

- latest quick benchmark artifact:
  - `.tmp/benchmarks/ai-benchmark-quick-latest.json`
- eval breakdown debug helper:
  - `scripts/evalBreakdownDebug.ts`
- eval gap analysis:
  - `ENGINE_PHASE_F_EVAL_SIGNAL_GAPS.md`
- regression harness policy:
  - `scripts/regressionHarness.ts`
  - `ENGINE_PHASE_G_REGRESSION_HARNESS_DESIGN.md`

## 1. How These Cases Currently Fail In Benchmark / Gate Output

### `small-piece-king-vs-men`

Current quick benchmark result from the latest JSON:

- `easy`: chose `22->17`, oracle `22->25`, `scoreDrop=500206`, `severeBlunder=true`
- `normal`: chose `22->17`, oracle `22->25`, `scoreDrop=500206`, `severeBlunder=true`
- `hard`: chose `22->17`, oracle `22->25`, `scoreDrop=500206`, `severeBlunder=true`
- `expert`: chose `22->17`, oracle `22->25`, `scoreDrop=500206`, `severeBlunder=true`

Regression harness treatment right now:

- `WARN`, not `FAIL`

### `small-endgame`

Current quick benchmark result from the latest JSON:

- `easy`: chose `7->10`, oracle `7->10`, `scoreDrop=0`
- `normal`: chose `7->10`, oracle `7->10`, `scoreDrop=0`
- `hard`: chose `7->10`, oracle `7->10`, `scoreDrop=0`
- `expert`: chose `7->10`, oracle `7->10`, `scoreDrop=0`

Important history:

- this case previously appeared in eval inspection as nearly flat / weakly discriminated
- it is not failing in the current latest quick benchmark artifact

### `low-mobility-squeeze`

Current quick benchmark result from the latest JSON:

- `easy`: chose `26->22`, oracle `26->22`, `scoreDrop=0`, `overrideReason=low-mobility recapture`
- `normal`: chose `26->22`, oracle `26->22`, `scoreDrop=0`, `overrideReason=low-mobility recapture`
- `hard`: chose `26->22`, oracle `26->22`, `scoreDrop=0`, `overrideReason=low-mobility recapture`
- `expert`: chose `26->23`, oracle `26->22`, `scoreDrop=1094`, `severeBlunder=true`

Regression harness treatment right now:

- not fatal at the current latest result
- would become `FAIL` only if the drop becomes catastrophic

### `quiet-hanging-piece`

Current quick benchmark result from the latest JSON:

- `quiet-hanging-piece-p1`: all levels solved, `scoreDrop=0`
- `quiet-hanging-piece-p2`: all levels solved, `scoreDrop=0`

Important history:

- this case still appears in eval-gap analysis because the current static hanging term does not explain why the case is solved
- it is not a current benchmark failure

## 2. Which Are True Catastrophic Vs Known Warnings

### True Catastrophic

At the current latest quick benchmark snapshot:

- none of the four focus cases are currently catastrophic in the harness sense except:
  - `small-piece-king-vs-men`, which is numerically catastrophic

However, policy distinction matters:

- `small-piece-king-vs-men` is currently treated as a known warning case
- `sac-two-win-three-*` remains the primary hard-stop catastrophic signature
- `low-mobility-squeeze` would be treated as fatal if it regressed into catastrophic-drop territory

### Known Warning

- `small-piece-king-vs-men`
  - catastrophic by raw drop
  - warning-only by current harness policy because it already appears in otherwise stable runs
- `low-mobility-squeeze`
  - current latest result is a real expert miss, but not catastrophic
- `small-endgame`
  - currently passes
  - remains diagnostically interesting because the eval breakdown is weak / nearly flat
- `quiet-hanging-piece`
  - currently passes
  - remains diagnostically interesting because the hanging term does not visibly explain the result

## 3. Current Eval Breakdown If Available

From `ENGINE_PHASE_F_EVAL_SIGNAL_GAPS.md` and `scripts/evalBreakdownDebug.ts`:

### `quiet-hanging-piece-p1`

```text
score=-32 sum=-32 consistency=PASS
mat=0 psqt=-36 mob=0 promo=0 hang=0 back=4 simp=0 kEg=0 kkEg=0 eg=0.25 kv=305 final=-32
mirrorScore=-32 mirrorFinal=-32 symmetry=PASS
```

Key reading:

- `hang=0` despite the benchmark label
- current signal comes mostly from `psqt`

### `low-mobility-squeeze`

```text
score=3 sum=3 consistency=PASS
mat=0 psqt=0 mob=0 promo=0 hang=0 back=4 simp=0 kEg=0 kkEg=0 eg=0.25 kv=305 final=3
mirrorScore=-7 mirrorFinal=-7 symmetry=WARN
```

Key reading:

- mobility signal is effectively absent in the base evaluator
- current practical success at easy/normal/hard depends on root override behavior, not on a strong eval signal

### `small-endgame`

```text
score=3 sum=3 consistency=PASS
mat=0 psqt=4 mob=-1 promo=0 hang=0 back=0 simp=0 kEg=0 kkEg=0 eg=0.50 kv=330 final=3
mirrorScore=-5 mirrorFinal=-5 symmetry=WARN
```

Key reading:

- endgame discrimination is very small
- current pass result in latest benchmark may be more about search/oracle alignment than a rich eval understanding

### `small-piece-king-vs-men`

- no dedicated eval breakdown fixture exists yet
- this is a prime candidate for a future debug fixture because the benchmark symptom is strong and repeatable

## 4. Likely Failure Type By Case

### `small-piece-king-vs-men`

Most likely profile:

- benchmark/oracle issue candidate: possible
- endgame search horizon / search-depth issue: possible
- endgame-specific eval weakness: possible
- root override issue: less likely from current evidence

Why:

- all levels fail the same way in the latest quick artifact
- `nodes=0` in the latest JSON suggests special-path behavior rather than a normal searched tactical miss
- this points to deeper inspection of oracle reliability, terminal/special-case handling, or endgame shortcut behavior before changing eval

### `small-endgame`

Most likely profile:

- eval issue: plausible
- oracle issue: low evidence at current snapshot
- search-depth issue: low-to-moderate
- root override issue: low

Why:

- it currently passes in the quick benchmark
- but the debug breakdown shows only tiny eval separation and symmetry warnings
- that makes it more of a latent weakness than an active failure

### `low-mobility-squeeze`

Most likely profile:

- root override issue / dependence: high
- eval issue: high
- search-depth issue: moderate
- oracle issue: low

Why:

- easy/normal/hard currently rely on `overrideReason=low-mobility recapture`
- expert currently misses without that override saving it
- eval breakdown shows `mob=0`, so the evaluator itself does not recognize the squeeze pattern well

### `quiet-hanging-piece`

Most likely profile:

- eval issue: high
- search-depth issue: low-to-moderate
- root override issue: low from current evidence
- oracle issue: low

Why:

- the case currently passes in benchmark
- but the static hanging term does not explain the pass
- this suggests the position is being solved by search/tactics rather than by an aligned quiet-vulnerability eval signal

## 5. Source Files / Functions Likely Involved

### Benchmark / Oracle Path

- `scripts/aiBenchmark.ts`
  - `oracleRoot(...)`
  - `runLevelOnCase(...)`
  - tactical case definitions

### Eval Path

- `src/coreClaude/eval.ts`
  - `evaluate(...)`
  - `handEvaluate(...)`
  - `mobilityScore(...)`
  - `hangingPiecesPenalty(...)`
  - `psqtScore(...)`
  - `kingEndgameScore(...)`
  - `allKingsEndgameScore(...)`

### Root Override Path

- `src/coreClaude/search/alphabeta.ts`
  - `overrideReason = 'low-mobility recapture'`
  - `overrideReason = 'root tactical safety'`
  - `overrideReason = 'forced recapture trap'`
  - `overrideReason = 'endgame promotion race'`

### Optional Endgame Special-Path / Oracle Support

- `src/coreClaude/search/endgameTablebase.ts`
  - likely relevant for cases where normal node counts or oracle behavior look unusual

### Debug Support

- `scripts/evalBreakdownDebug.ts`

## 6. Recommended Safe H Roadmap

### H.1 Add Debug Fixtures

Safest first step:

- add dedicated debug fixtures for:
  - `small-piece-king-vs-men`
  - `small-endgame`
  - `low-mobility-squeeze`
  - optionally `quiet-hanging-piece-p2`

Goal:

- make these cases inspectable without running the full benchmark repeatedly

Status:

- implemented in `scripts/endgameWeaknessFixtures.ts`
- existing `scripts/evalBreakdownDebug.ts` now reuses the shared fixtures instead of duplicating positions

Fixture structure:

- `EndgameWeaknessFixture`
  - `id`
  - `bucket`
  - `note`
  - `pos`

Supported fixture cases:

- `small-piece-king-vs-men`
- `small-endgame`
- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- `quiet-hanging-piece-p1`
- `quiet-hanging-piece-p2`

Future debugging workflow:

1. add or update a fixture in `scripts/endgameWeaknessFixtures.ts`
2. point a debug-only script at one or more fixture ids
3. inspect eval breakdown or oracle behavior offline
4. only then consider a tiny targeted experiment

### H.2 Inspect Oracle / Expected Move Reliability

Especially for:

- `small-piece-king-vs-men`

Goal:

- confirm whether the oracle move and score are reliable
- inspect why current latest JSON shows `nodes=0` on that case across all levels
- separate "engine weak" from "oracle/special-path mismatch"

### H.3 Tiny Endgame-Specific Eval Experiment

Only after fixture/oracle inspection:

- try a very small, isolated endgame-only signal
- avoid broad mobility rewrites
- keep it behind the eval experiment framework

Most promising first target:

- endgame-specific discrimination for king-vs-men / small-piece conversion, not general-purpose mobility

### H.4 Optional Tablebase-Like Rule Only If Justified

Only if H.1-H.3 show that:

- the failure is narrow
- the pattern is stable
- the issue is better handled by a small deterministic endgame rule than by broad eval tuning

This should remain the last option, not the first one.

## Summary

Current status of the focus set:

- `small-piece-king-vs-men`
  - strongest active weakness
  - currently warning-only in the harness, but still severe numerically
- `low-mobility-squeeze`
  - active expert weakness
  - currently masked at lower levels by a root override
- `small-endgame`
  - currently passing
  - still looks structurally under-explained by eval
- `quiet-hanging-piece`
  - currently passing
  - still looks under-explained by the static hanging term

The safest next move is not to tune yet. It is to add debug fixtures and inspect oracle / special-path reliability first, especially for `small-piece-king-vs-men`.
