# ENGINE_STATUS

## Purpose

This is the central status/index document for the `tuneCodex` engine research branch.

Use it as the first stop before:

- starting a new experiment
- resuming work in a new chat
- deciding whether a regression is likely eval, search, oracle, or harness related

## 1. Current Architecture Overview

Core engine areas:

- Search:
  - `src/coreClaude/search/alphabeta.ts`
  - `src/coreClaude/search/tt.ts`
  - `src/coreClaude/search/repetition.ts`
  - `src/coreClaude/search/endgameTablebase.ts`
- Evaluation:
  - `src/coreClaude/eval.ts`
- Legacy opening book:
  - `src/coreClaude/search/openingBook.ts`
- Fresh opening book scaffold:
  - `src/coreClaude/search/openingBookFresh.ts`
- Benchmark / analysis:
  - `scripts/aiBenchmark.ts`
  - `scripts/analyzeAiBenchmark.ts`
  - `scripts/regressionHarness.ts`

High-level runtime shape:

- normal engine play uses search + eval + optional legacy book paths
- tactical benchmark bypasses opening book and records case-level tactical outcomes
- regression harness parses benchmark JSON offline and classifies `PASS` / `WARN` / `FAIL`
- debug scripts inspect eval, endgame fixtures, oracle/probe behavior, and benchmark artifacts without changing engine behavior

## 2. Major Completed Phases

### Phase B

- Root override inspection completed
- low-mobility override disable experiment was rejected after tactical regression
- root override instrumentation was explored, then rolled back when it proved behavior-sensitive
- main outcome:
  - treat root-level override changes as high-risk
  - avoid touching them casually

Key docs:

- `ENGINE_PHASE_A_INSPECTION.md`
- `ENGINE_PHASE_A_PLAN.md`
- `ENGINE_PHASE_B_INSTRUMENTATION.md`

### Phase C

- repetition contempt was explored as infrastructure/experiment space
- contempt and contempt instrumentation both proved behavior-sensitive
- current safe state:
  - no active repetition contempt behavior in runtime
  - Phase C remains mostly documentation/history

Key doc:

- `ENGINE_PHASE_C_REPETITION.md`

### Phase D

- legacy opening book inspected and treated as reference-only
- opening book instrumentation and benchmark isolation added
- tactical benchmark remains book-free
- legacy direct selection was made deterministic
- fresh opening book scaffold added separately, default `OFF`
- scaffold validation added without integrating the fresh book into runtime or benchmark

Key docs:

- `ENGINE_PHASE_D_OPENING_BOOK_INSPECTION.md`
- `ENGINE_PHASE_D_FRESH_BOOK_DESIGN.md`

### Phase F

- evaluation system inspected
- passive eval breakdown scaffold added
- debug script added for eval breakdown inspection
- tiny mobility experiment was tried and rejected after catastrophic regression
- safe eval experiment framework added, disabled by default

Key docs/tools:

- `ENGINE_PHASE_F_EVAL_INSPECTION.md`
- `ENGINE_PHASE_F_EVAL_INSTRUMENTATION_DESIGN.md`
- `ENGINE_PHASE_F_EVAL_SIGNAL_GAPS.md`
- `scripts/evalBreakdownDebug.ts`

### Phase G

- lightweight regression harness designed and implemented as offline parser/tooling
- harness thresholds refined to separate fatal cases from warning cases
- npm workflow scripts added for quick safety checks

Key doc/tool:

- `ENGINE_PHASE_G_REGRESSION_HARNESS_DESIGN.md`
- `scripts/regressionHarness.ts`

### Phase H

- endgame weakness inspection added
- reusable fixtures added for weak cases
- symmetry/oracle behavior inspected
- `small-piece-king-vs-men` identified as probe-suspect rather than a clean pure-eval failure
- endgame probe diagnostics added

Key doc/tools:

- `ENGINE_PHASE_H_ENDGAME_WEAKNESS_INSPECTION.md`
- `scripts/endgameWeaknessFixtures.ts`
- `scripts/endgameProbeDebug.ts`

## 3. Current Benchmark Workflow

Fast safety workflow:

1. `npm run test:perft`
2. `npm run gate:ai:report`
3. if a fresh quick run is needed:
   - `npm run gate:ai:quick`

Useful benchmark scripts:

- `npm run bench:ai:fresh`
- `npm run bench:ai:full:tactical`
- `npm run bench:ai:analyze`

Important benchmark facts:

- tactical benchmark is the main experiment-time safety signal
- benchmark JSON is written under `.tmp/benchmarks/`
- benchmark should remain opening-book-free

## 4. Current Regression Harness Workflow

Current harness entry points:

- `npm run regression:harness`
- `npm run gate:ai:report`
- `npm run gate:ai:quick`

Current harness behavior:

- parses latest quick benchmark JSON offline
- checks `openingBookBypassed`
- scans catastrophic `scoreDrop`
- groups repeated misses across levels
- prints:
  - `classification`
  - `fatalReasons`
  - `warnings`

Current high-signal fatal cases:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- repeated catastrophic failures across levels

Known warning-only special case:

- `small-piece-king-vs-men`
  - reason: probe-suspect known case

## 5. Current Known Weak Cases

Primary active/diagnostic cases:

- `small-piece-king-vs-men`
  - numerically severe
  - currently warning-only in the harness
  - probe/oracle suspicion is high
- `low-mobility-squeeze`
  - vulnerable case
  - lower levels historically benefited from root override help
- `small-endgame`
  - currently often passes
  - still looks weakly discriminated by eval
- `quiet-hanging-piece`
  - currently often passes
  - static hanging term does not explain the result well

## 6. Known Probe / Oracle Caveats

Most important current caveat:

- `small-piece-king-vs-men` is not treated as a clean tactical fail signal yet

Current H.4 finding:

- root `probeSmallEndgame(...)` returns `undefined`
- many child positions get exact probe scores
- fallback root search and child exact scoring can diverge

Practical meaning:

- do not interpret this case as a pure eval failure without checking probe/oracle context
- inspect probe diagnostics before changing search or eval for this case

## 7. Stable Benchmark Expectations / Ranges

Frozen historical baseline:

- from `ENGINE_BASELINE.md`
  - `easy 90/5`
  - `normal 95/5`
  - `hard 95/5`
  - `expert 95/3`

Useful stable reference snapshots:

- Phase A snapshot:
  - `easy 92/8`
  - `normal 97/3`
  - `hard 95/3`
  - `expert 97/3`
- strong Phase B instrumentation snapshot:
  - `easy 95/3`
  - `normal 100/0`
  - `hard 100/0`
  - `expert 97/3`

Interpretation:

- exact percentages may drift a little
- named catastrophic drops matter more than small aggregate movement
- `hard` / `expert` blunder behavior matters more than `easy` noise

## 8. Important Debug Scripts / Tools

Core debug / analysis tools:

- `scripts/evalBreakdownDebug.ts`
  - inspect eval term breakdowns and mirror behavior
- `scripts/endgameWeaknessFixtures.ts`
  - shared fixture set for weak tactical/endgame cases
- `scripts/endgameProbeDebug.ts`
  - inspect root vs child endgame probe behavior
- `scripts/regressionHarness.ts`
  - offline regression classification
- `scripts/analyzeAiBenchmark.ts`
  - benchmark analysis helper

Validation / safety tools:

- `scripts/perftThaiCheckers.ts`
- `scripts/tacticalSuite.ts`
- `scripts/ruleInvariantSuite.ts`

## 9. Important Docs Index

Start here by topic:

- Baseline / checklist:
  - `ENGINE_BASELINE.md`
  - `ENGINE_REGRESSION_CHECKLIST.md`
- Root override history:
  - `ENGINE_PHASE_A_INSPECTION.md`
  - `ENGINE_PHASE_A_PLAN.md`
  - `ENGINE_PHASE_B_INSTRUMENTATION.md`
- Repetition:
  - `ENGINE_PHASE_C_REPETITION.md`
- Opening book:
  - `ENGINE_PHASE_D_OPENING_BOOK_INSPECTION.md`
  - `ENGINE_PHASE_D_FRESH_BOOK_DESIGN.md`
- Eval:
  - `ENGINE_PHASE_F_EVAL_INSPECTION.md`
  - `ENGINE_PHASE_F_EVAL_INSTRUMENTATION_DESIGN.md`
  - `ENGINE_PHASE_F_EVAL_SIGNAL_GAPS.md`
- Regression harness:
  - `ENGINE_PHASE_G_REGRESSION_HARNESS_DESIGN.md`
- Endgame / oracle / probe:
  - `ENGINE_PHASE_H_ENDGAME_WEAKNESS_INSPECTION.md`

## 10. Recommended Workflow For Future Experiments

Preferred discipline:

1. read the relevant phase/status docs first
2. keep the change tiny and isolated
3. keep it disabled by default when possible
4. run `npm run test:perft`
5. run `npm run gate:ai:report`
6. if needed, run a fresh quick tactical benchmark
7. inspect named weak cases before trusting aggregate numbers
8. if the change touches eval/search-like behavior, stop quickly on catastrophic signals

If the change is about weak-case diagnosis:

1. use fixtures/debug scripts first
2. avoid jumping straight to full benchmark tuning
3. separate oracle/probe suspicion from pure eval suspicion

## 11. Rollback Philosophy / Experiment Discipline

Current project discipline:

- one tiny experiment at a time
- easy revert over clever integration
- documentation and diagnostics before behavior change
- rollback the smallest behavior-risk change first
- keep safe docs even when an experiment is rejected

Lessons already paid for:

- passive-looking hot-path instrumentation can still regress tactics
- tiny local eval tweaks can trigger catastrophic misses elsewhere
- root-level override changes are especially dangerous
- benchmark interpretation must account for oracle/probe caveats on special cases

## 12. Current “Do Not Touch Casually” Systems

High-risk systems:

- root override logic in `src/coreClaude/search/alphabeta.ts`
- TT policy / packing in `src/coreClaude/search/tt.ts`
- repetition handling in `src/coreClaude/search/repetition.ts`
- move ordering behavior
- benchmark oracle logic in `scripts/aiBenchmark.ts`
- endgame probe behavior in `src/coreClaude/search/endgameTablebase.ts`

Also treat cautiously:

- legacy opening book runtime wiring
- eval hot path in `src/coreClaude/eval.ts`
- any instrumentation added inside search/eval hot paths

## Summary

Current engine work is in a disciplined research/tuning phase, not a broad refactor phase.

The safest pattern is:

- inspect first
- isolate changes
- validate cheaply
- trust catastrophic named-case signals over small aggregate drift
- treat probe/oracle-suspect endgames carefully before changing eval or search
