# PROJECT_HANDOFF_ENGINE_STATUS

## 1. Current Branch

- Branch: `tuneCodex`
- Current reference HEAD at handoff time: `69ea3c8`

## 2. Latest Stable Benchmark Expectations

Use these as practical safety bands, not exact required percentages:

- frozen baseline:
  - `easy 90/5`
  - `normal 95/5`
  - `hard 95/5`
  - `expert 95/3`
- stronger later stable reference snapshot:
  - `easy 95/3`
  - `normal 100/0`
  - `hard 100/0`
  - `expert 97/3`

Interpretation:

- named catastrophic drops matter more than small aggregate drift
- `hard` / `expert` blunder behavior matters more than `easy` noise

## 3. Major Completed Phases

- Phase B:
  - root override inspection completed
  - root override changes proved high-risk
- Phase C:
  - repetition contempt experiments documented
  - no active contempt behavior kept in runtime
- Phase D:
  - legacy opening book inspected
  - benchmark isolation added
  - fresh opening book scaffold added, default `OFF`
- Phase F:
  - eval inspection completed
  - eval breakdown scaffold/debug tooling added
  - tiny mobility experiment rejected after catastrophic regression
- Phase G:
  - lightweight regression harness implemented
  - npm usability scripts added
- Phase H:
  - endgame weakness fixtures/debug tooling added
  - `small-piece-king-vs-men` identified as probe-suspect
- Phase Q:
  - tiny low-mobility research signal added
  - signal remains experiment-only and `OFF` by default
  - default gate behavior unchanged

## 4. Current Regression Harness Workflow

Fast loop:

1. `npm run test:perft`
2. `npm run gate:ai:report`
3. if a fresh quick run is needed:
   - `npm run gate:ai:quick`

Current harness behavior:

- parses latest quick benchmark JSON
- prints `classification`, `fatalReasons`, `warnings`
- checks `openingBookBypassed=yes`
- current default-OFF gate status:
  - `classification=WARN`
  - `fatalReasons=(none)`

Current hard-stop cases:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`

## 5. Important Known Caveats

- `small-piece-king-vs-men` is warning-only in the harness for now
- reason: probe-suspect known case
- H.4 finding:
  - root `probeSmallEndgame(...)` returns `undefined`
  - many child positions get exact probe scores
  - fallback root search and child exact scoring can diverge

Practical rule:

- do not treat `small-piece-king-vs-men` as a pure eval failure without checking probe/oracle context first

## 6. Known Dangerous Experiments / Regressions

- root override disable/edit experiments
  - especially low-mobility override changes
- hot-path instrumentation inside search/eval
- tiny local eval tweaks without harness checks
- repetition contempt experiments
- broad mobility signal changes

Observed failure style:

- apparently small changes can trigger `998k-999k` tactical collapses in `sac-two-win-three-*`

## 7. Important Scripts

- `scripts/regressionHarness.ts`
- `scripts/evalBreakdownDebug.ts`
- `scripts/endgameWeaknessFixtures.ts`
- `scripts/endgameProbeDebug.ts`
- `scripts/aiBenchmark.ts`
- `scripts/analyzeAiBenchmark.ts`
- `scripts/perftThaiCheckers.ts`

Most-used npm commands:

- `npm run test:perft`
- `npm run gate:ai:report`
- `npm run gate:ai:quick`
- `npm run bench:ai:fresh`

## 8. Important Docs

- `ENGINE_STATUS.md`
- `ENGINE_BASELINE.md`
- `ENGINE_REGRESSION_CHECKLIST.md`
- `ENGINE_PHASE_D_FRESH_BOOK_DESIGN.md`
- `ENGINE_PHASE_F_EVAL_INSTRUMENTATION_DESIGN.md`
- `ENGINE_PHASE_F_EVAL_SIGNAL_GAPS.md`
- `ENGINE_PHASE_G_REGRESSION_HARNESS_DESIGN.md`
- `ENGINE_PHASE_H_ENDGAME_WEAKNESS_INSPECTION.md`

## 9. Recommended Next Phases

Safest likely next areas:

- deeper probe/oracle diagnosis for `small-piece-king-vs-men`
- optional root-probe diagnostics refinement, still debug-only
- small eval experiments only behind the existing disabled framework
- inspect why the low-mobility research signal affects `quiet-hanging-piece-p1` before any further tuning
- fresh opening book integration only after the fresh scaffold remains isolated and validated

Avoid next:

- broad search changes
- casual root override edits
- broad mobility/endgame rewrites

## 10. Current Stable Workflow For Experiments

Recommended discipline:

1. read the relevant phase doc first
2. keep the change tiny and isolated
3. keep it disabled by default when possible
4. run `npm run test:perft`
5. run `npm run gate:ai:report`
6. if needed, run a fresh quick benchmark
7. inspect named weak cases before trusting aggregate percentages
8. revert the smallest behavior-risk change immediately if a catastrophic signal appears

## Summary

The project is in a careful research/tuning phase, not a broad refactor phase.

Latest Phase Q result:

- low-mobility research signal exists in the eval experiment framework
- it is `OFF` by default and not promoted
- default gate remains `WARN` with `fatalReasons=(none)`
- experiment improved `low-mobility-squeeze` from `drop=1094` to `drop=807`, but did not fix it
- `quiet-hanging-piece-p1` worsened / shifted into a repeated medium miss across `easy` / `hard` / `expert`
- safest next step is to inspect why that signal affects `quiet-hanging-piece-p1` before tuning further

The safest mindset is:

- docs first
- debug before tuning
- one tiny experiment at a time
- trust catastrophic named-case regressions more than small aggregate drift
