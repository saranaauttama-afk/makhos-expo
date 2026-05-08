# ENGINE_PHASE_G_REGRESSION_HARNESS_DESIGN

## Scope

Phase G.0 is design only.

No engine behavior, benchmark scoring, or benchmark output is changed here.

The goal is to define a lightweight regression harness that can catch clearly bad tactical regressions quickly, without turning normal benchmark variance into noisy failures.

## 1. Current Benchmark Entry Points / Scripts

Current tactical benchmark flow is centered on `scripts/aiBenchmark.ts`.

Relevant scripts from `package.json`:

- `npm run test:perft`
- `npm run test:tactical`
- `npm run bench:ai`
- `npm run bench:ai:fresh`
- `npm run bench:ai:full`
- `npm run bench:ai:full:fresh`
- `npm run bench:ai:full:tactical`
- `npm run bench:ai:teacher`
- `npm run bench:ai:teacher:fresh`
- `npm run bench:ai:analyze`

Current benchmark-related files:

- `scripts/aiBenchmark.ts`
- `scripts/analyzeAiBenchmark.ts`
- `.tmp/benchmarks/ai-benchmark-*.json` report outputs

Current benchmark characteristics:

- tactical benchmark computes an oracle move and score per case
- each difficulty level records solve rate, blunder rate, elapsed time, depth, nodes, and qnodes
- benchmark reports already serialize case-level `scoreDrop`
- benchmark reports already record `openingBookBypassed=true`
- the quick benchmark path is the normal tuning-time safety check

## 2. Metrics Stable Enough To Gate

These look stable enough to use as primary regression-harness signals:

- perft pass/fail
- case-level catastrophic `scoreDrop`
- per-level blunder rate in `hard` and `expert`
- per-level solve rate in `hard` and `expert`
- presence of specific known catastrophic miss cases
- confirmation that `openingBookBypassed=true`

Why these are stable enough:

- they map directly to tactical correctness or severe tactical quality loss
- they are already exposed by the current benchmark flow
- they were repeatedly useful across prior phases for catching real regressions quickly

Best primary gate metrics:

- `scoreDrop` on named tactical cases
- `hard` / `expert` blunder rate
- catastrophic-case pass/fail

## 3. Metrics Too Noisy To Gate Hard

These should be warnings first, not hard fail signals:

- `easy` solve rate
- `easy` blunder rate
- `normal` solve/blunder drift when the move-quality change is small
- raw `avgMs`
- raw `avgNodes`
- raw `avgQNodes`
- small `avgDepth` drift
- one-off timing spikes on a single run

Why they are noisy:

- lower levels intentionally contain policy noise and weaker move selection
- timing depends on machine load, filesystem warmth, and transient `.tmp` state
- small solve-rate differences can come from a handful of borderline cases

Use them as:

- warning signals
- trend indicators
- tie-break evidence when a larger tactical regression is already suspected

## 4. Known Catastrophic Signatures

Known hard-stop signatures from prior phases:

- `sac-two-win-three-p1` catastrophic drop around `998k-999k`
- catastrophic trap-style drops around `500k-999k`
- `small-piece-king-vs-men` huge drop
- severe regression in `low-mobility-squeeze` when a tiny eval tweak destabilizes tactical choice

Known named tactical cases already present in `scripts/aiBenchmark.ts`:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- opening trap / recapture cases

Known stable benchmark snapshots worth preserving as reference bands:

- Phase A inspection snapshot:
  - `easy 92/8`
  - `normal 97/3`
  - `hard 95/3`
  - `expert 97/3`
- Phase B instrumentation validation snapshot:
  - `easy 95/3`
  - `normal 100/0`
  - `hard 100/0`
  - `expert 97/3`

Interpretation:

- exact percentages may move a little between safe changes
- catastrophic case failures are much more trustworthy than small aggregate drift

## 5. Suggested Lightweight Gate Rules

Recommended harness structure:

1. `precheck`
   - run `npm run test:perft`
   - fail immediately if perft fails

2. `quick tactical safety`
   - run the quick tactical benchmark path
   - parse its JSON report, not only console text

3. `catastrophic case scan`
   - fail immediately on known catastrophic signatures

4. `aggregate guard`
   - apply small set of hard fail thresholds
   - emit warnings separately

Recommended hard fail rules:

- fail if perft fails
- fail if `openingBookBypassed !== true`
- fail if any named catastrophic case exceeds catastrophic drop threshold
- fail if `hard` blunder rate exceeds release-quality limit
- fail if `expert` blunder rate exceeds stable upper bound
- fail if `hard` or `expert` solve rate drops materially from the frozen baseline band

## 6. Suggested Warning Vs Fail Thresholds

Recommended fail thresholds:

- case `scoreDrop >= 500000`
  - fail immediately
- `sac-two-win-three-p1` large drop
  - fail immediately
- `hard` blunder rate `> 0` when the target is release-quality
  - fail
- `expert` blunder rate `> 0.03`
  - fail
- `hard` or `expert` score drop on any case `> 100000`
  - fail

Recommended warning thresholds:

- `normal` blunder rate drift of `+0.02` or more vs baseline snapshot
- `easy` or `normal` solve-rate drift of `-0.05` or more
- `p95Ms` drift beyond historical quick-run range
- `overrides` count or timeout count rising materially without immediate tactical failure

Recommended philosophy:

- use named-case catastrophic drops as the strongest fail signal
- use aggregate level metrics as confirmation, not the only guard
- use timing mostly as warning-only in the quick harness

## 7. How To Keep Runtime Fast

Keep the harness lightweight by:

- using quick tactical mode only for experiment-time safety
- reusing the existing benchmark report output instead of building a second benchmark runner
- reading the latest JSON report rather than re-running analysis repeatedly
- keeping the gate focused on a small set of high-signal metrics
- not running head-to-head games in the lightweight harness
- keeping oracle and benchmark settings exactly as the current quick flow defines them

Recommended fast loop:

1. clear `.tmp` before critical runs
2. run perft
3. run quick tactical benchmark
4. parse latest JSON report
5. print a compact pass/warn/fail summary

## 8. How To Avoid Flaky Failures

Recommended anti-flake rules:

- gate on catastrophic miss signatures first
- do not hard-fail on small `easy` / `normal` drift alone
- do not hard-fail on raw timing alone in quick mode
- compare against stable bands, not a single exact percentage point
- separate `warning` from `fail`
- prefer named-case thresholds over broad average thresholds

If a run is suspicious but not catastrophic:

- rerun once with `.tmp` cleared
- only escalate if the same warning persists or worsens

Do not rely on:

- one isolated timing spike
- one tiny solve-rate wobble at `easy`
- one aggregate metric without named-case evidence

## 9. How Future Eval Experiments Should Use The Harness

Recommended workflow for future eval experiments:

1. keep the experiment isolated and disabled by default
2. enable only one tiny eval change at a time
3. run `npm run test:perft`
4. run the quick tactical regression harness
5. inspect named miss cases first
6. only proceed to broader tuning if the harness stays clean

Special lesson from Phase F.5:

- a tiny local eval signal improvement can still cause catastrophic tactical failure elsewhere
- therefore the harness should treat catastrophic case regressions as more important than local improvement in one diagnostic case

## 10. Rollback / Recovery Workflow

Recommended recovery path after a regression:

1. stop stacking new changes
2. identify the smallest recent behavior change
3. revert only that change first
4. rerun perft
5. rerun quick tactical harness
6. if the regression remains unclear, restore the touched behavior-critical file to the last stable committed state
7. keep only documentation or passive inspection changes when safe

Recommended rollback priority:

- behavior code first
- benchmark/report formatting only if it affects timing or hot paths
- docs last

## Proposed Future Harness Shape

When Phase G implementation begins later, the safest first version is:

- one lightweight script
- reads the existing latest benchmark JSON
- checks a small set of named fail/warn rules
- exits nonzero only on clear safety failures

Suggested future responsibilities:

- verify `openingBookBypassed`
- scan named catastrophic cases
- check `hard` / `expert` blunder thresholds
- print warnings for softer drift

Do not add in the first implementation:

- CI integration
- new benchmark scoring logic
- automatic rerun loops
- heavy statistical baselining

## Parser Scaffold Status

Phase G.1 adds a standalone parser scaffold:

- `scripts/regressionHarness.ts`

Current scaffold behavior:

- reads an existing benchmark JSON report offline
- extracts level table fields:
  - `solve`
  - `blunder`
  - `avgMs`
  - `avgDepth`
- scans case-level `scoreDrop`
- detects catastrophic drops around the current hard-stop range
- groups repeated tactical failures across levels
- prints a compact non-blocking classification:
  - `PASS`
  - `WARN`
  - `FAIL`
- prints a compact tactical-miss summary

Important scope limit:

- this parser does not block anything yet
- it is a reporting helper only
- it does not modify benchmark output format
- it does not add runtime benchmark overhead

Phase G.2 refines the parser thresholds so the scaffold is useful without being too flaky.

Current parser output now prints:

- `classification`
- `fatalReasons`
- `warnings`

## Fatal Cases

Current fatal signatures in the parser scaffold:

- catastrophic `sac-two-win-three-p1` drop
- catastrophic `sac-two-win-three-p2` drop
- catastrophic `low-mobility-squeeze` drop
- catastrophic `low-mobility-squeeze-p2` drop
- repeated `999k`-style catastrophic failures across levels
- `openingBookBypassed=no`

Why `sac-two-win-three` remains `FAIL`:

- prior phases repeatedly showed it as a high-signal tactical safety canary
- when it fails catastrophically, the regression is usually real and severe
- it correlates with the kind of tactical drift we want the harness to catch immediately

## Known Warning Cases

Current warning-only special case:

- `small-piece-king-vs-men`

Why `small-piece-king-vs-men` is `WARN` for now:

- it already appears as a known weak/noisy case in otherwise stable quick benchmark runs
- treating it as fatal right now would make the harness too noisy for everyday experiment use
- it is still surfaced prominently so the weakness is visible, but it does not dominate classification by itself

Other current warnings:

- repeated non-fatal miss cases
- non-catastrophic misses under roughly `1000` cp
- elevated `hard` blunder rate
- elevated `expert` blunder rate

This split is intentional:

- fatal = clear tactical safety break
- warning = real concern, but not yet strong enough to halt all local experimentation

## Known Limitations

Current scaffold limitations:

- it parses the JSON benchmark artifact, not raw console text
- it uses simple heuristic thresholds, not a historical baseline database
- it does not yet compare one report against another report directly
- it does not yet distinguish "expected weak at easy" from "unexpected weak at easy" beyond simple heuristics
- it does not yet export machine-readable pass/warn/fail metadata beyond console output
- it does not yet integrate with `bench:ai:analyze`

These limitations are intentional for the first implementation because they keep the harness:

- offline
- fast
- easy to inspect
- easy to revert

## Next Implementation Ideas

Safest next steps after the parser scaffold:

1. add optional baseline-file comparison
   - compare current report against one frozen stable report

2. add explicit named-case rules table
   - mark known catastrophic cases separately from softer warnings

3. add machine-readable output mode
   - JSON summary for future tooling, still offline

4. only after that, consider optional nonzero exit on clear fail classification
   - still local only
   - still no CI by default

## Summary

The regression harness should be built around the current quick tactical benchmark, not beside it.

The highest-signal safety gates are:

- perft correctness
- known catastrophic case failures
- `hard` / `expert` blunder regressions
- confirmation that tactical benchmark remains book-free

The harness should stay fast, conservative, and anti-flaky:

- fail on catastrophic signatures
- warn on softer aggregate drift
- keep implementation thin by reusing the existing benchmark JSON report flow
