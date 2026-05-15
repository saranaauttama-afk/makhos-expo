# Experiment Results

## Result Template

### YYYY-MM-DD - Experiment Name

- Status:
- Scope:
- Flag:
- Baseline gates:
- Experiment-on gates:
- Protected cases:
- Decision:
- Revert:
- Next:

## Recorded Results

### 2026-05-12 - AL.2 near-horizon capture extension

- Status: `rejected`
- Scope: narrow search experiment in `src/coreClaude/search/alphabeta.ts`
- Flag: `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`
- Baseline gates:
  - `npm.cmd run test:perft` passed
  - clean-OFF repeat showed `sac-two-win-three-p1` still unstable
- Experiment-on gates:
  - `npm.cmd run test:perft` passed
  - experiment-ON repeat made the protected tactical case worse
- Protected cases:
  - `sac-two-win-three-p1`: clean-OFF `1/3` catastrophic repeat, experiment-ON `2/3` catastrophic repeat
  - `sac-two-win-three-p2`: remained clean
- Decision: reject
- Revert: reverted only the AL.2 code in `src/coreClaude/search/alphabeta.ts`
- Next: do not retry near-horizon capture extension unless search architecture changes significantly

### 2026-05-12 - Root tactical move ordering only

- Status: `aborted`
- Scope: root-only move ordering experiment in `src/coreClaude/search/alphabeta.ts`
- Flag: `MAKHOS_ENABLE_ROOT_TACTICAL_MOVE_ORDERING`
- Baseline gates:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\ai\run-engine-gate.ps1`
  - `npm.cmd run test:perft` passed
  - clean-OFF repeat worsened the protected baseline on `sac-two-win-three-p1`
- Experiment-on gates:
  - not run
  - autonomous loop stopped before experiment-ON because the clean-OFF protected baseline worsened
- Protected cases:
  - `sac-two-win-three-p1`: catastrophic in `2/3` clean-OFF repeats
  - `sac-two-win-three-p2`: clean in all `3/3` clean-OFF repeats
- Decision: abort this experiment attempt and stop the autonomous run
- Revert: reverted only the root tactical move ordering code in `src/coreClaude/search/alphabeta.ts`; `npm.cmd run test:perft` passed after revert
- Next: do not continue queue execution until the clean-OFF protected baseline is stable enough to compare experiments safely

## Baseline Stability Diagnostics

### 2026-05-13 - Clean-OFF protected-case stability check

- Status: `diagnostic`
- Scope: clean-OFF repeated gate only, no experiment flags
- Repeat count: `6` runs total
- Commands:
  - `npm.cmd run gate:ai:repeat`
  - `npm.cmd run gate:ai:repeat`

- Protected-case outcomes by run:
  - `batch1-run1`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch1-run2`
    - `sac-two-win-three-p1`: catastrophic
    - `sac-two-win-three-p2`: clean
  - `batch1-run3`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run1`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run2`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean
  - `batch2-run3`
    - `sac-two-win-three-p1`: clean
    - `sac-two-win-three-p2`: clean

- Ratios:
  - `sac-two-win-three-p1`: clean `5/6`, catastrophic `1/6`
  - `sac-two-win-three-p2`: clean `6/6`, catastrophic `0/6`

- Key diagnostic observation:
  - on the catastrophic `p1` run, the benchmark `oracleMove` itself changed materially versus the clean runs
  - clean runs saw `oracleMove` variants such as `7->2K` or `8->4K`
  - the catastrophic run switched to `10->28` with near-`999k` score drops

- Suspected instability cause:
  - primary: oracle/search time-budget instability in the benchmark path
  - secondary: nondeterministic search/order behavior inside the oracle/engine interaction
  - not the main suspect:
    - stale benchmark artifact reuse
    - pure `small-piece-king-vs-men`-style oracle noise on the protected case

- Supporting evidence:
  - all six runs were freshly generated and saved with distinct timestamps
  - `sac-two-win-three-p2` remained stable across all six runs
  - `sac-two-win-three-p1` flipped only when the benchmark oracle target changed
  - `scripts/aiBenchmark.ts` uses a time-limited oracle search (`ORACLE_MS=1500`, `ORACLE_DEPTH=11` in quick mode), so oracle output can vary by run

- Worktree note:
  - `src/coreClaude/search/alphabeta.ts` still shows modified in `git status`
  - `git diff -- src/coreClaude/search/alphabeta.ts` is empty
  - `git ls-files --eol` reports `i/lf` and `w/mixed`
  - with `core.autocrlf=true`, this strongly suggests line-ending normalization / mixed-EOL worktree state rather than a live content diff

- Decision:
  - keep the autonomous queue paused
  - do not resume item 2 yet

- Next:
  - prefer diagnostics/tooling to stabilize or explain the protected-case benchmark oracle path before resuming autonomous experiments

### 2026-05-13 - BD1 manual protected benchmark oracle diagnostics

- Status: `diagnostic`
- Scope:
  - protected benchmark oracle tracing for `sac-two-win-three-p1` / `sac-two-win-three-p2`
  - clean-OFF only
  - no experiment flags
  - no runtime tuning
- Commands:
  - `node .tmp/protected-oracle-debug/scripts/protectedOracleDebug.js sac-two-win-three-p1 6`
  - `node .tmp/protected-oracle-debug/scripts/protectedOracleDebug.js sac-two-win-three-p2 4`
  - `npm.cmd run gate:ai:repeat`

- Direct oracle debug evidence:
  - `p1`:
    - `oracleRoot(...)` used `fallbackIterativeDeepening` in all `6/6` runs
    - `timedOut=true` in all `6/6` runs
    - fallback oracle move flipped between `7->2K` and `8->4K`
    - observed fallback depths ranged from `0` to `8`
    - `override=forced recapture trap` appeared in multiple runs
  - `p2`:
    - `oracleRoot(...)` used `fallbackIterativeDeepening` in all `4/4` runs
    - `timedOut=true` in all `4/4` runs
    - fallback oracle move stayed stable on `15->18`

- Fresh clean-OFF repeat evidence:
  - repeat batch result: `WARN / FAIL / WARN`
  - `sac-two-win-three-p1`:
    - run1: `0/0/0/0`
    - run2: `0/614613/0/614613`
    - run3: `0/0/0/0`
  - `sac-two-win-three-p2`:
    - run1: `0/0/0/0`
    - run2: `0/0/0/0`
    - run3: `0/0/0/0`
  - critical run detail:
    - in repeat run2, `p1` oracle flipped to `8->4K`
    - `normal` and `expert` then chose `7->2K`, producing the catastrophic drops
    - `hard` matched the flipped oracle on `8->4K`

- Root-cause ranking:
  1. timeout-limited oracle instability in `oracleRoot(...)` fallback search
  2. root override interaction, especially `forced recapture trap`, amplifying timeout-sensitive oracle choice on `p1`
  3. aspiration / shallow-root verification sensitivity under incomplete search
  4. transposition / history-order sensitivity as a secondary contributor
  5. pure nondeterministic move ordering is unlikely:
     - oracle path is deterministic in code for `diversifyRoot=false`
     - `p2` remains stable under the same fallback timeout conditions

- Additional worktree evidence:
  - `src/coreClaude/search/alphabeta.ts` still shows modified in `git status`
  - `git diff -- src/coreClaude/search/alphabeta.ts` is empty
  - `git status --porcelain=v2` shows `.M`
  - `git ls-files --eol` reports `i/lf` and `w/mixed`
  - interpretation: line-ending normalization / mixed-EOL worktree state, not a live textual runtime diff

- Decision:
  - keep the autonomous queue paused
  - do not resume item 2
  - do not retry AL.2
  - do not launch new search experiments until the benchmark oracle confidence on `p1` improves

- Next:
  - BD2 should focus on oracle confidence diagnostics, not tuning
  - recommended BD2 direction:
    - compare `p1` oracle behavior at higher oracle budgets
    - add benchmark-side shadow reporting of raw fallback result versus final override result
    - define a queue-resume threshold based on protected-case oracle stability, not only final pass/fail labels

### 2026-05-13 - BD2 protected oracle budget / override diagnostics

- Status: `diagnostic`
- Scope:
  - protected benchmark oracle budget sweep for `sac-two-win-three-p1`
  - benchmark-side shadow reporting only
  - no runtime tuning
  - no autonomous queue activity
- Script:
  - `scripts/protectedOracleBudgetDebug.ts`
- Command:
  - `node .tmp/protected-oracle-budget-debug/scripts/protectedOracleBudgetDebug.js sac-two-win-three-p1`

- Budget comparison:
  - `1500 ms`
    - repeats: `3`
    - raw fallback moves: `7->3K | 7->2K | 7->2K`
    - final oracle moves: `8->4K | 7->2K | 8->4K`
    - timed out: `3/3`
    - override changed move: `2/3`
    - forced recapture trap participated: `2/3`
    - final vectors: `998745/998896/998747/998896 | 0/0/0/0 | 0/0/0/0`
  - `3000 ms`
    - repeats: `3`
    - raw fallback moves: `7->2K | 7->3K | 7->3K`
    - final oracle moves: `7->2K | 7->3K | 7->3K`
    - timed out: `3/3`
    - override changed move: `0/3`
    - forced recapture trap participated: `0/3`
    - final vectors: `0/0/0/0 | 998745/998894/998894/999186 | 0/0/0/0`
  - `5000 ms`
    - repeats: `3`
    - raw fallback moves: `7->2K | 20->16 | 7->2K`
    - final oracle moves: `7->2K | 20->16 | 7->2K`
    - timed out: `3/3`
    - override changed move: `0/3`
    - forced recapture trap participated: `0/3`
    - final vectors: `0/0/0/0 | 998756/998905/998905/998905 | 0/0/0/0`
  - `10000 ms`
    - repeats: `2`
    - raw fallback moves: `31->26 | 7->3K`
    - final oracle moves: `31->26 | 8->4K`
    - timed out: `2/2`
    - override changed move: `1/2`
    - forced recapture trap participated: `1/2`
    - final vectors: `999056/384656/998457/998745 | 998745/998448/998596/0`

- Interpretation:
  - raising oracle budget did **not** make `p1` stable
  - all oracle-root fallback runs still reported `timedOut=true`
  - instability persisted even in runs with no override change
  - root override is a secondary amplifier, not the sole cause
  - budget increase alone is not a sufficient stabilization plan for protected-case oracle confidence

- Override vs fallback comparison:
  - override changed the final oracle move at `1500 ms` and in one `10000 ms` run
  - at `3000 ms` and `5000 ms`, the final oracle matched the inferred raw fallback every run
  - despite that, catastrophic final protected-case results still occurred at both `3000 ms` and `5000 ms`
  - conclusion: fallback root search instability is primary; override sensitivity is real but secondary

- Decision:
  - keep the autonomous queue paused
  - do not treat a simple oracle-budget increase as a fix
  - do not pin a protected-case oracle move yet without a higher-confidence reference

- Next:
  - BD3 should separate raw fallback search from final override behavior more explicitly
  - likely next direction:
    - add a diagnostic-only higher-confidence oracle reference path
    - compare raw no-override root choice against final override root choice
    - keep `sac-two-win-three-p1` classified as protected-with-repeat / oracle-sensitive until confidence improves

### 2026-05-13 - BD3 high-confidence root-move reference diagnostics

- Status: `diagnostic`
- Scope:
  - independent root-move scoring for `sac-two-win-three-p1`
  - diagnostics only
  - no runtime tuning
  - no benchmark default changes
- Script:
  - `scripts/protectedRootReferenceDebug.ts`
- Command:
  - `node .tmp/protected-root-reference-debug/scripts/protectedRootReferenceDebug.js 2`

- Setup:
  - quick oracle comparison path:
    - root fallback / override at `1500 ms`, depth cap `11`
  - independent reference path:
    - score each legal root move separately at `10000 ms`, depth cap `15`

- Quick-oracle comparison:
  - pass1:
    - raw fallback: `31->26`
    - final override move: `8->4K`
    - override: `forced recapture trap`
  - pass2:
    - raw fallback: `7->2K`
    - final override move: `7->2K`
    - override: none

- Independent root-move ranking:
  - pass1:
    - `7->3K` score=`999992`, depth=`15`, timedOut=`false`
    - `20->16` score=`999988`, depth=`15`, timedOut=`false`
    - `31->26` score=`999988`, depth=`15`, timedOut=`false`
    - `7->2K` score=`999988`, depth=`15`, timedOut=`false`
    - `31->27` score=`999984`, depth=`15`, timedOut=`false`
    - `8->4K` score=`361`, depth=`1`, timedOut=`true`
  - pass2:
    - `7->3K` score=`999992`, depth=`15`, timedOut=`false`
    - `8->4K` score=`999990`, depth=`15`, timedOut=`false`
    - `31->26` score=`999988`, depth=`15`, timedOut=`false`
    - `7->2K` score=`999988`, depth=`15`, timedOut=`false`
    - `20->16` score=`999986`, depth=`15`, timedOut=`false`
    - `31->27` score=`999984`, depth=`15`, timedOut=`false`

- Interpretation:
  - `7->3K` was the top move in both high-confidence reference passes
  - there is still a tight winning cluster rather than one overwhelmingly separated move
  - likely top cluster:
    - `7->3K`
    - `7->2K`
    - `31->26`
    - `20->16`
    - sometimes `8->4K`
  - quick oracle fallback/override does not reliably track that stronger ranking
  - `forced recapture trap` override can point away from the stronger reference ordering
  - timeout does not prevent convergence for most top candidates:
    - the top cluster mostly completed at depth `15` without timeout
    - lower-ranked moves still timed out heavily

- Answer to BD3 questions:
  - one move clearly best but fallback misses it?
    - partly
    - `7->3K` was top in both passes, but the margin over nearby winning moves was small
  - multiple moves genuinely close?
    - yes
    - there is a close high-confidence winning band near the top
  - timeout prevents convergence for most candidates?
    - no
    - it mainly affects lower-ranked and some volatile tactical candidates
  - forced trap override points toward or away from the reference?
    - mixed, but in BD3 it can point away from the strongest independent reference

- Decision:
  - keep `sac-two-win-three-p1` protected
  - keep it repeat-required
  - do not downgrade it to warning-only yet
  - do not treat a single quick oracle move as authoritative for this case

- Next:
  - BD4 should focus on benchmark policy, not engine tuning
  - likely options:
    - use a fixed high-confidence reference ranking / band for `p1`
    - keep `p1` protected but classify it as oracle-sensitive
    - separate `p1` from normal autonomous hard-stop policy until benchmark policy is redesigned

### 2026-05-13 - BD4 protected oracle-sensitive benchmark policy

- Status: `policy`
- Scope:
  - docs/policy only
  - no runtime behavior change
  - no benchmark default change yet
  - no autonomous resume

- Policy decision:
  - `sac-two-win-three-p1` remains protected
  - `p1` remains repeat-required
  - `p1` remains oracle-sensitive
  - `p1` should not be judged by a single quick oracle move
  - `p1` should be judged against an accepted reference band

- Provisional accepted reference band for `p1`:
  - `7->3K`
  - `8->4K`
  - `20->16`
  - `31->26`
  - `7->2K`

- Warning policy:
  - warning only if the quick oracle move changes but the final move stays inside the accepted band
  - warning only if repeated runs show band-internal drift without a clear increase in catastrophic outcomes

- Fail policy:
  - fail if an experiment pushes `p1` outside the accepted band repeatedly
  - fail if catastrophic `p1` outcomes increase clearly across repeats
  - fail if the move leaves the band while stronger reference diagnostics still support the band

- Tooling / workflow implication:
  - manual experiments may resume cautiously under this policy
  - autonomous mode should remain paused until tooling/harness logic explicitly understands:
    - accepted-band membership
    - repeated-run confirmation
    - warning-only band drift vs true protected-case regression

- Decision:
  - do not pin a single oracle move in engine runtime
  - do not change default oracle behavior yet
  - do not downgrade `p1` to warning-only overall
  - keep `p1` protected, but interpret it through band-aware repeated-run policy

- Next:
  - either:
    - resume manual isolated search experiments under BD4 policy
    - or encode `p1` accepted-band logic into harness/tooling before any autonomous restart

### 2026-05-13 - S1 manual root tactical ordering

- Status: `rejected`
- Scope:
  - narrow root-only move ordering experiment
  - no eval tuning
  - no depth/budget change
- Flag:
  - `MAKHOS_ENABLE_S1_ROOT_TACTICAL_ORDERING`

- Implemented rule:
  - root-only quiet-move ordering bias using existing one-ply tactical signals
  - formula combined:
    - negative immediate capture risk
    - promotion bonus
    - sound forced-trap bonus
    - forced-recapture bonus

- Baseline gates:
  - `npm.cmd run test:perft` passed
  - clean-OFF `gate:ai:report` = `WARN`, no fatal reasons
  - clean-OFF `gate:ai:repeat`:
    - `p1`: `0/0/157/0`, `0/0/0/0`, `0/0/0/0`
    - `p2`: clean in all `3/3` runs
    - `low-mobility-squeeze`: worst expert miss `516`

- Experiment-on gates:
  - `npm.cmd run test:perft` passed
  - experiment-ON `gate:ai:report` = `WARN`, no fatal reasons
  - experiment-ON `gate:ai:repeat`:
    - `p1`: `0/0/0/0`, `0/0/0/0`, `0/357/0/0`
    - `p2`: clean in all `3/3` runs
    - `low-mobility-squeeze`: worst expert miss `1568`
    - `quiet-hanging-piece-p1`: clean in all `3/3` runs

- Protected-case review:
  - `p1` stayed inside the BD4 accepted band across repeats
  - `p2` stayed clean

- Decision:
  - reject
  - reason:
    - protected cases were acceptable
    - but `low-mobility-squeeze` worsened materially under the flag
    - OFF worst expert miss: `516`
    - ON worst expert miss: `1568`

- Revert:
  - reverted only the S1 code in `src/coreClaude/search/alphabeta.ts`
  - `npm.cmd run test:perft` passed after revert

- Next:
  - do not retry the same root quiet-move ordering formula
  - if Phase S continues, choose a different isolated search idea

### 2026-05-13 - S2 quiescence capture-ordering tie-break

- Status: `inconclusive`
- Scope:
  - narrow quiescence-only capture-ordering tie-break experiment
  - no eval tuning
  - no root ordering change
  - no broad search rewrite
- Flag:
  - `MAKHOS_ENABLE_S2_QUIESCENCE_CAPTURE_ORDERING`

- Implemented rule:
  - preserve existing quiescence capture ordering first:
    - recapture priority via `lastCapSquare`
    - larger capture length
  - add tie-break only when those are equal:
    - prefer child position with lower immediate opponent capture risk
    - optional final tie-break: promotion capture

- Baseline gates:
  - `npm.cmd run test:perft` passed
  - clean-OFF `gate:ai:repeat`:
    - `p1`: `0/0/0/0`, `998897/999642/998896/999196`, `998896/998895/998896/998896`
    - `p2`: clean in all `3/3` runs
    - `low-mobility-squeeze`: worst expert miss `956`
    - `quiet-hanging-piece-p1`: clean in all `3/3` runs

- Experiment-on gates:
  - `npm.cmd run test:perft` passed
  - experiment-ON `gate:ai:repeat`:
    - `p1`: `998897/998898/998898/999480`, `0/0/0/28`, `0/0/0/0`
    - `p2`: clean in all `3/3` runs
    - `low-mobility-squeeze`: clean in all `3/3` runs
    - `quiet-hanging-piece-p1`: `402/423/425/0`, `0/0/0/0`, `0/0/0/0`

- Protected-case review:
  - `p2` stayed clean `3/3`
  - `p1` remained volatile under both OFF and ON runs
  - `p1` did not provide enough confidence to treat the experiment as safe/promising

- Monitored-case review:
  - `low-mobility-squeeze` improved under the flag
  - `quiet-hanging-piece-p1` picked up collateral noise:
    - OFF clean `3/3`
    - ON one noisy run `402/423/425/0`

- Decision:
  - inconclusive
  - reason:
    - `p2` stayed clean
    - `low-mobility-squeeze` improved
    - but `p1` remained too volatile for confidence
    - and `quiet-hanging-piece-p1` showed collateral noise under the flag

- Revert:
  - reverted only the S2 code in `src/coreClaude/search/alphabeta.ts`
  - `npm.cmd run test:perft` passed after revert
  - no runtime change remains active

- Next:
  - do not promote
  - possible future revisit only with a narrower `quiet-hanging-piece-p1` guard
  - or with a better joint policy for `p1` and `quiet-hanging-piece-p1`
