# ENGINE_MASTER_ROADMAP

This document is the master source of truth for the `makhos-v2` / `makhos-expo` Thai checkers engine project.

Use it as the first file to read in any future ChatGPT, Codex, or Claude session.

---

## 1. PROJECT OVERVIEW

### Current engine direction

The engine is in a cautious research-and-stabilization phase.

Primary direction:

- preserve current runtime strength while improving experimental discipline
- prioritize isolated search or gameplay-strength experiments over tiny heuristic retuning
- improve benchmark/oracle reliability before resuming autonomous experiment execution

### Current priorities

- stabilize the benchmark/oracle interpretation around protected tactical cases
- preserve strict experiment safety rules
- keep experiments isolated, `OFF` by default, and easy to revert
- improve long-term continuity through docs, tooling, and explicit handoff rules

### What we are NOT doing anymore

- no aggressive tiny heuristic eval tuning loops
- no broad mobility/eval rewrites
- no casual root override editing
- no retry of rejected AL.2 near-horizon capture extension on the current architecture
- no automatic promotion of experiment behavior

### Current branch

- `tuneCodex`

### High-level architecture summary

Core engine areas:

- search:
  - `src/coreClaude/search/alphabeta.ts`
  - `src/coreClaude/search/tt.ts`
  - `src/coreClaude/search/repetition.ts`
  - `src/coreClaude/search/endgameTablebase.ts`
- eval:
  - `src/coreClaude/eval.ts`
- move generation:
  - `src/coreClaude/movegen.ts`
- level/budget policy:
  - `src/coreClaude/search/levelPolicy.ts`
- benchmark and analysis:
  - `scripts/aiBenchmark.ts`
  - `scripts/regressionHarness.ts`
  - `scripts/repeatedRunBenchmarkSummary.ts`

Important search reality:

- quiescence already exists
- tactical depth extensions already exist
- several tactical/root override behaviors already exist

---

## 2. CURRENT REALITY

### Benchmark infrastructure exists

The project already has:

- `test:perft`
- `gate:ai:report`
- `gate:ai:repeat`
- tactical benchmark JSON artifacts in `.tmp/benchmarks/`
- offline harness classification
- repeated-run summaries for protected-case interpretation

### Protected benchmark cases

Primary protected cases:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`

Secondary monitored cases:

- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- `quiet-hanging-piece-p1`
- `small-piece-king-vs-men`

### Stable vs noisy cases

Current reality is mixed:

- `sac-two-win-three-p2` is the cleanest tactical guardrail
- `sac-two-win-three-p1` is protected but baseline-unstable
- `small-piece-king-vs-men` is noisy/oracle-sensitive
- low-mobility cases remain useful but are not clean enough to over-interpret casually

### Heuristic tuning problems

Recent work showed repeated problems with tiny eval-side changes:

- small local heuristic changes can destabilize protected tactical cases
- low-mobility research can help target cases while harming protected tactical stability
- broad or semi-broad heuristic nudges are too fragile relative to the benchmark risk

### AL.1 and AL.2 findings

AL.1 planning result:

- a generic `+1` capture extension was not meaningfully distinct from existing search behavior
- the only distinct slice was the near-horizon `depth == 1` case

AL.2 implementation result:

- near-horizon capture extension was tested narrowly
- protected tactical case `sac-two-win-three-p1` became more unstable under the flag
- AL.2 was rejected
- AL.2 must not be retried unless search architecture changes significantly

### BD1 manual oracle diagnostics findings

Direct protected-case diagnostics and a fresh clean-OFF repeat batch both point at the benchmark oracle path itself as the main source of `sac-two-win-three-p1` volatility.

Key findings:

- `oracleRoot(...)` for both protected cases is taking the fallback iterative-deepening path, not the quiet-low-mobility direct child-scoring path
- in direct `p1` oracle diagnostics, the fallback search timed out on every run and flipped between `7->2K` and `8->4K`
- the same `p1` oracle flips also appeared in fresh clean-OFF benchmark repeats
- `p2` also times out in the oracle fallback path, but still remains stable on `15->18`
- `forced recapture trap` root override appears on many `p1` oracle runs and is a likely amplifier of timeout instability

### BD2 oracle budget / override diagnostics findings

Raising the protected-case oracle budget did **not** stabilize `sac-two-win-three-p1`.

Key findings:

- `p1` remained unstable at `ORACLE_MS=1500`, `3000`, `5000`, and `10000`
- `p1` root oracle search still reported `timedOut=true` on every diagnostic run, even at higher budgets
- higher budgets changed the fallback oracle move, but did not converge it to one stable protected benchmark target
- root override changed the final oracle move on some runs, especially `forced recapture trap`, but instability also persisted on runs with no override change
- this means the problem is not just "too little oracle time at 1500ms"; the fallback root search itself remains unstable on this case

### BD3 high-confidence root-move reference findings

Independent per-move scoring gave a much cleaner picture than quick root-oracle fallback alone.

Key findings:

- a strong per-move reference search at `10000 ms` / depth cap `15` consistently ranked `7->3K` at the top in both BD3 passes
- several moves formed a tight high-confidence winning cluster near the top:
  - `7->3K`
  - `7->2K`
  - `31->26`
  - `20->16`
  - sometimes `8->4K`
- the quick benchmark oracle did **not** reliably choose the top BD3 reference move:
  - one pass ended at quick final `8->4K`
  - one pass ended at quick final `7->2K`
- `forced recapture trap` override can push the quick final oracle move away from the strongest independent reference ordering
- timeout does **not** block convergence for every candidate:
  - the top cluster mostly reached depth `15` without timing out
  - several lower-ranked moves still timed out heavily

### Autonomous loop findings

The autonomous experiment loop was scaffolded in docs/tooling and attempted its first queued item.

Result:

- the first autonomous experiment attempt was aborted before experiment-ON comparison
- reason: the clean-OFF protected baseline itself worsened on `sac-two-win-three-p1`

### Current queue pause reason

The autonomous experiment queue is currently paused because:

- clean-OFF `sac-two-win-three-p1` is not stable enough yet for trustworthy experiment comparison
- current best diagnosis is oracle/search time-budget instability in the benchmark path
- BD1 narrows that further to timeout-limited oracle selection on `sac-two-win-three-p1`, with root override interaction as a likely amplifier
- BD2 adds that simply increasing oracle budget does not remove the instability, so queue resumption should not depend on a quick budget bump alone
- BD3 adds that `p1` should not be judged only by a single quick-root oracle move, because a stronger independent reference shows a tight winning cluster that the quick fallback/override path does not track reliably

---

## 3. PROTECTED / STABLE / NOISY CASE CLASSIFICATION

### `sac-two-win-three-p1`

- protected
- tactical
- baseline-unstable
- repeat-required
- oracle-sensitive
- high-priority guardrail
- do not treat as optional or secondary

Notes:

- clean-OFF baseline sample currently shows `5/6` clean and `1/6` catastrophic
- catastrophic flips appear tied to oracle/search instability as much as engine behavior
- BD4 policy: judge `p1` against an accepted reference band, not one quick oracle move

### `sac-two-win-three-p2`

- protected
- tactical
- stable
- strongest clean guardrail case

Notes:

- remained clean across the recent 6-run clean-OFF diagnostic sample

### `low-mobility-squeeze`

- tactical
- monitored
- unstable / override-sensitive

Notes:

- useful for targeted low-mobility research
- not reliable enough to justify protected-case regressions

### `low-mobility-squeeze-p2`

- monitored
- semi-stable weakness historically
- currently noisy enough to require repeats

Notes:

- often acts as a practical weakness signal
- should not outrank protected tactical cases in decision-making

### `quiet-hanging-piece-p1`

- monitored
- tactical / positional hybrid
- noisy / collateral-sensitive

Notes:

- can move under unrelated experiments
- current hanging-term explanations are incomplete

### `small-piece-king-vs-men`

- informational
- oracle-sensitive
- noisy
- do not use as primary tuning evidence

Notes:

- repeatedly identified as probe/oracle suspect
- still useful as a diagnostic signal, not as a clean win/loss tuning target

---

## 4. HARD SAFETY RULES

- no global depth increase
- no broad search rewrite
- no eval rewrite
- no mass revert
- no automatic promotion
- no retry of rejected AL.2 near-horizon capture extension
- benchmark every experiment
- protect `sac-two-win-three-p1`
- protect `sac-two-win-three-p2`
- revert immediately on protected regression
- keep experiments `OFF` by default
- one isolated experiment at a time
- do not tune around a protected-case failure

---

## 5. BENCHMARK / GATE WORKFLOW

### Core commands

- `npm.cmd run test:perft`
- `npm.cmd run gate:ai:report`
- `npm.cmd run gate:ai:repeat`

### Repeat-run expectations

Single quick artifacts are not enough for experiment decisions.

Use repeated runs to judge:

- protected tactical stability
- whether a failure is persistent or volatile
- whether a result is likely oracle-sensitive, search-sensitive, or clearly real

Protected-case special rule:

- `sac-two-win-three-p1` must be judged by repeated-run band membership, not only by one quick oracle move string

### Clean-OFF vs experiment-ON logic

For any experiment:

1. run clean-OFF gates first
2. confirm protected baseline is acceptable
3. run experiment-ON gates with the explicit feature flag
4. compare protected cases first
5. only then consider secondary monitored cases

### How experiments are evaluated

An experiment is rejected if:

- `test:perft` fails
- a protected case regresses
- default-OFF behavior changes unexpectedly
- clean-OFF baseline itself becomes unsuitable for comparison

For `sac-two-win-three-p1`, protected regression means:

- repeated movement outside the accepted reference band
- or a clear increase in catastrophic repeated-run outcomes

Band-internal oracle drift is warning-level only.

An experiment may be marked promising only if:

- `test:perft` passes
- protected cases remain acceptable
- repeated-run comparison shows repeatable improvement on secondary targets

Even then:

- do not promote automatically

### When queue execution must stop

Stop the autonomous queue when:

- queue is exhausted
- `test:perft` fails after revert
- revert fails
- runtime tree remains dirty unexpectedly after failed revert
- clean-OFF protected baseline worsens
- 3 experiments have been attempted in one autonomous run

---

## 6. KNOWN FINDINGS

- quiescence already exists
- tactical extensions already exist
- generic capture extension duplicated existing logic
- AL.2 near-horizon capture extension worsened `sac-two-win-three-p1` instability
- `sac-two-win-three-p1` clean-OFF baseline currently looks like `5/6` clean, `1/6` catastrophic in the latest manual diagnostic sample
- likely root cause is oracle/search time-budget instability in the benchmark path
- more specifically:
  - `oracleRoot(...)` for `sac-two-win-three-p1` uses fallback iterative deepening under `ORACLE_MS=1500`
  - that fallback search is timing out consistently
  - the timeout-limited root result can flip between `7->2K` and `8->4K`
  - `forced recapture trap` root override frequently participates in the final oracle move
- BD2 showed that larger oracle budgets also remain unstable:
  - `1500`, `3000`, `5000`, and `10000` ms all produced unstable `p1` oracle outputs
  - override participation is a secondary amplifier, not the sole cause
  - budget increase alone is not a sufficient stabilization plan
- BD3 showed a stronger reference pattern:
  - `7->3K` was the top move in both high-confidence reference passes
  - multiple nearby winning moves appear genuinely close
  - quick oracle fallback/override can drift within or away from that top cluster
- BD4 policy conclusion:
  - `sac-two-win-three-p1` remains protected
  - it should be judged against a provisional accepted winning band:
    - `7->3K`
    - `8->4K`
    - `20->16`
    - `31->26`
    - `7->2K`
  - warning only if quick oracle drift stays inside that band
  - fail if experiments push the final move outside the band repeatedly or clearly increase catastrophic repeats
- `src/coreClaude/search/alphabeta.ts` showing `M` with empty textual diff is best explained by line-ending normalization:
  - `git diff -- src/coreClaude/search/alphabeta.ts` is empty
  - `git ls-files --eol` reports `i/lf` and `w/mixed`
  - `core.autocrlf=true`

---

## 7. REJECTED IDEAS

### AL.2 near-horizon capture extension

- status: rejected
- category: narrow search extension experiment
- reason:
  - worsened protected tactical instability
  - specifically made `sac-two-win-three-p1` more unstable under the experiment flag

Rule:

- do not retry AL.2 unless the search architecture changes significantly

---

## 8. ACTIVE ROADMAP

### Phase BD: Benchmark / Oracle Stabilization

Goals:

- understand and reduce protected-case baseline instability
- isolate oracle-path variability from engine-path variability
- make autonomous comparisons trustworthy again

Estimated duration:

- short to medium

Risks:

- diagnostic work may reveal benchmark limitations rather than easy engine fixes
- oracle and engine may both contribute to instability

Expected outputs:

- repeatable protected-case diagnostic protocol
- stronger explanation of `sac-two-win-three-p1` variance
- queue-resume decision criteria

Checklist:

- [x] reproduce additional clean-OFF protected-case samples
- [x] inspect oracle path for `sac-two-win-three-p1`
- [x] compare oracle move changes across runs
- [x] separate engine move instability from oracle target instability
- [ ] define clear resume threshold for autonomous queue
- [x] run BD2 oracle-budget / override diagnostics without changing runtime behavior
- [x] run BD3 shadow no-override / higher-confidence oracle diagnostics
- [x] define BD4 protected-case benchmark policy using high-confidence reference ranking / band
- [ ] encode band-aware `p1` policy into benchmark tooling / harness

### Phase S: Search Strength Experiments

Goals:

- test isolated, `OFF`-by-default search-strength ideas
- prefer ordering and diagnostics over broad depth changes

Estimated duration:

- medium

Risks:

- search changes can interact with existing tactical extensions and root overrides
- protected tactical cases can regress even from small changes

Expected outputs:

- rejected/promising decisions for isolated search ideas
- repeatable revert discipline

Checklist:

- [ ] root tactical move ordering revisit only after BD stabilization
- [ ] capture-only quiescence ordering research
- [ ] recapture-priority ordering research
- [ ] keep experiments isolated and `OFF` by default

### Phase G: Gameplay / Practical Strength

Goals:

- improve practical play quality without unsafe structural rewrites
- focus on decision quality in actual gameplay settings

Estimated duration:

- medium to long

Risks:

- apparent practical gains may be benchmark-noise artifacts
- gameplay tweaks may hide tactical regressions

Expected outputs:

- stronger real-game behavior under controlled checks
- documented practical-strength experiments

Checklist:

- [ ] test practical move-quality ideas after BD stabilization
- [ ] compare against tactical regression harness before trusting gameplay wins
- [ ] avoid hidden depth/budget inflation

### Phase E: Endgame Reliability

Goals:

- improve confidence in endgame oracle/probe interpretation
- separate genuine engine weakness from tablebase/probe inconsistency

Estimated duration:

- medium

Risks:

- endgame diagnostics can be mistaken for eval problems
- oracle/probe sensitivity may remain partly irreducible

Expected outputs:

- cleaner endgame reliability notes
- better classification of informational vs actionable endgame signals

Checklist:

- [ ] continue probe consistency diagnostics
- [ ] avoid using oracle-sensitive cases as primary tuning targets
- [ ] document endgame-specific confidence levels

---

## 9. DETAILED CHECKLIST

### BD1 oracle diagnostics

- [x] rerun protected clean-OFF repeats in controlled batches
- [x] capture `oracleMove` for `sac-two-win-three-p1` across runs
- [x] compare chosen move, oracle move, score drop, depth, elapsed time
- [x] inspect whether oracle path flips between quiet-low-mobility exhaustive scoring and iterative deepening
- [x] identify that the current oracle path is iterative-deepening fallback with repeated timeout behavior
- [x] document that instability is mostly oracle-side first, with engine-side disagreement as a secondary effect
- [x] test whether a higher oracle budget or shadow raw-root diagnostic stabilizes `p1` without changing engine runtime behavior
- [x] separate budget sensitivity from root-override sensitivity with a higher-confidence shadow oracle baseline
- [x] decide that `p1` should use a fixed reference band instead of a single quick oracle move
- [ ] decide whether harness/tooling should treat band-internal drift as warning-only automatically

### Search ordering experiments

- [ ] keep root tactical move ordering paused until BD1 is complete
- [ ] evaluate quiescence ordering ideas only after baseline is trustworthy
- [ ] evaluate recapture-priority ordering only after baseline is trustworthy
- [ ] require `OFF`-by-default flags for any future search experiment

### Gameplay testing

- [ ] preserve tactical harness checks before trusting gameplay improvements
- [ ] avoid using gameplay improvements to excuse protected tactical regressions
- [ ] compare practical move choices only after protected baseline is stable

### Opening / trap ideas

- [ ] keep opening/tactical pattern book work in research/docs mode for now
- [ ] do not broaden opening-book runtime behavior casually
- [ ] preserve forced-trap safeguards already in runtime unless separately justified

### Endgame reliability

- [ ] continue `small-piece-king-vs-men` as informational only
- [ ] inspect probe consistency before touching endgame eval heuristics
- [ ] keep endgame reliability separate from generic tactical tuning

### Benchmark hygiene

- [ ] always distinguish clean-OFF from experiment-ON
- [ ] save repeat-run artifacts when diagnosing protected cases
- [ ] compare protected cases before secondary cases
- [ ] never trust a single quick benchmark artifact
- [ ] record baseline instability before launching autonomous experiments

---

## 10. AUTONOMOUS LOOP STATUS

Current status:

- paused

Why autonomous execution is paused:

- the first autonomous experiment attempt was aborted because the clean-OFF protected baseline itself worsened
- `sac-two-win-three-p1` is not yet stable enough to support fair experiment comparison
- BD4 now defines a workable manual policy, but autonomous tooling still does not encode that policy

When it may resume:

- after BD-phase stabilization work improves confidence in the clean-OFF protected baseline

Conditions required before resuming:

- protected clean-OFF baseline is trustworthy enough to compare against experiment-ON
- no unresolved revert/runtime-tree problems remain
- AL.2 remains excluded from retry
- benchmark tooling must understand `p1` accepted-band logic and repeated-run warning/fail rules

---

## 11. NEXT RECOMMENDED ACTION

Current recommended next task is:

- **manual Phase S search experiments under BD4 policy, or harness-policy tooling**

BD4 is now defined.

The safest next options are:

- keep `p1` protected but repeat-required
- resume manual isolated search experiments only with band-aware repeated-run review
- or encode the new `p1` band-aware policy into benchmark tooling before resuming autonomous mode

---

## 12. SESSION HANDOFF INSTRUCTIONS

For future chats:

- read `ENGINE_MASTER_ROADMAP.md` first
- read `docs/ai/PROTECTED_CASE_POLICY.md`
- do not restart heuristic tuning loops
- preserve protected benchmark policy
- preserve rejected experiment history
- do not resume the autonomous queue until the BD phase stabilizes the protected baseline
- treat `sac-two-win-three-p1` as protected but baseline-unstable
- treat `sac-two-win-three-p1` as accepted-band judged, not single-oracle judged
- treat `sac-two-win-three-p2` as the cleaner stable tactical guardrail
- keep experiments isolated, `OFF` by default, and easy to revert

---

## 13. APPENDIX

### Important files

- `ENGINE_MASTER_ROADMAP.md`
- `ENGINE_STATUS.md`
- `PROJECT_HANDOFF_ENGINE_STATUS.md`
- `CURRENT_TASK.md`
- `docs/ai/AUTONOMOUS_ENGINE_LOOP.md`
- `docs/ai/EXPERIMENT_QUEUE.md`
- `docs/ai/EXPERIMENT_RESULTS.md`
- `docs/ai/PROTECTED_CASE_POLICY.md`
- `docs/ai/phase-al-tactical-extension-research.md`
- `docs/ai/phase-al1-capture-extension-plan.md`
- `docs/ai/phase-al2-near-horizon-capture-extension-result.md`

### Important scripts

- `scripts/aiBenchmark.ts`
- `scripts/regressionHarness.ts`
- `scripts/repeatedRunBenchmarkSummary.ts`
- `scripts/protectedOracleDebug.ts`
- `scripts/protectedOracleBudgetDebug.ts`
- `scripts/protectedRootReferenceDebug.ts`
- `scripts/perftThaiCheckers.ts`
- `scripts/ai/run-engine-gate.ps1`

### Benchmark commands

- `npm.cmd run test:perft`
- `npm.cmd run gate:ai:report`
- `npm.cmd run gate:ai:repeat`
- `powershell -ExecutionPolicy Bypass -File .\scripts\ai\run-engine-gate.ps1`

### Environment flags

Known current/recent flags:

- `MAKHOS_ENABLE_EVAL_EXPERIMENTS`
- `MAKHOS_ENABLE_LOW_MOBILITY_RESEARCH`
- `MAKHOS_LOW_MOBILITY_RESEARCH_SCALE_PCT`
- `MAKHOS_ENABLE_LOW_MOBILITY_EXACT_TIEBREAK`
- `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`
  - rejected experiment history only; do not retry on current architecture
- `MAKHOS_ENABLE_ROOT_TACTICAL_MOVE_ORDERING`
  - aborted attempt history only; do not resume until BD stabilization work is complete

### Docs references

- `docs/ai/AUTONOMOUS_ENGINE_LOOP.md`
- `docs/ai/EXPERIMENT_QUEUE.md`
- `docs/ai/EXPERIMENT_RESULTS.md`
- `docs/ai/PROTECTED_CASE_POLICY.md`
- `docs/ai/EXPERIMENT_LOG.md`
