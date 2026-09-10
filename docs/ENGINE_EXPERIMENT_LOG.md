# Makhos Engine Experiment Log

This is the permanent evidence log for engine-strength work on `engine-mainline`.

Do not erase failed experiments. A failed idea can prevent future chats/agents from repeating the same work.

## Rules for logging

For every change intended to affect playing strength or search performance, record:

- date,
- baseline commit,
- candidate commit,
- hypothesis,
- exact changed files/features,
- test environment,
- correctness results,
- puzzle/tactical results,
- head-to-head results,
- performance results,
- conclusion: KEEP / REVERT / NEEDS MORE DATA.

Do not promote from one good puzzle or a few self-play games.

---

# Canonical baselines

## Teacher Baseline v0

**Status:** FROZEN — correctness passed; known benchmark and fixture limitations below

**Branch:** `engine-mainline`

**Starting checkpoint:** `db28e143bb06fb8a35a643877a055c5df9770020`

Documentation commits after checkpoint do not count as engine-strength changes.

### Environment

- Date: 2026-09-07 UTC
- OS: Linux 6.18.35, x86_64
- CPU: Intel Xeon Platinum 8370C @ 2.80 GHz (3 logical CPUs available)
- RAM: 17 GiB
- Node: v24.15.0; npm 11.4.2
- Commit tested: `54278041642aa4aa6abca3e26f9085bdb6d301b1` (engine code is unchanged from starting checkpoint `db28e143bb06fb8a35a643877a055c5df9770020`)
- Search/time settings: quick benchmark time scale 0.08, oracle 1,500 ms/depth 11; full tactical time scale 1, oracle 5,000 ms/depth 15; opening book bypassed. Puzzle run used depth <= 8 and 10,000 ms per fixture. Small-endgame probes can report a DTM deeper than that search-depth limit.

### Correctness

| Test | Result | Notes |
|---|---|---|
| `npm run test:rules` | PASS | 3,501 checks |
| `npm run test:perft` | PASS | 8/8 positions; initial-position counts 7, 49, 392 at depths 1-3 |
| `npm run test:tactical` | PASS | 8 checks |
| regression harness | WARN | No fatal reasons; known `small-piece-king-vs-men` oracle/probe instability at every level, 500,190 score drop; hard blunder 3% in quick run |
| `npm ci --ignore-scripts --offline` + `npx tsc --noEmit` | PASS after harness repair | Clean install exposed missing explicit Node types plus stale errors in three diagnostic scripts; measurement-only fixes made |

### Search benchmark

Production full-budget tactical results (39 cases per level, 156 searches; no opening book):

| Level | Solved / blunder | Avg depth | Avg nodes + qnodes | Approx. NPS | Avg / p95 think time |
|---|---:|---:|---:|---:|---:|
| easy | 100% / 0% | 2.7 | 83,589 | 192,590 | 434 / 1,203 ms |
| normal | 100% / 0% | 3.4 | 233,149 | 242,436 | 962 / 2,505 ms |
| hard | 100% / 0% | 4.2 | 433,873 | 269,929 | 1,607 / 3,282 ms |
| expert | 100% / 0% | 5.1 | 946,510 | 284,858 | 3,323 / 6,684 ms |

The full release gate passed all four levels. This was the existing `--full --tactical-only --fresh` production-budget variant; the full 72-game difficulty ladder was intentionally not treated as a candidate-vs-baseline strength test because Phase 0 has no candidate. The quick run scored 97% solved / 3% blunder at every level. Quick averages (easy through expert) were 89/132/157/305 ms, depths 1.5/2.0/2.8/3.7, and combined-node NPS approximately 58k/50k/101k/160k.

### Puzzle baseline

Historical report `PUZZLE_BASELINE.md` counted 1/14, but fixture validation proves that number is not a valid accuracy metric. Ten expected first moves are illegal under the current Thai-rules move generator (usually because a mandatory/maximal capture exists, or because the destination is not legal). Only four fixtures have a legal expected first move, none has authoritative provenance, and `solution` contains prose rather than a replayable line. All 14 must therefore remain a development/diagnostic set; there is no holdout set and no measurable full-line accuracy yet.

| Dataset | Correct | Total | Full-line solved | Avg depth | Avg nodes | Avg time |
|---|---:|---:|---:|---:|---:|---:|
| development (legally scoreable subset) | 1-2 | 4 | unavailable | 3.3-3.7 across all 14 | 1,958,049-2,223,799 across all 14 | 6,984-7,538 ms across all 14 |
| holdout | unavailable | 0 | unavailable | — | — | — |

`promo-01-race-to-king` (10->6) passed both runs. `trap-01-bait-sacrifice` (expected 26->23, chose 26->22) and `trap-02-double-bait` (expected 27->23, chose 21->16) failed both. `endgame-04-triangulation` (expected 18->22) exposed wall-clock instability: it chose 18->0 in the first run and the expected 18->22 in the verification run, both at reported depth 8. Thus observed validated accuracy was 1/4 then 2/4, not a single reproducible strength number. The other ten fixtures are explicitly excluded, not counted as engine failures. All positions returned a legal move. Mean combined main/qsearch NPS was 219,142 then 233,399. The stable command is `npm run test:puzzles`; it records legality, nodes, qnodes, NPS, depth and elapsed time in `.tmp/puzzles/puzzle-results.json`.

### Head-to-head

Not applicable until the reproducible candidate-vs-baseline A/B harness is established. The existing full benchmark's difficulty ladder compares level policies, not two engine revisions.

### Notes

- Active engine inventory: position/bitboards, Thai move generation, evaluation, PVS/alpha-beta iterative deepening, TT/Zobrist/repetition, level policy, opening book/fresh-book scaffold and small-endgame tablebase under `src/coreClaude/`.
- `src/coreClaude/search/timeManager.ts` is a scaffold: repository search finds imports only in `scripts/testTimeManager.ts`. Production `iterativeDeepening` uses the caller's fixed `timeMs` deadline plus its own best-move-stability early stop; it does not call `allocateTime` or `calculateTimeBudget`.
- Puzzle fixture quality is the largest measurement defect: ten illegal answers, no source citations, no executable principal variations, no holdout split, and one of four scoreable fixtures changed answer across consecutive identical runs. Do not tune against these answers.
- The quick benchmark oracle is time-sensitive: `small-piece-king-vs-men` is catastrophic at quick budgets but the full-budget run solved all cases. A fixed-depth/fixed-node mode and a more trustworthy exact endgame oracle are Phase 1 priorities.
- Teacher v0 is a competent classical tactical engine on the repository's production-budget benchmark (100% there), but that suite is not independent evidence of general playing strength. Its observable weaknesses are shallow-budget endgame instability, benchmark-oracle sensitivity, missing trustworthy puzzle/holdout evidence, and wall-clock nondeterminism.
- Phase 1 should first harden rules/search semantics and deterministic measurement: audit draw-state hashing/TT and mandatory-capture qsearch, verify tablebase/oracle agreement, add fixed-depth/fixed-node controls, and replace or independently validate puzzle fixtures before any evaluation/search/pruning strength tuning.

---

## EXP-2026-001 — Phase 1A deterministic search correctness

**Status:** KEEP (correctness/measurement only; not a strength promotion)

**Date:** 2026-09-08

**Baseline commit:** `e0f6cc3` (Teacher v0 engine checkpoint remains `db28e143`)

**Candidate commit:** `6dc23f1`

### Hypothesis

Fixed-work search and draw-complete TT keys make repeated measurements stable and prevent cached scores from crossing positions with different draw outcomes, without changing evaluation weights or pruning thresholds.

### Bugs found and fixed

1. `hashPosition` correctly served repetition identity but was also used as the TT key, so the TT could reuse a score across different `halfmoveClock` values and different prior repetition histories. Search now uses dual search-state keys containing board, side, inactivity clock, and the complete repetition count multiset; board-only hashes remain unchanged for repetition detection and opening-book identity.
2. The small-endgame memo key contains only the current board's repetition count, not the complete history. A full-history key was tested but caused state-space explosion and wall-clock oracle timeouts, so that implementation was reverted. This remains a known issue requiring a deterministic solver redesign and dedicated regression oracle.
3. Quiescence correctly forbade stand-pat when a capture was mandatory, but it neither checked nor updated threefold repetition during capture continuations. Qsearch now carries the same repetition state as main search and pushes/pops every forced capture.
4. Search exposed only a callback PV capped at ten moves and returned no PV in `SearchResult`. Results now contain a legal root-first PV (up to the search ply cap), and the benchmark records PV length.

### Deterministic guarantees

- `fixedDepthSearch` ignores wall time, disables the budgeted root tablebase probe and adaptive-time early stop, resets history heuristics, and completes exactly the requested nominal iterative-deepening depth.
- `fixedNodeSearch` applies one exact combined main+qsearch budget over all completed iterations and the final partial iteration. It reports `limitReached: "nodes"`, does not report a timeout, and retains the result of the last completed depth.
- `npm run test:search-determinism` repeats each mode five times with a fresh TT and requires identical best move, score, main nodes, qnodes, completed depth, PV, and stopping reason. Phase 1A observed:

| Mode | Runs | Best | Score | Main nodes | Qnodes | Completed depth | PV length |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed depth 4 | 5 | 27→23 | 9 | 291 | 185 | 4 | 4 |
| fixed 5,000 combined nodes | 5 | 27→23 | 21 | 3,165 | 1,835 | 5 | 5 |

### Correctness and benchmark results

| Check | Result |
|---|---|
| Rules | PASS — 3,501 checks |
| Perft | PASS — 8/8; initial 7/49/392 |
| Tactical core | PASS — 8 checks |
| Deterministic suite | PASS — 14 assertions |
| Quick tactical benchmark | gate thresholds pass — Easy/Normal/Expert 97% solved, Hard 100%; 0% blunder at every level |
| Regression harness over quick report | WARN — no fatal reasons; one repeated medium miss plus three sub-1,000-point misses |
| Full tactical benchmark | release gate FAIL at Expert; Easy 100/0, Normal–Expert 97/3 solved/blunder |

The full run's `sac-two-win-three-p1` oracle changed its preferred move relative to the quick run: quick treated 7→2 as equal-best, while full treated that same move as a 999,640-point blunder behind 8→4. This is further evidence that the wall-clock oracle/tablebase path is not deterministic; it is recorded as a known measurement failure, not as proof that Phase 1A made the engine stronger or weaker.

### Nodes, qnodes, depth and PV analysis

The Phase 1A full run averaged, from Easy through Expert, main nodes 72,999 / 180,531 / 326,890 / 750,962 and qnodes 777 / 2,023 / 1,851 / 4,387. Expert qnodes were only about 0.58% of its 755,349 combined nodes, so qsearch is not the reason for the near-million-node cost. The primary explanation is that reported nodes accumulate every iterative-deepening iteration, aspiration retry, and optional root verification, while the reported depth is only the last fully completed *nominal* iteration. Extensions can search individual forced/low-material/tactical branches beyond that nominal depth, and a partial next iteration consumes nodes without raising the reported depth. Full-run average completed depths were 2.5 / 2.9 / 3.4 / 3.5.

The benchmark gained `pvLength`, but the full run above preceded that reporting field. The deterministic canonical checks measured PV lengths 4 and 5 for depth 4 and the 5,000-node run respectively; the follow-up quick run is the first wall-clock report that includes average PV length.

### Known issues

- Wall-clock production results and the endgame oracle remain unsuitable as deterministic correctness gates. The small-endgame memo still summarizes only the current-position repetition count; a complete-history prototype exhausted the oracle time budget. Fixed-work oracle fixtures and a scalable history-sensitive solver key are still required.
- Full repetition context makes TT reuse more conservative and can reduce depth at a fixed time. This is a correctness tradeoff, not a strength claim.
- The broader Phase 1 audits of TT bound/mate normalization and individual pruning mechanisms remain open; Phase 1A does not complete all of Phase 1.
- Existing puzzle fixtures still lack provenance and a holdout set.

### Decision

**KEEP** the correctness and deterministic measurement infrastructure. Do not promote Teacher v0 or claim increased playing strength from this change.

---


## EXP-2026-002 — Phase 1B TT, oracle and pruning correctness audit

**Status:** KEEP (correctness infrastructure; NEEDS MORE DATA for every strength effect)

**Date:** 2026-09-08

**Baseline commit:** `6dc23f1`

**Implementation commit:** `55df3c1e37aa7af7957655ef2301642a914f25eb`

### Scope and constraints

No evaluation weight, pruning margin, reduction table, or promotion threshold changed. Phase 1B adds correctness guards, deterministic oracle infrastructure and independent feature controls. It is not a Teacher strength promotion.

### TT audit and fixes

- **EXACT/LOWER/UPPER:** probe semantics are correct: EXACT returns directly, LOWER raises alpha, UPPER lowers beta, and cutoff returns the decoded TT score. Entries classify against the original alpha/beta window. All three bounds now have direct round-trip fixtures.
- **Empty/collision verification bug fixed:** a zero-filled `Int32Array` fabricated a valid EXACT entry for primary key 0 and verification key 0. TT now has an explicit occupancy bitmap. Tests also cover same-index primary collisions, verification mismatch, clearing, and deeper-entry replacement protection.
- **Mate/terminal normalization fixed:** terminal scores encode distance using the current ply (`±INF ∓ ply`) but were stored raw, so the same TT position retrieved at another ply had the wrong mate distance. `scoreToTT`/`scoreFromTT` now normalize on store/probe; quiet scores remain unchanged.
- **Interrupted entry bug fixed:** a timeout or fixed-node stop could unwind and publish a provisional bound from a partially searched node. TT stores are now suppressed whenever the shared stop flag is set.
- **Iterative reuse:** a depth-3-warmed TT and a fresh TT agree on depth-5 root score/best move. A TT retained after a 300-node stopped search agrees with a clean depth-4 root score. Equal-score root move ordering may still differ when a pre-warmed TT supplies a different tie-order; this is not treated as a score correctness failure.

`npm run test:tt-correctness` passes 34 assertions, including eight one-at-a-time feature-disable searches.

### Tablebase/oracle audit

Phase 1B separates three previously conflated sources of nondeterminism:

1. `sac-two-win-three-p1` has 12 pieces and is outside `probeSmallEndgame`; its quick/full disagreement comes from the wall-clock search oracle reaching different completed work, not from a tablebase hit.
2. Small endgames use a shared memo and a wall-clock deadline. A timed-out first probe can warm exact child entries, allowing a later identical wall-clock probe to complete differently. That API is appropriate as a production opportunistic probe but not as a deterministic gate.
3. The legacy shared memo key includes `halfmoveClock` and only the current board's repetition count, not the complete repetition-history vector. Replacing it globally with a full-history key caused state-space explosion in Phase 1A, so the production solver still has this known limitation.

`probeSmallEndgameDeterministic` is a new test/oracle path with a local memo, complete sorted repetition-count state, and an exact node limit. It never consults or warms the production shared cache. `npm run test:tablebase-determinism` repeats three fixtures five times each:

| Fixture | Result | Nodes | Repeats |
|---|---|---:|---:|
| forced king capture | win, 18→9 captures 14, DTM 1 | 2 | 5/5 identical |
| same board at threefold root | draw, no move | 1 | 5/5 identical |
| `small-piece-king-vs-men`, node limit 1 | incomplete, explicit `limitReached` | 1 | 5/5 identical |

This proves outcome/history/limit separation for the fixtures; it does not claim the current on-demand tablebase is complete or near-perfect.

### Pruning and extension audit

All defaults remain byte-for-byte equivalent except the null-move draw guard:

- **Reverse futility and razoring:** restricted to quiet, non-root, non-repeated positions; still heuristic and require fixed-work ablation evidence.
- **Null move:** already excluded forced-capture nodes. Phase 1B additionally disables it when the synthetic pass could approach/cross the 32-ply inactivity limit or 16-ply all-kings limit. Synthetic null positions remain absent from repetition history by design.
- **ProbCut:** only runs on mandatory-capture nodes at depth ≥5 and returns a fail-high lower-bound cutoff. It remains tactically risky and independently disableable.
- **LMR/LMP:** already avoid forced captures; LMR also avoids quiet sacrifices that give the opponent a forced capture. Neither is proven strength-safe, so both remain ablation targets.
- **IID:** only supplies ordering at PV nodes, prevents recursive IID, and is independently disableable.
- **Extensions:** single-move, low-material, tactical, trap and root-verification extensions are grouped behind one independent switch. MAX_PLY remains the termination backstop. Splitting extension subtypes can be Phase 2 work if attribution requires it.

`SearchFeatureFlags` independently controls `reverseFutility`, `razoring`, `nullMove`, `probCut`, `iid`, `lmr`, `lmp`, and `extensions`; fixed-depth/fixed-node callers can pass overrides without environment variables or source edits.

### Verification results

| Command | Result |
|---|---|
| `npm run test:rules` | PASS — 3,501 checks |
| `npm run test:perft` | PASS — 8/8; initial 7/49/392 |
| `npm run test:tactical` | PASS — 8 checks |
| `npm run test:search-determinism` | PASS — 14 assertions; both modes 5/5 identical |
| `npm run test:tt-correctness` | PASS — 34 assertions |
| `npm run test:tablebase-determinism` | PASS — 3 assertions and 15 repeated fixture runs |
| `npm run bench:ai:fresh` | PASS thresholds — all four levels 100% solved / 0% blunder |
| `npm run regression:harness` | PASS — no warnings or fatal reasons |

Quick benchmark averages (wall-clock, therefore observational rather than deterministic):

| Level | Avg ms | Avg depth | Main nodes | Qnodes | Avg PV |
|---|---:|---:|---:|---:|---:|
| easy | 88 | 1.6 | 3,685 | 68 | 5.6 |
| normal | 125 | 2.2 | 7,530 | 111 | 5.4 |
| hard | 158 | 2.9 | 14,436 | 112 | 5.9 |
| expert | 315 | 3.4 | 41,249 | 281 | 5.7 |

### Decision and remaining issues

**KEEP** the TT occupancy fix, mate-distance normalization, interrupted-store guard, null-move draw guard, deterministic oracle, tests, and feature flags. **NEEDS MORE DATA** applies to playing strength and every future feature ablation.

Remaining issues:

- production `probeSmallEndgame` is still wall-clock/shared-cache dependent and its memo key does not encode complete history;
- the deterministic solver deliberately returns incomplete at its node limit and is only practical for bounded fixtures;
- pre-warmed TT root ordering can choose a different move among equal scores;
- pruning soundness is not proven by smoke tests; Phase 2 must run paired fixed-work ablations before Phase 3 tuning;
- the puzzle corpus still lacks authoritative solutions and a holdout split.

---

## EXP-2026-003 — Phase 1B correctness follow-up after conflict recovery

**Status:** KEEP (correctness fixes); NEEDS MORE DATA (playing strength)

**Date:** 2026-09-08

**Baseline commit:** `ed00e9afbf5b639f27060741d4a9546d9f6c155c`

**Reference-only stale commit:** `759af1b98a0987ce6f238927cec155c5828e14a4` (not cherry-picked)

**Implementation commit:** `4e2e6e91afd72f5ad9fd6f1053ddadd21955d226`

### Bugs found and fixes retained

- Conflict resolution had dropped the three npm entry points and all eight Phase 1B search-feature controls. The scripts and flags were merged back onto current source. Defaults are all `true`, preserving the production paths and existing thresholds.
- Negamax decoded mate scores with `scoreFromTT` but stored raw root-relative terminal scores. Every search TT write now uses `scoreToTT(best, ply)`, and stopped/partial searches do not publish a TT bound.
- The deterministic oracle treated its depth horizon and DFS back-edges as exact draws. Node exhaustion, depth exhaustion and cycle detection now propagate as incomplete with a specific `limitReason`; only explicit inactivity/threefold rules produce draws. A proven winning child may still resolve a node, while loss/draw requires every child to resolve.

### Regression evidence

- TT suite: PASS, 36 assertions. In addition to normalization unit checks, a forced one-move win is searched through real negamax, stored at ply 3, reused at ply 9 and compared with a fresh ply-9 search (`999990` in both cases).
- Deterministic oracle suite: PASS, 4 assertions and 20 repeated fixture runs. The node-limit fixture returns `nodes`; the new zero-depth fixture returns `depth`; both have no probe and therefore cannot claim `exact: true`. Threefold remains an exact draw and the forced capture remains an exact DTM-1 win.
- Rules: PASS, 3,501 checks. Perft: PASS, 8/8 with initial counts 7/49/392. Tactical core: PASS, 8 checks. Search determinism: PASS, 14 assertions, five identical runs per mode.
- Quick benchmark was run only to supply the regression harness input: Easy/Hard/Expert 100% solved, Normal 97%; 0% blunder at all levels. Harness classification was **WARN**, with one non-catastrophic 157-point Normal miss (`opening-bait-double-recapture-p1`) and no fatal reasons.
- `npx tsc --noEmit` could not be completed after clean dependency installation was blocked by registry HTTP 403 / incomplete npm cache. The resulting partial install lacked React Native/Expo modules. This is an environment limitation, not recorded as a passing typecheck.

### Decision and remaining issues

**KEEP** the recovered infrastructure and correctness fixes. **NEEDS MORE DATA** for any strength effect; this work changes no evaluation weights or pruning thresholds and is not evidence that the engine became stronger.

Phase 1 is **not ready to close**. The broader Thai-rules/state re-audit remains open, production `probeSmallEndgame` remains wall-clock/shared-cache dependent with an incomplete repetition-history memo key, and pruning soundness still needs paired fixed-work ablations. The dependency/lockfile installation warning also remains reproducible in this environment.

---

## EXP-2026-004 — Phase 1C final correctness audit

**Status:** KEEP (correctness only; no playing-strength claim)

**Date:** 2026-09-08

**Baseline commit:** `cd840ed1802d29776d362e13c8968da96f1c45c5` (merge PR #5)

**Candidate commit:** recorded by the Phase 1C pull request

### Scope and rule evidence

The audit started at the requested PR #5 merge commit. This checkout has no configured Git remote, so upstream freshness beyond that exact commit could not be queried; the commit is nevertheless the documented `engine-mainline` merge baseline. Source, executable perft fixtures, and existing regression history were treated as project-rule evidence over generic draughts assumptions.

The follow-up review found that the original audit had incorrectly inferred a majority-capture rule from current source. FMJD's Thai/Makhos family entry and Thai Sports Association material instead support compulsory capture with free choice among complete sequences. The global maximum-length filter was removed, and the focused fixture now requires both a complete one-piece sequence and a complete two-piece sequence to be legal. A chosen piece must still continue until it has no further capture. This was the foundational correctness bug found by Phase 1C.

The suite also covers preservation of multiple equal captures; complete multi-capture moves; forward-only movement/capture for men; crowning only after a complete move; flying-king quiet rays; first-empty-square landing after a flying-king capture; no-move and elimination terminals; threefold repetition; the retained 32/16-ply inactivity policies; capture reset and quiet increment of `halfmoveClock`; and draw/history interactions. `docs/THAI_RULES_SPEC.md` is now canonical and records evidence status. In particular, threefold and automatic 16/32-ply draws remain project policy/variant-dependent because this audit did not establish authoritative Thai competition support; they are not described as universal Thai rules.

### Fixture/search compatibility audit

- The unequal-capture perft changed from the obsolete expected count 1 to the corrected count 2. The mislabeled king-capture fixture was also repaired: its old non-capture position counted seven quiet moves, while the corrected 22x17→13 capture position counts one forced capture. Initial perft remains 7/49/392; every other recorded fixture count is unchanged.
- Tactical benchmark dataset IDs containing `max-capture` are retained to avoid changing benchmark identity/answers, but their display buckets no longer claim a majority rule. No expected answer was changed.
- Opening-book moves that were complete captures remain legal; the correction adds alternatives rather than invalidating them. Coverage/choice quality is measurement work, not a Phase 1 correctness issue.
- Puzzle fixtures may have newly legal alternatives. Existing expected answers were not changed, and their previously documented provenance/uniqueness limitations remain.
- Search capture-length sorting remains a move-ordering heuristic only; it does not filter legal moves. No evaluation or search parameter was tuned.

### Production small-endgame correctness bug and fix

The production solver could publish `exact: true` without a proof:

1. its shared memo key contained only the current board repetition count, so the same board could reuse a result across different prior histories;
2. DFS back-edges and the 60-ply horizon became draws;
3. completed descendants survived a timed-out call in shared cache, so wall-clock scheduling/cache warmth could change a later exact answer.

A full-history production memo had already demonstrated state explosion in Phase 1A. Phase 1C therefore chooses the conservative design rather than a large tablebase rewrite: production probing now returns exact results only for explicit repetition/inactivity draws, terminal losses, and directly verified mate-in-one moves. Every other eligible endgame returns incomplete (`undefined`) and safely falls through to regular search. Cache warmth and deadlines can no longer create a production exact claim. The full-history, fixed-node deterministic solver remains the bounded regression oracle.

The cross-check set contains an exact king-vs-man win, a terminal loss, repetition and inactivity draws, the same winning board under threefold history, and an unresolved king-vs-men position. Whenever production returns exact, outcome and best-move semantics agree with the deterministic oracle; unresolved production calls are explicitly accepted as fall-through.

### Search correctness verification

The Phase 1B TT suite continues to cover bound semantics, mate-distance normalization, collision/occupancy behavior, interrupted-store protection, warmed-TT reuse, and all eight pruning feature flags with production-equivalent `true` defaults. Fixed-depth/fixed-node runs remain repeatable, and qsearch continues through mandatory captures rather than standing pat. No evaluation weight, pruning threshold, search margin, benchmark answer, or feature default changed.

### Verification

| Command | Result |
|---|---|
| `npm run test:rules` | PASS — 3,755 checks after free-choice correction |
| `npm run test:perft` | PASS — 8/8; initial 7/49/392 |
| `npm run test:tactical` | PASS — 8 checks |
| `npm run test:search-determinism` | PASS — 14 assertions; both modes 5/5 identical |
| `npm run test:tt-correctness` | PASS — 36 assertions |
| `npm run test:tablebase-determinism` | PASS — 4 assertions and 20 repeated runs |
| `npm run test:phase1c` | PASS — 21 focused assertions |
| `npm run regression:harness` | PASS — no warnings or fatal reasons after capture-priority correction |
| `npx tsc --noEmit` | ENVIRONMENT LIMITATION — installed tree lacks `@expo-google-fonts/kanit` and `expo-av`; not recorded as PASS |

### Decision

**Phase 1 READY TO CLOSE.** The audit leaves no known correctness bug. Opportunistic production endgame coverage is intentionally narrower, which is a performance/strength opportunity rather than a correctness blocker. The engine is ready for Phase 2 measurement and A/B infrastructure; Phase 2 work is deliberately not started here. Puzzle quality, Elo evidence, and pruning-strength uncertainty are not Phase 1 blockers.

---

## EXP-2026-005 — Phase 2A paired A/B measurement infrastructure

**Status:** KEEP (measurement infrastructure; no playing-strength claim)

**Date:** 2026-09-08

**Baseline:** `ff6e3dad6cd5d70ed8253d4fb90ad10545c52586` (merge PR #6)

### Scope and protocol

Replaced the historical head-to-head prototype with an identity-based harness.
Each seeded, legal, deduplicated measurement start is played twice with colors
reversed. Player configs carry independent identities, feature overrides and
per-game TTs. Fixed nodes is canonical; fixed depth and time are also exposed.
The runner preserves position halfmove state and complete in-game repetition
history, applies project inactivity/threefold policy, and separates normal
draws from unresolved safety limits and engine/harness errors.

Detailed JSON and game-summary CSV include moves, per-move search metrics,
aggregate metrics, configs and environment metadata. Candidate Elo has a 95%
pair-level bootstrap interval; incomplete pairs are excluded. See
`docs/TOURNAMENT_PROTOCOL.md` for formulas and limitations.

### Validation and interpretation

The regression command checks deterministic suite generation, deduplication,
paired color assignment, identical A/B move sequences, exact repeat-run output,
and symmetric identity results. The smoke command additionally passes exactly
one existing difference (`nullMove=false`) to Candidate. That small controlled
run proves plumbing only and is **not evidence** that either configuration is
stronger. No evaluation weight, pruning threshold, search margin, default
feature flag, rule semantic, or benchmark expected answer changed.

Phase 2A does not complete position-suite curation, puzzle development/holdout
splitting, real-game strategic starts, SPRT/distributed execution, or
authoritative opening provenance.

### PR #7 measurement-correctness review

The follow-up makes compute comparability explicit: only equal node budgets and
equal depth-cap semantics are canonical. Unequal modes/budgets/caps suppress
score/Elo/CI unless explicitly requested as non-canonical descriptive data.
Score confidence bounds remain numeric on `[0,1]`, while JSON-safe tagged Elo
values distinguish finite, negative-infinity and positive-infinity results.

Focused tests now cover no-move loss, repetition and inactivity draws,
`maxPlies` unresolved classification, injected search errors, incomplete-pair
exclusion, unequal-control rejection and every Elo boundary. The controlled
`nullMove=false` smoke observes the effective feature value at actual search
dispatch rather than merely inspecting output metadata. These are measurement
correctness changes only; engine defaults, thresholds and playing strength are
unchanged.

---

## EXP-2026-006 — Phase 2B verified position-suite baseline

**Status:** KEEP (measurement infrastructure; no strength claim)

**Date:** 2026-09-08

**Engine commit tested:** `04d0d2f569ef77e411079f25125b0fb1b4e72ef6`
(merge PR #7; suite code changes no engine evaluation/search parameters)

**Suite/schema:** `makhos-position-suite-v1` /
`makhos-position-case-schema-v1`

**Canonical budget:** fixed nodes, 5,000 maximum combined main/qsearch nodes per
case; fresh TT and complete recorded history per case.

### Corpus and audit

- Verified development: 2 rule-exact cases.
- Verified holdout: 2 disjoint rule-exact cases.
- Verified diagnostic: 1 explicit history-sensitive threefold draw.
- Unresolved/unverified: 14 `legacy-unverified` puzzles plus 4 unlabeled
  deterministic-trace strategic positions.
- Of the 14 legacy expected moves, 4 are legal and 10 illegal at the current
  root. None has evidence for bestness or uniqueness. No label was repaired.
- No legacy root has capture choices of differing lengths; effects of the old
  maximum-capture assumption deeper in the undocumented claimed lines remain
  unresolved.

### Post-Phase-1 position-suite baseline

| Measurement | Result |
|---|---:|
| Development verified | 2/2 |
| Holdout verified | 2/2 |
| Diagnostic verified repetition | 1/1 |
| Legacy/unverified diagnostic rows | 14 |
| Unlabeled strategic rows | 4 |

Verified motif results were: forced capture 4/4, king technique 3/3,
endgame conversion 3/3, low mobility 1/1, and draw/repetition 1/1. Other
taxonomy motifs have a verified denominator of zero and are not reported as
accuracy evidence. Wall time/NPS were recorded in the untracked raw artifact,
not treated as deterministic.

### Decision

**KEEP. Phase 2 READY TO CLOSE.** The schema, split, audit, runner and regression
gates make future position measurements reproducible without calling engine
labels truth. The baseline does not demonstrate or claim a strength improvement,
and it is not Teacher v1. SPRT/distributed testing remains optional later work.

### Review correction — measurement semantics and holdout freeze

Review found that the initial runner compared generic search-score sign with
W/D/L, serialized only the numeric from/to move key, and froze v1 only by its
version string. These were measurement-correctness defects, not engine defects.

- W/D/L now requires direct repetition/inactivity or terminal-rule evidence,
  or an actually resolved deterministic exact-oracle result. Ordinary positive,
  zero, and negative heuristic scores cannot adjudicate W/D/L; unresolved W/D/L
  rows are unscored.
- JSON retains the complete chosen move and CSV uses a signature containing
  from/to, captured sequence, path, and promotion.
- V1 freezes scored/leakage-relevant content with SHA-256
  `f2d5d8d41275836904750825b60230f283724ac3d97e18305157169d03d5aa30`.
- Default artifacts omit all holdout case rows while retaining aggregate
  accuracy. Blinded per-motif statistics also exclude holdout cases, preventing
  singleton motif buckets from disclosing verdicts indirectly. Detailed rows
  and full motif statistics require `--reveal-holdout`; the canonical initial
  baseline invokes that evaluation-only option explicitly.

Regression tests include positive/negative heuristic-score non-proof, direct
threefold adjudication, same-endpoint capture identity, fingerprint mutation,
and default holdout redaction. Corpus membership and baseline scores remain
unchanged. No engine defaults or strength parameters changed.

---

# Experiment template

## EXP-2026-007 — Phase 3A search-feature ablation baseline

**Status:** KEEP infrastructure and evidence; NEEDS MORE DATA before any production change

**Date:** 2026-09-08

**Baseline commit:** `5b9659a13d22bd748474ec82947db72e48dc9131`

### Hypothesis and controls

One-at-a-time disabling at equal fixed work can identify which existing search
feature is the strongest Phase 3B hypothesis without tuning a threshold or
changing engine defaults. The audit, trigger details, fixed corpus, verdict
definition, and replay commands are in `docs/SEARCH_ABLATION_PROTOCOL.md`.
Screening used the first 32 of 64 frozen paired starts, 5,000 nodes per move,
maximum 160 played plies, and two color-swapped games per start. All pairs
completed without unresolved games or errors. Raw artifacts were generated in
ignored `.tmp/search-ablation/`; only their concise evidence is recorded here.

### Screening results

Candidate is the engine with the named feature disabled. W/D/L is candidate
wins/draws/losses. Depth is candidate average completed depth. All intervals
except extensions cross or touch zero Elo and are therefore inconclusive.

| Feature disabled | Pairs | W/D/L | Score | Elo | 95% CI | Depth | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| reverseFutility | 32 | 30/5/29 | 50.78% | +5.4 | [-43.7, +54.7] | 4.30 | inconclusive |
| razoring | 32 | 29/10/25 | 53.13% | +21.7 | [0.0, +54.7] | 4.63 | inconclusive |
| nullMove | 32 | 34/5/25 | 57.03% | +49.2 | [-5.4, +106.3] | 4.57 | inconclusive |
| probCut | 32 | 32/8/24 | 56.25% | +43.7 | [-21.7, +112.3] | 4.17 | inconclusive |
| iid | 32 | 29/7/28 | 50.78% | +5.4 | [-16.3, +32.7] | 4.74 | inconclusive |
| lmr | 32 | 31/7/26 | 53.91% | +27.2 | [-21.7, +83.0] | 4.72 | inconclusive |
| lmp | 32 | 34/5/25 | 57.03% | +49.2 | [-5.4, +106.3] | 4.95 | inconclusive |
| extensions | 32 | 57/3/4 | 91.41% | +410.7 | [+303.8, +596.5] | 12.69 | likely harmful |

At fixed nodes, candidate total main/qnodes respectively were: RFP
6,671,866/547,851; razoring 6,615,131/400,973; null move6,844,060/234,345;
ProbCut 6,607,999/467,312; IID 6,382,021/517,220; LMR
6,379,877/570,788; LMP 6,334,837/533,111; extensions
3,970,242/2,242,840. NPS is retained in the machine-readable raw summary but
is diagnostic only because fixed-node strength does not depend on wall clock.

### Expanded-corpus evidence and diagnostic

The sole expanded-corpus candidate was `extensions=false`: 64/64 paired starts
(128 games), equal 5,000 nodes per move, maximum 160 plies; all pairs completed
with zero unresolved/errors. Candidate W/D/L was **110/5/13**, score **87.89%**,
estimated Elo **+344.3**, pair-bootstrap 95% CI **[+273.2, +438.4]**. Candidate
average depth was **12.47**, with 7,779,030 main and 4,441,253 qnodes across
the run. This expands the same generated v1 screening corpus from its first 32
starts to all 64 and repeats the direction, but it is **not** an independent
Phase 3B confirmation and does not authorize changing the default.

A depth-4 diagnostic over eight pairs proved impractical in this environment:
grouped extensions expand forced/endgame lines beyond nominal depth and the run
was stopped after more than five minutes before completing the first report; a
depth-3/four-pair fallback was likewise stopped after three minutes without a
complete report. This runtime limitation is recorded rather than substituting
partial data. Fixed-depth data is explanatory only and is not included in the
Elo conclusion.

### Correctness and blinded position status

Rules (3,755 checks), perft (8/8), tactical (8 checks), fixed-work search
determinism (14 assertions), TT correctness (36 assertions, including each
one-at-a-time flag), deterministic tablebase (4 assertions), Phase 1C (21
assertions), tournament harness, position-suite tests (49 checks), and the new
64-start ablation infrastructure test passed. The development position run at
5,000 nodes scored 2/2 verified development cases. It used the default blinded
mode (`revealHoldout=false`); no holdout case detail or per-case feedback was
used for selection. Full application typecheck was blocked by absent installed
Expo/font modules, not an engine TypeScript error.

### Decision

**`extensions` is the strongest Phase 3B hypothesis.** Evidence is unusually
large and directionally repeated, but uncertainty remains about which member of
the grouped extension switch causes the loss, corpus representativeness, and
the relationship between equal nodes and mobile wall-clock cost. Phase 3B must
test exactly one controlled hypothesis, repeat all gates with that candidate,
and confirm it using an independent corpus/seed or otherwise independent
evidence. Do **not** disable the
group in production from Phase 3A, combine it with another ablation, or call the
result Teacher v1.

---

Copy this section for each experiment.

## EXP-YYYY-NNN — Short title

**Status:** PLANNED / RUNNING / KEEP / REVERT / NEEDS MORE DATA

**Date:** YYYY-MM-DD

**Baseline commit:** `<sha>`

**Candidate commit:** `<sha>`

### Hypothesis

One clear statement. Example: “Using full flying-king ray mobility in evaluation improves positional strength without tactical regressions.”

### Change

- files:
- feature/parameter:
- feature flag if applicable:

### Environment

- OS:
- CPU:
- Node:
- mode: fixed-depth / fixed-nodes / fixed-time
- limits:

### Correctness gates

| Gate | Baseline | Candidate | Pass? |
|---|---|---|---|
| Rules | | | |
| Perft | | | |
| Tactical core | | | |
| Tablebase agreement | | | |

### Puzzle results

| Set | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Development | | | |
| Holdout | | | |

List newly solved positions and newly regressed positions separately.

### Head-to-head

- games:
- paired openings:
- W/D/L candidate perspective:
- score %:
- estimated Elo delta:
- confidence interval / sequential-test result:

### Performance

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| nodes/move | | | |
| depth | | | |
| NPS | | | |
| think time | | | |

### Analysis

Why did the result happen? Note tactical regressions, speed-vs-strength tradeoffs, and any suspicious overfitting.

### Decision

**KEEP / REVERT / NEEDS MORE DATA**

Reason:

---

# Historical context worth preserving

These are historical observations from earlier branches/reports and must not be treated as freshly verified results:

- Low-mobility override work around commit `9f7eee3` reported expert benchmark solve 100% and blunder 0%, after fixing a regression caused by a too-aggressive fixed threshold.
- Historical `PUZZLE_BASELINE.md` reported only 1/14 expected puzzle moves correct.
- Earlier NN experiments reported impressive validation/benchmark numbers, but at least one training setup reused the same data for train/validation and some match samples were small. Do not use those numbers as promotion evidence for future models.
- `tuneNN` contains later NN/training/mobile work and some classical changes; recover tools selectively instead of merging the branch wholesale.

The purpose of this section is to preserve context while forcing all future claims through the new baseline/testing protocol.

## EXP-2026-008 — Extension subtype attribution and independent confirmation

**Status:** KEEP AS PHASE 3C HYPOTHESIS (production defaults unchanged)

**Date:** 2026-09-08

**Baseline commit:** `4149c4b1989378b2a85746b36190c915fdb16d4d`

### Hypothesis and change

Split the grouped extension control without changing semantics, instrument its
actual triggers/cost, screen each subtype alone, and independently confirm only
the strongest. No evaluation, pruning threshold, other feature, or production
default changed. The frozen protocol and source audit are in
`docs/EXTENSION_ABLATION_PROTOCOL.md`.

### Phase 3A-corpus screening

Candidate W/D/L is candidate wins/draws/baseline wins. All runs completed 32
pairs/64 games with no unresolved games or errors. Main/qnodes are totals for
baseline then candidate. Trigger/add are the baseline count and cumulative
actual added depth for the disabled subtype; R/I splits triggers by root/interior.

| Disabled subtype | W/D/L | Score | Elo (95% pair CI) | Avg depth B/C | Main/qnodes B; C | Trigger / add (R/I) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| single legal move | 32/6/26 | 54.69% | +32.7 [-32.7,+100.4] | 4.70/4.84 | 6,213,712/525,374; 6,130,619/642,475 | 721,093 / 721,093 (5,139/715,954) | inconclusive |
| small endgame | 40/7/17 | 67.97% | +130.7 [+60.3,+213.3] | 4.58/6.38 | 6,343,357/525,060; 4,694,428/2,214,103 | 5,874,554 / 5,229,114 (6,534/5,868,020) | likely harmful |
| tactical capture | 34/7/23 | 58.59% | +60.3 [-10.9,+130.7] | 5.00/5.57 | 6,207,047/521,915; 6,118,296/675,749 | 89,202 / 89,202 (4,821/84,381) | inconclusive |
| multi-capture | 28/9/27 | 50.78% | +5.4 [-32.7,+43.7] | 4.87/4.75 | 6,684,630/546,827; 6,685,048/557,150 | 9,497 / 9,497 (273/9,224) | inconclusive |
| opponent forced capture | 27/6/31 | 46.88% | -21.7 [-106.3,+60.3] | 4.51/5.84 | 6,433,376/539,133; 6,027,441/956,207 | 287,647 / 252,371 (10,680/276,967) | inconclusive |
| single-capture recapture | 27/10/27 | 50.00% | 0.0 [0.0,0.0] | 4.54/4.54 | 6,643,365/532,631; identical | 0 / 0 (0/0) | unreachable/inconclusive |
| root low mobility | 27/10/27 | 50.00% | 0.0 [0.0,0.0] | 4.50/4.61 | 6,662,962/534,150; 6,655,532/535,269 | 2,038 / 5,263 (2,038/0) | inconclusive |
| sound forced trap | 35/12/17 | 64.06% | +100.4 [+54.7,+156.3] | 4.06/4.53 | 6,814,873/526,426; 6,791,896/536,378 | 904 / 1,420 (904/0) | likely harmful |

Small-endgame was selected mechanically as the strongest single candidate by
screening score, ahead of sound-forced-trap. Root low mobility did add 2.58 plies
per trigger on average but occurred only 2,038 times and its removal produced
identical game scores; it did not consume a disproportionate share compared
with small-endgame's 5.23 million actually added plies. The grouped Phase 3A
ceiling was 91.41% (+410.7 Elo) on the same 32 starts, materially larger than
any subtype, so interaction/additional-subtype effects remain plausible.

### Independent confirmation

On the pre-frozen independent v1 suite, disabling only `smallEndgame` scored
**83/11/34, 69.14%, +140.1 Elo, pair-bootstrap 95% CI [+94.6,+194.5]** over
64 complete pairs/128 games, with zero unresolved/errors. Average completed
depth rose from 4.64 to 6.66. Baseline main/qnodes were
12,454,210/1,027,820; candidate 9,146,904/4,367,355. Baseline small-endgame
instrumentation recorded 11,666,298 triggers, 10,365,493 actual added plies,
including 17,047 root triggers/11,863 root added and 11,649,251 interior
triggers/10,353,630 interior added. Thus its repeated late-game extension work
explains a large part of the fixed-node completed-depth collapse.

### Correctness and decision

Semantic tests compare full move identity, score, completed depth and PV for
quiet, forced-capture, multi-capture, <=5-piece, and low-mobility fixtures with
implicit versus explicit all-on defaults; they also prove master-off equals all
subtypes-off. All required correctness gates passed (the blinded development
measurement used no `--reveal-holdout`). Full app typecheck remains blocked only
by the repository environment's missing Expo modules.

**Phase 3C hypothesis: disable `smallEndgame`.** The enabled subtype is likely
harmful because the independent candidate score CI is wholly above 50%.
Production defaults remain unchanged in this PR. This is not Teacher v1.

---

## EXP-2026-009 — Phase 3C smallEndgame promotion

**Status:** KEEP — PROMOTED AS TEACHER V1

**Date:** 2026-09-09

**Baseline commit:** `564b298ac07f30709b81093861e61ccf01cb0d4e`

**Teacher v1 commit:** `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f` — frozen Teacher v1 release identity.

### Frozen hypothesis and controls

Candidate was identical to the production baseline except
`smallEndgame=false`. No evaluation weight, pruning threshold, or other search
feature changed. All games used paired colors and the already frozen,
Phase 3B `makhos-extension-confirmation-starts-v1` corpus (seed 1095353521,
SHA-256 `58475f4a2913f4a6c093d12a54a89e6a30d6160f5921f63964bb83267d5384ab`).
The first 32 starts were selected before these results, with 160 plies maximum;
all 256 games completed with no unresolved games or errors. Pair-bootstrap CIs
used 20,000 deterministic resamples.

### Paired promotion tournaments

W/D/L is from the candidate perspective. Node and qnode figures are total
baseline then candidate; elapsed is total search milliseconds and NPS includes
both main and qnodes.

| Budget | W/D/L | Score | Elo (95% pair CI) | Depth B/C | Main nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 nodes/move | 38/6/20 | 64.06% | +100.4 [+38.2,+169.8] | 4.78/6.76 | 6,270,683/4,594,434 | 508,492/2,196,311 | 81,865/96,058 | 82,809/70,694 |
| 20,000 nodes/move | 47/1/16 | 74.22% | +183.7 [+112.3,+273.2] | 4.94/7.87 | 24,725,757/17,911,737 | 1,189,972/8,094,019 | 190,921/196,935 | 135,741/132,052 |
| 50,000 nodes/move | 53/4/7 | 85.94% | +314.4 [+229.2,+448.5] | 4.87/9.28 | 63,589,752/45,694,914 | 2,038,696/20,502,333 | 439,990/432,760 | 149,159/152,965 |
| 100 ms/move | 48/4/12 | 78.13% | +221.1 [+149.8,+314.4] | 2.18/3.68 | 17,737,355/11,589,544 | 812,042/6,233,116 | 153,106/144,612 | 121,154/123,245 |

The win does not disappear at higher fixed work: it grows from +100 Elo at 5k
to +314 Elo at 50k, and every CI excludes zero. The production-like equal-time
test also strongly favors the candidate, while using 5.5% less aggregate search
time and slightly higher NPS. The candidate consistently converts the removed
extension overhead into much greater completed nominal depth. Phase 3B's
separate 64-pair confirmation (+140.1 Elo, CI [+94.6,+194.5]) selected this
corpus. Therefore the multi-budget subset is robustness/replication on the same
frozen corpus, not a second independent-corpus result.

### New disjoint-corpus confirmation (PR review)

The final corpus is `makhos-phase3c-final-confirmation-starts-v1`, seeded with
`0xb59570ad` (3046469805), mechanically derived from the first eight digits of
GitHub-resolvable PR #11 commit
`b59570adb3e94dc629fc5a9af262b96b9d030498`. No alternate seed was tried. Its
fingerprint is `fc48f9aee765fd13d30c2577d6c3031bada555214d1d831148a58ae68ea16d0f`;
all 64 states are checked disjoint from Phase 3A, Phase 3B, and the superseded
Phase 3C review corpus. Final results are recorded below after the run.

| Budget | W/D/L | Score | Elo (95% pair CI) | Depth B/C | Main nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes/move | 43/6/15 | 71.88% | +163.0 [+100.4,+237.5] | 4.50/7.51 | 26,543,806/18,881,891 | 1,215,961/8,891,277 | 211,978/214,209 | 130,956/129,655 |
| 100 ms/move | 44/0/20 | 68.75% | +137.0 [+54.7,+221.1] | 2.35/3.62 | 19,378,770/13,301,190 | 866,472/7,167,844 | 140,652/134,109 | 143,939/152,630 |

Both final confirmation intervals exclude zero Elo. All 128 games completed
with paired colors, no unresolved games, and no errors. This remotely auditable,
new-corpus result is the independent evidence supporting promotion; the earlier
Phase 3C runs remain same-corpus robustness evidence only.

### Correctness and regression gates

Rules passed 3,755 checks; perft passed 8/8 (initial 7/49/392); tactical passed
8 checks; fixed-node/depth determinism passed 14 assertions; TT passed 36;
deterministic tablebase passed four assertions/20 repeat probes; Phase 1C passed
21; position infrastructure passed 49; and Phase 3B semantic/corpus checks
passed. Full TypeScript `npx tsc --noEmit` also passed.

The verified position suite was kept blind through selection. After the
promotion decision was fixed, the evaluation-only comparison revealed baseline
and candidate both at development 2/2 and holdout 2/2. All three endgame-tagged
rows, including the <=5-piece exact cases most directly exposed to this change,
passed for the candidate. There were no newly regressed verified tactical,
endgame, or holdout cases; the corpus is small, so this is a regression guard
rather than a broad endgame-strength estimate.

### Environment and reproducibility

Ubuntu 24.04.4 LTS, Intel Xeon Platinum 8370C @ 2.80 GHz, Node v24.15.0. Run
`npm run experiment:phase3c` for the frozen multi-budget tournament and
`npm run test:phase3c-positions` for the post-decision verified-suite comparison.
Both emit machine-readable JSON/CSV under ignored `.tmp/phase3c/`.

### Decision

**KEEP AND APPROVE FOR PROMOTION.** All promotion criteria passed. Production now defaults
only `smallEndgame` to false; every other extension/search feature remains at
the baseline value. PR #11 merge commit
`b2e6a35db6a50ea294a10f6b76a90b4e70e0689f` **is the frozen Teacher v1 release identity**.

---

## EXP-2026-010 — Phase 3D soundForcedTrap confirmation

**Status:** NEEDS MORE DATA — candidate supported, production default unchanged

**Date:** 2026-09-09

**Canonical starting commit:** `075cd5bfb69eabac4555eef6d048fa35700efadb`

**Frozen Teacher v1 identity:** `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`

### Frozen hypothesis and controls

The candidate differs from Teacher v1 only by `soundForcedTrap=false`.
Both tournament engines explicitly pin the complete extension configuration,
including Teacher v1's `smallEndgame=false`; no production default, evaluation,
pruning, reductions, null move, ProbCut, IID, ordering, or
`singleCaptureRecapture` behavior changed. The protocol was fixed before games
were inspected. Phase 3B's approximately +100 Elo screen was not confirmation.

The fresh suite is
`makhos-phase3d-sound-forced-trap-confirmation-starts-v1`, seed `0x075cd5bf`
(123524543), mechanically derived from the canonical starting commit, with
SHA-256 `d68b1b8f33b3f4e7b490e44ff8bd2c4576ba11b14c81a52a987471598cb1ab6d`.
Its frozen v1 generator and first-64 selection rule are documented in the
protocol. Automated complete-state checks prove all 64 unique states disjoint
from every retained Phase 3A, 3B, superseded Phase 3C, and final Phase 3C
corpus. No alternate corpus or seed was generated after outcomes.

### Independent paired-color results

Candidate W/D/L is candidate wins/draws/baseline wins. Each row contains 32
pairs/64 games, depth cap 64, and 160-ply maximum. All games completed with no
unresolved games or errors. Node and time metrics are totals for baseline then
candidate. Pair-bootstrap intervals use 20,000 deterministic pair resamples.

| Budget | W/D/L | Score | Elo (95% pair CI) | Depth B/C | Main nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C | baseline trap trigger/add |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes/move | 35/6/23 | 59.38% | +65.9 [+21.7,+118.4] | 7.92/8.30 | 18,787,579/18,846,016 | 8,668,785/8,650,141 | 228,743/234,979 | 120,031/117,015 | 761/1,197 |
| 50,000 nodes/move | 36/2/26 | 57.81% | +54.7 [0.0,+112.3] | 9.44/9.66 | 46,688,804/46,914,627 | 20,151,518/20,027,273 | 488,743/494,535 | 136,760/135,363 | 1,690/2,748 |
| 100 ms/move | 30/13/21 | 57.03% | +49.2 [-5.4,+106.3] | 4.06/4.09 | 11,557,331/11,344,447 | 6,192,873/6,109,077 | 142,964/143,210 | 124,159/121,874 | 399/489 |

All measured trap events were root events. Candidate trap counters were zero,
as expected. The canonical 20k interval excludes zero and all point estimates
favor the candidate. The 50k lower bound touches zero and the production-like
time interval crosses zero, however, so the preferred evidence standard is not
met at both fixed work and time.

### Correctness, holdout, and decision

Rules passed 3,755 checks; perft 8/8; tactical 8; search determinism 14; TT 36;
tablebase determinism four assertions/20 repeat probes; Phase 1C 21; tournament
harness; position-suite infrastructure 49; Phase 3B/3C extension semantics and
all corpus checks; and full TypeScript typecheck all passed. The Phase 3C
position regression gate also passed. Phase 3D development output was first run
blinded. Only after candidate, corpus, results, and decision rule were frozen
was holdout revealed: baseline and candidate both passed development 2/2 and
holdout 2/2, with all three endgame-tagged rows passing. No verified tactical,
holdout, or endgame regression occurred.

**Decision: NEEDS MORE DATA.** Independent evidence supports the hypothesis,
and correctness gates pass, but the preferred confidence condition is not met
at 50k or equal time. Retain the candidate as a supported follow-up hypothesis;
do not promote or change the production `soundForcedTrap` default in this PR.

### Untouched second-half confirmation (PR #14 continuation)

Before inspecting starts 33-64, the runner was corrected to encode all eight
Teacher v1 extension flags as literals rather than spreading contemporary
defaults. The position comparison uses the same literal baseline. Candidate is
constructed from that frozen object with only `soundForcedTrap=false`. A
zero-based start offset selects the existing suite range without changing its
seed, content, ordering, or fingerprint. No first-half games were rerun and no
candidate setting was changed in response to outcomes.

Candidate W/D/L remains candidate wins/draws/baseline wins. These are the
independent results from only the previously untouched starts 33-64:

| Budget | W/D/L | Score | Elo (95% pair CI) | Depth B/C | Main nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C | baseline trap trigger/add |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes/move | 34/4/26 | 56.25% | +43.7 [0.0,+94.6] | 7.92/7.89 | 19,206,922/19,340,662 | 8,851,181/8,834,450 | 231,663/239,905 | 121,116/117,443 | 603/955 |
| 50,000 nodes/move | 33/7/24 | 57.03% | +49.2 [0.0,+100.4] | 8.51/8.82 | 47,288,991/47,310,592 | 21,689,124/21,548,152 | 452,379/462,701 | 152,479/148,819 | 759/1,085 |
| 100 ms/move | 36/13/15 | 66.41% | +118.4 [+65.9,+176.7] | 3.95/3.87 | 12,789,573/12,461,005 | 6,728,776/6,533,743 | 151,279/150,955 | 129,022/125,831 | 253/323 |

All 192 second-half games completed with zero unresolved games and errors. All
trap instrumentation was at the root, and candidate trap counters were zero.
The 100 ms interval is wholly above zero, but both fixed-work lower bounds only
touch zero rather than exceed it. Therefore the stated continuation promotion
rule is not satisfied.

### All-64 descriptive aggregation

The two halves aggregate descriptively as follows. This aggregation is not a
substitute for the independent second half. The retained first-half artifact
contains aggregate statistics rather than individual pair scores, so an exact
all-64 pair bootstrap cannot be reconstructed without rerunning forbidden
first-half games; the combined CI is reported as unavailable rather than using
an invalid game-level or synthetic-pair interval.

| Budget | W/D/L | Score | Elo | Depth B/C | Main nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C | baseline trap trigger/add |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes/move | 69/10/49 | 57.81% | +54.7 | 7.92/8.09 | 37,994,501/38,186,678 | 17,519,966/17,484,591 | 460,406/474,884 | 120,577/117,231 | 1,364/2,152 |
| 50,000 nodes/move | 69/9/50 | 57.42% | +52.0 | 8.97/9.24 | 93,977,795/94,225,219 | 41,840,642/41,575,425 | 941,122/957,236 | 144,315/141,867 | 2,449/3,833 |
| 100 ms/move | 66/26/36 | 61.72% | +83.0 | 4.00/3.98 | 24,346,904/23,805,452 | 12,921,649/12,642,820 | 294,243/294,165 | 126,659/123,904 | 652/812 |

**Continuation decision: NEEDS MORE DATA.** The untouched second half again
has positive point estimates and provides strong equal-time evidence, but its
fixed-work intervals do not lie entirely above zero. Keep the hypothesis; do
not mark it a strong promotion candidate and do not change the production
default in PR #14.

## EXP-2026-011 — Phase 3E move-ordering ablation

**Status:** MEASURED — countermove removal selected for confirmation; no production change

**Date:** 2026-09-09

**Canonical starting commit:** `07a3f052904a4e72095b694d4e546f16c4162afa`

**Frozen Teacher v1 identity:** `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`

### Audit, controls, and frozen corpus

The pre-outcome audit is recorded in `docs/PHASE3E_MOVE_ORDERING_PROTOCOL.md`.
All six requested mechanisms exist, although recapture is qsearch-only and uses
the prior capture's `from` square; there is no distinct main-search recapture
bonus. Phase 3E added measurement-only flags and counters. All defaults remain
true. Baseline and candidates explicitly pin all search flags, all extension
flags (`smallEndgame=false`, `soundForcedTrap=true`), and all ordering flags.
No value was tuned and no evaluation, LMR/LMP, pruning, extension, TT
replacement, qsearch legality/rule, or production default changed.

The fresh 64-state corpus uses seed `0x07a3f052` (128184402) and unchanged
deterministic generator style. Version
`makhos-phase3e-move-ordering-starts-v1` has fingerprint
`00db4d9cedfe76202dae15373d33bfdc89f31726ad2b9b4e0fe3438701e97091`.
Tests prove complete-state uniqueness and disjointness from all retained Phase
3A–3D corpora. Identity, starts 1–32 screening, starts 33–64 confirmation, and
the point-estimate selection rule were frozen before outcomes.

### Screening — starts 1–32 only

Candidate W/D/L means removal wins/draws/baseline wins. Every row is 32 paired
starts (64 games), 5,000 nodes/move, depth cap 64, max 160 plies. All games
completed without errors or unresolved adjudications. CIs are deterministic
20,000-resample pair bootstraps. Nodes, qnodes, elapsed, and NPS are baseline /
candidate totals; instrumentation is preserved in the JSON artifact.

| Disabled | W/D/L | Score | Elo (95% pair CI) | Avg depth B/C | Nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TT move | 21/16/27 | 45.31% | -32.7 [-83.0,+16.3] | 6.16/5.65 | 5,299,142/5,257,992 | 2,647,276/2,686,441 | 110,298/104,285 | 72,045/76,180 |
| capture | 27/12/25 | 51.56% | +10.9 [-10.9,+32.7] | 6.05/6.06 | 5,165,293/5,167,558 | 2,526,141/2,529,327 | 156,950/160,966 | 49,006/47,817 |
| killers | 27/9/28 | 49.22% | -5.4 [-77.2,+65.9] | 6.14/6.17 | 4,994,428/4,951,085 | 2,419,721/2,448,947 | 153,227/155,233 | 48,387/47,670 |
| history | 24/10/30 | 45.31% | -32.7 [-88.7,+21.7] | 5.88/5.94 | 5,291,627/5,217,716 | 2,597,001/2,632,103 | 164,551/162,036 | 47,940/48,445 |
| countermove | 28/11/25 | 52.34% | +16.3 [-32.7,+65.9] | 5.93/6.19 | 5,074,463/5,042,891 | 2,453,579/2,465,328 | 158,344/156,366 | 47,542/48,017 |
| recapture | 25/14/25 | 50.00% | 0.0 [0.0,0.0] | 5.94/5.94 | 5,146,317/5,146,323 | 2,535,162/2,535,156 | 157,366/158,320 | 48,813/48,519 |

The mechanical rule selected **countermove removal**, the strongest point
estimate strictly above 50%. This weak screen (CI crosses zero) is only a way to
choose one candidate; it is not promotion evidence. Starts 33–64 were not read
before that selection.

### Confirmation — untouched starts 33–64

The selected candidate alone was measured on the reserved 32 paired starts.
All 192 games completed, with no errors or unresolved games.

| Budget | W/D/L | Score | Elo (95% pair CI) | Avg depth B/C | Nodes B/C | Qnodes B/C | Elapsed ms B/C | NPS B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes/move | 31/11/22 | 57.03% | +49.2 [-10.9,+112.3] | 7.35/7.46 | 21,797,752/21,797,777 | 10,406,637/10,438,943 | 277,232/275,997 | 116,164/116,801 |
| 50,000 nodes/move | 28/9/27 | 50.78% | +5.4 [-49.2,+60.3] | 8.46/8.53 | 50,575,993/50,422,129 | 22,905,743/22,991,429 | 513,761/516,891 | 143,027/142,029 |
| 100 ms/move | 30/7/27 | 52.34% | +16.3 [-49.2,+83.0] | 3.67/3.64 | 12,680,378/12,583,617 | 6,832,481/6,772,488 | 147,301/147,024 | 132,469/131,653 |

All confirmation point estimates favor removal, but every interval crosses
zero and the effect contracts materially at 50k. **Decision: inconclusive for
promotion.** Keep countermove removal only as a measured follow-up hypothesis.
Do not change any production default in this PR.

### Pre-merge production instrumentation correction

The accepted frozen results, corpus, selection, and inconclusive decision above
were not rerun or reinterpreted. Move-ordering counter collection is now
independently opt-in and defaults off, avoiding per-search statistics allocation
and hot-path increments in production. The Phase 3E runner opts both baseline
and candidate into collection to retain the experiment's measurement behavior.
Targeted quiet/capture tests prove that default ordering, explicit all-enabled
ordering, and statistics-enabled ordering produce identical best move, score,
completed depth, PV, nodes, and qnodes; only the optional instrumentation field
differs.

## EXP-2026-012 — Phase 3F LMR schedule tuning

Phase 3F began at canonical tip `8208f6f68bc16d0df00e68691e8e1b319db6cf00` with Frozen Teacher v1 identity `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`. The source audit, four-profile inventory, mechanical rule, new 64-start corpus, partition, and fingerprint were frozen before outcomes. Production remains the exact `current` LMR schedule.

The new `makhos-phase3f-lmr-starts-v1` corpus uses seed `0x8208f6f6` (`2181625590`), fingerprint `b22433396fe82ec54461f449944cb780fedca7caa4f5fab4a440a0e980972f89`, and is complete-state unique and disjoint from all retained Phase 3A-3E corpora. Starts 1-32 screened; starts 33-64 were untouched until confirmation.

### Screening (5,000 nodes/move, 32 pairs / 64 games)

| profile | W/D/L | score | Elo | pair-bootstrap Elo 95% CI |
|---|---:|---:|---:|---:|
| off | 31/6/27 | 53.125% | +21.7 | [-43.7, +88.7] |
| gentler | 32/6/26 | 54.688% | +32.7 | [-38.2, +106.3] |
| delayed | 31/7/26 | 53.906% | +27.2 | [-43.7, +100.4] |
| aggressive | 34/7/23 | 58.594% | +60.3 | [-10.9, +137.0] |

All runs had zero unresolved games and errors. Per the predeclared rule, `aggressive` was selected solely because it had the highest score strictly above 50%. This is selection evidence only.

### Untouched confirmation: aggressive versus current

| budget | W/D/L | score | Elo | pair-bootstrap Elo 95% CI | avg depth current/aggressive | main nodes current/aggressive | qnodes current/aggressive | elapsed ms current/aggressive | NPS current/aggressive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 nodes | 28/8/28 | 50.000% | 0.0 | [-71.6, +71.6] | 7.710/8.175 | 19,856,212/20,316,278 | 9,139,117/8,732,942 | 245,913/254,507 | 117,909/114,139 |
| 50,000 nodes | 32/11/21 | 58.594% | +60.3 | [-5.4, +130.7] | 8.287/9.023 | 50,179,502/51,557,690 | 23,426,729/22,382,398 | 552,027/533,324 | 133,338/138,640 |
| 100 ms | 29/9/26 | 52.344% | +16.3 | [-60.3, +88.7] | 3.790/4.149 | 10,812,640/10,618,394 | 5,737,218/5,285,870 | 134,599/134,058 | 122,957/118,637 |

All confirmation runs had zero unresolved games and errors. The equal-time run deliberately had no LMR instrumentation. Fixed-node counters show the intended mechanism: at 50,000 nodes, current reduced 6,655,725 eligible moves for 8,553,436 plies (histogram 0/1/2/3 = 0/4,869,896/1,673,947/111,882; 40,411 full-depth re-searches), while aggressive reduced 7,610,339 moves for 17,389,894 plies (0/0/5,441,123/2,169,216; 67,757 re-searches). Full root/interior partitions and the equivalent 20,000-node data are retained in the machine-readable artifacts.

The point estimates favor aggressive at two confirmation budgets, but every confirmation CI crosses zero. The result is **inconclusive**, so no production change is promoted. No evaluation, pruning, extension, ordering, TT, or qsearch behavior changed; the replay control and instrumentation remain opt-in.
