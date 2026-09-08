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

The canonical machine-readable copy of these Phase 0 results is `benchmarks/teacher-v0-baseline.json`. Generated reports under `.tmp/` remain non-canonical and may be overwritten by later runs.

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

**Candidate commit:** this Phase 1A commit

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

# Experiment template

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
