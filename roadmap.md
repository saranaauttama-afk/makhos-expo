# Makhos AI Improvement Roadmap

This roadmap is the working plan for improving Level 1-4 as one integrated engine system, not as isolated patches for a single minimax depth.

## Current Handoff

Last updated: 2026-05-03.

Current work state:
- Final full ladder benchmark completed on 2026-05-04 03:32 ICT.
- Latest full report: `.tmp/benchmarks/ai-benchmark-full-latest.json`.
- Final analyzer result: `Overall: PASS`.
- Final tactical gate: easy 97% solve / 3% blunder / p95 1406ms; normal 97% / 3% / 2607ms; hard 100% / 0% / 3484ms; expert 100% / 0% / 6290ms.
- Final H2H ladder diagnostics all pass: normal > easy 8.5-3.5, hard > easy 7.0-5.0, hard > normal 8.5-3.5, expert > easy 8.5-3.5, expert > normal 10.0-2.0, expert > hard 8.5-3.5.
- Current AI work can be considered release-gate passed for benchmark purposes. Remaining work is gameplay smoke testing on device and non-AI release readiness.
- Latest tactical validation after oracle/low-mobility/root-safety fixes: `npm run bench:ai:full:tactical` passes release gate.
- Full tactical summary after fixes: easy solve 95% / blunder 3%; normal 97% / 3%; hard 97% / 3%; expert 100% / 0%; all p95 targets pass.
- Fixed a benchmark oracle false positive where low-mobility root search could report losing `26->23` as oracle; low-mobility oracle now scores each legal move in quiet <=3-move, <=8-piece positions.
- Fixed low-mobility search fallback by delaying root low-mobility extension until depth >=4; this prevents depth-0 fallback to the first legal move.
- Added low-mobility recapture resolver so positions that offer a piece to force a stronger recapture choose the recapture line without globally raising depth.
- Reduced false positives from root/absolute anti-hang safety: root safety now needs at least depth 2, and absolute anti-hang only triggers on heavier immediate risk.
- Next validation gate: full H2H ladder after the tactical gate pass.
- Latest ladder result after low-mobility fix: tactical full gate passed at 100% solve / 0% blunder for all L1-L4, but ladder still failed because hard did not beat easy/normal reliably.
- Diagnosis: `hard` was still using post-search skill-gap selection, so it could choose safe near-best quiet moves like the lower levels. This made hard's identity too close to normal despite higher search budget.
- Change after that result: `selectStrictLevelMove()` now applies skill-gap move selection only to easy/normal. Hard and expert keep the engine best move.
- Quick smoke after hard-best change: `npx tsc --noEmit`, `npm run test:rules`, and `npm run test:tactical` pass; quick tactical hard/expert solve 100% / blunder 0%.
- Next validation gate: rerun full head-to-head after hard-best change and check `hard > easy` plus `hard > normal`.
- Latest status after 2026-05-03 ladder follow-up: quick/rule/tactical smoke tests pass after root low-mobility tuning.
- Latest full benchmark before this tuning passed tactical release gate but failed ladder monotonicity only at `normal > easy` (5.5-6.5 over 12 games); all other pairwise ladder checks passed.
- Added root low-mobility extension for positions with <=3 legal quiet moves and <=8 pieces, so squeeze-like positions get selective child-depth instead of globally raising every move budget.
- Adjusted strict L1/L2 budgets for low-mobility positions and disabled post-search skill weakening when there are <=3 legal moves.
- Latest fresh quick benchmark after these changes: easy solve 97%, blunder 0%; normal/hard/expert solve 100%, blunder 0%. Remaining easy miss is `small-piece-king-vs-men` with score drop 155, not severe.
- `npm run test:rules`, `npm run test:tactical`, and `npx tsc --noEmit` pass after the latest changes.
- Next validation gate: rerun `npm run bench:ai:teacher:fresh` and then `npm run bench:ai:full:fresh`; quick is green but teacher/full are needed before release confidence.
- First teacher/deep-oracle run found one apparent severe miss in `all-kings-2v1-corner-win`.
- Follow-up exact tablebase probe showed the chosen move and oracle move are both winning with the same root score, so the teacher miss was a benchmark oracle/scoring false positive.
- Benchmark oracle now probes exact small endgames before root selection and before scoring candidate moves.
- Teacher/deep-oracle rerun started on 2026-05-03 07:09 ICT as background process.
- Current teacher rerun log: `teacher-benchmark-rerun.log`.
- Current pipeline/watch log: `deep-benchmark-watch.log`.
- Current teacher checkpoint: `.tmp/benchmarks/ai-benchmark-teacher-checkpoint.json`.
- Teacher rerun completed with 100% solve and 0% blunder for all L1-L4.
- Full fresh benchmark after teacher completed at 2026-05-03 10:16 ICT.
- Full benchmark `releaseGatePassed=true`.
- Full benchmark head-to-head completed 72/72, but the L1-L4 ladder is still noisy/non-monotonic and needs a separate ladder-tuning pass.
- Added strict-level skill-gap move selection: lower levels may choose a safe near-best quiet move in non-critical positions, while forced captures, tactical overrides, and small endgames keep the best move.
- Rebalanced strict level budgets/depths: L1/L2 are lighter, L4 is deeper, tactical guards remain shared.
- Quick benchmark after ladder tuning: easy solve 97%, blunder 0%; normal/hard/expert solve 100%, blunder 0%.
- Full ladder-tuning benchmark should be rerun from scratch to verify head-to-head monotonicity.
- Added `npm run bench:ai:analyze` to summarize tactical gates and pairwise ladder violations from the latest full report.
- Full ladder benchmark is currently running in background: `full-benchmark-ladder.log`.
- A watcher will run `npm run bench:ai:analyze` into `ladder-analysis.log` after the full ladder benchmark finishes.
- Full benchmark completed and produced `.tmp/benchmarks/ai-benchmark-full-latest.json`.
- Latest full result passed tactical solve/blunder for most levels, but release gate still failed because expert had one severe tactical miss and the head-to-head ladder is not monotonic.
- Added separate teacher/deep-oracle benchmark mode for stricter tactical analysis without running head-to-head.

Important commands:
- Teacher benchmark: `npm run bench:ai:teacher *> teacher-benchmark.log`
- Watch teacher progress: `Get-Content .\teacher-benchmark.log -Tail 120 -Wait`
- Resume teacher benchmark: `npm run bench:ai:teacher *> teacher-benchmark-resume.log`
- Fresh teacher benchmark: `npm run bench:ai:teacher:fresh *> teacher-benchmark-fresh.log`
- Resume full benchmark: `npm run bench:ai:full *> full-benchmark-resume.log`
- Fresh full benchmark: `npm run bench:ai:full:fresh *> full-benchmark-fresh.log`
- Quick benchmark: `npm run bench:ai`
- Fresh quick benchmark: `npm run bench:ai:fresh`

Recent full tactical summary seen in log:
- easy: solve 97%, blunder 0%, avgDepth 2.9
- normal: solve 100%, blunder 0%, avgDepth 4.2
- hard: solve 97%, blunder 0%, avgDepth 4.9
- expert: solve 97%, blunder 3%, avgDepth 5.3

Known observation from full tactical log:
- `small-piece-men-race` still has promotion/endgame-race misses in lower levels.
- Some misses are not immediate giveaway blunders, but oracle-preferred moves differ.

Latest fresh quick benchmark after the teacher-mode update:
- easy: solve 100%, blunder 0%, avgDepth 1.3
- normal: solve 100%, blunder 0%, avgDepth 1.3
- hard: solve 100%, blunder 0%, avgDepth 2.1
- expert: solve 100%, blunder 0%, avgDepth 3.0

Current quick status:
- No tactical misses in the latest fresh quick run.
- The previous full severe miss `low-mobility-squeeze` is fixed in the fresh quick run.
- Benchmark classification now only treats `forced recapture trap` as safe when the oracle score drop stays small, so false-positive trap overrides are no longer hidden.

Latest teacher rerun after exact tablebase oracle fix:
- easy: solve 100%, blunder 0%, p95 1582ms
- normal: solve 100%, blunder 0%, p95 2084ms
- hard: solve 100%, blunder 0%, p95 3484ms
- expert: solve 100%, blunder 0%, p95 5685ms

Latest full fresh benchmark after teacher pass:
- `releaseGatePassed=true`
- easy: solve 100%, blunder 0%, p95 1591ms
- normal: solve 97%, blunder 3%, p95 2085ms
- hard: solve 97%, blunder 3%, p95 3483ms
- expert: solve 100%, blunder 0%, p95 5683ms
- head-to-head completed 72/72
- head-to-head matrix is still not monotonic: easy scored strongly vs normal/expert and normal scored strongly vs expert, so ladder tuning remains open.

Latest ladder-tuning changes:
- `selectStrictLevelMove()` applies skill gap after search, not inside rule/search correctness.
- Selector is disabled for exact tactical contexts: forced captures, root safety/trap/promotion overrides, and <=5-piece endgames.
- Quick oracle increased to 1500ms/depth 11 to avoid shallow false oracle misses in low-mobility positions.

Teacher/oracle mode:
- Keep current `bench:ai:full` as gameplay-budget benchmark.
- Use `npm run bench:ai:teacher` as the slower tactical teacher.
- Teacher mode uses higher oracle budget/depth and separate checkpoint/report files.
- Teacher mode skips head-to-head so it can be run often while tuning tactical positions.

## Goals

- Improve strength and stability for Level 1-4.
- Reduce tactical blunders: hanging pieces, missed forced captures, missed multi-capture chains, and bad recapture decisions.
- Keep mobile move time responsive.
- Replace case-by-case patches with repeatable correctness tests, benchmark data, and engine-level policies.

## Current Architecture Notes

- Rule engine: `src/coreClaude/movegen.ts`
- Position/draw state: `src/coreClaude/position.ts`
- Evaluation: `src/coreClaude/eval.ts`
- Main alpha-beta search: `src/coreClaude/search/alphabeta.ts`
- L1-L4 gameplay policy: `src/ui/HumanVsCodexArenaScreen.tsx`
- Existing smoke test: `scripts/tacticalSuite.ts`

## Five-Layer Problem Model

### 1. Rule Correctness

Risks:
- `generateMoves` and `hasCapturesAvailable` are separate implementations and can drift.
- Max-capture and multi-capture bugs can poison every search/eval benchmark.
- Draw and repetition state can interact badly with transposition table reuse.

Required work:
- Add rule invariant tests for move legality, max-capture, capture-chain uniqueness, promotion, occupancy, side toggle, and draw clock.
- Add curated perft-style positions for men, kings, max-capture ties, promotion, blocked movement, and endgames.
- Add differential checks between fast capture detection and full move generation.

### 2. Search Correctness

Risks:
- Alpha-beta contains several interacting optimizations: aspiration, TT, IID, null move, razoring, reverse futility, probcut, LMR, LMP, and extensions.
- Some pruning assumptions are risky in forced-capture Thai Checkers positions.
- UI-level fallback guards can hide search bugs instead of fixing them.

Required work:
- Add search profiles: safe, balanced, aggressive.
- Disable or limit risky pruning inside forced-capture, immediate-recapture, promotion-race, and low-depth tactical nodes.
- Move final tactical safety checks into the engine layer so all callers share the same behavior.
- Instrument pruning counts and root override reasons.

### 3. Quiescence And Tactical Horizon

Risks:
- Stand-pat evaluation can be unsafe when a capture is forced.
- Delta pruning can incorrectly skip forced tactical continuation.
- Low levels can stop before seeing recapture chains.

Required work:
- Treat forced-capture qsearch nodes as non-stand-pat nodes.
- Extend qsearch for capture chains, immediate recaptures, and promotion-after-capture cases.
- Track qnodes and q-depth in benchmark logs.

### 4. Evaluation Quality

Risks:
- Current eval is mostly material, PSQT, mobility, promotion threat, back-rank guard, simplification, and king endgame guidance.
- It lacks a cheap integrated model for hanging pieces, recapture risk, exchange safety, and promotion races.

Required work:
- Add cheap root/leaf tactical features without calling full movegen at every leaf.
- Add recapture-risk and hanging-piece signals.
- Add promotion-race distance and king trap signals.
- Tune weights through benchmark outcomes, not isolated visual cases.

### 5. Time Management Per Level

Risks:
- L1-L4 use fixed/adaptive depth and time from UI, not measured node-rate.
- Mobile p95 can regress even if desktop tests look good.
- Level differences can become inconsistent when tactical guard overrides too often.

Required work:
- Centralize level policy outside UI.
- Add node-rate-aware budgets.
- Increase time only for volatile tactical positions.
- Track avg/p95 elapsed time, timeout rate, reached depth, and override rate.

## Phase A: Correctness-First

Status: Completed for the current cleanup/correctness pass.

Tasks:
- [x] Create this roadmap.
- [x] Clear generated project clutter.
- [x] Add rule invariant test script.
- [x] Add curated rule/perft positions.
- [x] Add differential check for `hasCapturesAvailable`.
- [x] Review qsearch stand-pat behavior in forced-capture nodes.
- [x] Add benchmark logging schema.
- [x] Add shared L1-L4 benchmark script.
- [x] Centralize strict level policy outside UI.

Risks:
- Tests may expose existing behavior changes that affect AI choices.
- Tightening qsearch may slow tactical positions.

Testing:
- `npm run test:tactical`
- New rule invariant script.
- TypeScript compile check.

Expected impact:
- Higher confidence that later strength benchmarks measure engine quality, not rule bugs.
- Fewer capture-chain and legality regressions.

## Phase B: Strength-First

Status: In progress.

Tasks:
- [ ] Add safe/balanced/aggressive search profiles.
- [x] Make tactical-node pruning policy explicit for forced-capture qnodes.
- [x] Add engine-level root verifier diagnostics.
- [x] Strengthen engine-level anti-hanging root safety for immediate capture blunders.
- [x] Add endgame forced-bait/recapture trap detection so sound sacrifices are not blocked by anti-hang safety.
- [x] Expand forced-bait/recapture trap detection to opening and midgame positions.
- [x] Support multi-piece sacrifice traps such as giving 2 pieces to win 3 back when the forced recapture sequence is sound.
- [x] Improve 2v1/all-kings finisher handling with stronger tablebase budget and all-kings pursuit evaluation.
- [ ] Add cheap tactical eval features.
- [x] Build L1-L4 tactical benchmark suite.
- [x] Expand tactical case bank across forced capture, recapture, promotion, king, endgame, low-mobility, and opening buckets.
- [x] Export benchmark reports as JSON under `.tmp/benchmarks/`.
- [x] Add full benchmark progress logging and checkpoint/resume support.
- [ ] Run ablation tests for each search/eval change.
- [x] Add full benchmark release-gate checks for tactical solve, blunder, and timing thresholds.

Risks:
- L1 may become too strong if safety correction is too aggressive.
- Eval features can add cost if implemented with full move generation.

Testing:
- Tactical solve rate by bucket.
- Blunder rate against high-depth oracle.
- Old-vs-new self-play and head-to-head matrix.

Expected impact:
- Fewer one-ply and two-ply tactical blunders.
- More consistent difficulty ladder.

## Phase C: Speed And UX Tuning

Status: Pending deeper benchmark data.

Tasks:
- [ ] Add node-rate-aware time manager.
- [ ] Measure desktop and mobile p50/p95 move time.
- [ ] Cache root tactical analysis.
- [ ] Revisit worker strategy for mobile responsiveness.
- [ ] Tune level budgets to preserve UX.

Risks:
- Mobile performance may vary widely by device.
- Dynamic budgets can make benchmarks noisy without good logs.

Testing:
- Avg, p50, p95, and timeout rate per level.
- Real-device gameplay smoke test.
- Arena head-to-head under gameplay budgets.

Expected impact:
- Better tactical stability without making the app feel slow.

## Shared Benchmark Design

Metrics:
- Tactical solve rate by bucket.
- Blunder rate against oracle.
- Average move time and p95 move time.
- Reached depth and node count.
- Timeout/fallback rate.
- Root safety override rate.
- Head-to-head matrix for L1-L4, alternating P1/P2.

Suggested benchmark buckets:
- Forced single capture.
- Forced multi-capture.
- Max-capture choice.
- Quiet move that hangs a piece.
- Capture that loses to recapture.
- Promotion race.
- King fly capture.
- Small-piece endgame.
- All-kings endgame.

Minimum log fields:
- `positionId`
- `positionHash`
- `side`
- `level`
- `budgetMs`
- `maxDepth`
- `elapsedMs`
- `reachedDepth`
- `nodes`
- `qnodes`
- `score`
- `legalMoves`
- `forcedCapture`
- `maxCaptureLen`
- `chosenMove`
- `oracleMove`
- `oracleScore`
- `scoreDrop`
- `overrideReason`

Implemented scripts:
- `npm run test:rules`
- `npm run test:tactical`
- `npm run bench:ai`
- `npm run bench:ai:fresh`
- `npm run bench:ai:full`
- `npm run bench:ai:teacher`
- `npm run bench:ai:analyze`

## Release Gate

Pass criteria:
- Rule invariant and curated perft tests pass 100%.
- No legal position returns no AI move.
- Tactical solve rate: L1 >= 70%, L2 >= 80%, L3 >= 88%, L4 >= 93%.
- Severe blunder rate: L1 <= 8%, L2 <= 5%, L3 <= 3%, L4 <= 1.5%.
- Mobile p95 move time target: L1 <= 1.8s, L2 <= 2.8s, L3 <= 4.5s, L4 <= 7.0s.
- Head-to-head ladder is monotonic after alternating sides.
- Root override rate is low enough that the engine is not relying on patchy fallback behavior.

Fail criteria:
- Any rule invariant failure.
- Any engine stall on a legal position.
- Any repeated capture-chain regression.
- p95 move time exceeds target by more than 20%.
- L4 loses head-to-head to L3 over a meaningful sample after side balancing.

## Immediate Next Steps

1. Remove ignored generated folders and tracked Python bytecode. Done.
2. Add `scripts/ruleInvariantSuite.ts`. Done.
3. Wire `npm run test:rules`. Done.
4. Run `npm run test:rules`, `npm run test:tactical`, and `npx tsc --noEmit`.
5. Then start qsearch correctness changes. Done: forced-capture qnodes no longer use stand-pat cutoff.
