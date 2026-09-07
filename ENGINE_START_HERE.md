# Makhos Engine — START HERE

This file is the persistent handoff for future chats/agents. Read this file before changing the engine.

## Current canonical branch

- Branch: `engine-mainline`
- Created from checkpoint: `db28e143bb06fb8a35a643877a055c5df9770020`
- Purpose: clean continuation of the strongest classical Thai-checkers engine work before the project moved heavily into NN/mobile experiments.
- Old branches (`tuneNN`, `tuneCodex`, `tuneGPT`, `tuneAI`, `ml`, etc.) are reference/archive branches. Do not delete or merge them blindly.

## Product goal

Build a Thai checkers AI that is competitive / near-perfect in practical play and can ultimately run on mobile.

The intended path is not “NN at all costs”. The working strategy is:

1. Build a trusted, measurable classical engine (Alpha-Beta/PVS) that can act as a teacher and reference opponent.
2. Strengthen tactics, positional evaluation, opening knowledge, and endgames.
3. Generate high-quality training data using deep search/tablebases/self-play.
4. Train policy/value models with proper train/validation/test separation.
5. Compare pure classical, pure NN, and hybrid NN+search approaches.
6. Promote a model/engine to production only after statistically meaningful strength tests and mobile deployment checks.

## Non-negotiable rules

- Rules are **Thai checkers**, not American/English draughts.
- Source code is the truth. Old roadmaps/session notes may be stale.
- Inspect the actual implementation before architecture changes.
- Never claim an engine improvement from intuition alone.
- Every search/eval/pruning change must be measured against a fixed baseline.
- Preserve or improve rule correctness, tactical regression results, and solved/tablebase positions.
- Do not tune exclusively to the known puzzle set.
- Keep development puzzles and holdout puzzles separate.
- Do not use training data as validation/test data.
- Do not call a small self-play sample “proof”. Track uncertainty/statistical significance.
- Mobile deployability is a requirement from the architecture stage, not an afterthought.

## What “stronger” means

A candidate is considered stronger only when evidence agrees across multiple axes:

- Rule/perft correctness: no regression.
- Tactical/puzzle strength: more correct solutions and/or same solutions at lower depth/nodes/time.
- Head-to-head strength: candidate beats baseline over paired, opening-diverse games with enough samples.
- Endgame accuracy: high agreement with solved/tablebase positions where available.
- Search efficiency: nodes, depth, NPS, and wall-clock are tracked.
- Stability: repeated runs do not reveal fragile wins/regressions.

Puzzle solve rate by itself is **not** sufficient proof of whole-game strength.

## Known repository state at branch creation

The branch already contains substantial engine/testing work, including:

- `src/coreClaude/` classical engine implementation.
- Alpha-Beta/PVS search and evaluation code.
- opening-book related code and researched Thai opening patterns.
- endgame tablebase code.
- tactical/rule/perft/AI benchmark/regression scripts referenced by `package.json`.
- puzzle fixtures/solver scripts and historical puzzle reports.
- `src/coreClaude/search/timeManager.ts` added at the checkpoint. Its runtime integration must be verified before assuming it is active.

Historical puzzle report `PUZZLE_BASELINE.md` recorded only **1/14 expected first moves correct (7%)** on an earlier run. Treat this as historical evidence, not the new canonical baseline. Re-run and validate puzzle definitions/solutions before using the number as a gate.

## Immediate next step

Before modifying playing strength, establish **Teacher Baseline v0** on `engine-mainline`:

1. Verify TypeScript/build health.
2. Run rule invariant suite.
3. Run perft suite.
4. Run tactical suite.
5. Run AI/regression benchmark(s).
6. Run the puzzle suite and validate the expected solutions.
7. Record nodes/depth/time/NPS and machine/runtime details.
8. Inspect whether adaptive time management is actually integrated into iterative deepening.
9. Identify which historical branch tools are worth cherry-picking, but do not cherry-pick strength changes without A/B evidence.

The detailed sequence and promotion gates are in `docs/ENGINE_ROADMAP.md`.

## Standard workflow for every engine experiment

1. Create/identify baseline commit.
2. State one hypothesis.
3. Make one coherent change.
4. Run correctness gates.
5. Run tactical/puzzle tests.
6. Run paired head-to-head games vs baseline.
7. Record results in `docs/ENGINE_EXPERIMENT_LOG.md`.
8. Keep only changes that pass the agreed evidence threshold.
9. Revert failed experiments cleanly; preserve the result in the log.

Prefer small, attributable improvements over large bundles of changes.

## Mobile target

Neural/hybrid work must remain exportable to a practical mobile runtime. Current preferred evaluation path is ONNX/ONNX Runtime Mobile (or another mobile runtime if measurement later justifies it).

For every future NN architecture, track at minimum:

- model file size,
- operator/runtime compatibility,
- desktop-vs-export numerical parity,
- Android/iOS load success,
- single-position inference latency on representative mid-range hardware,
- memory footprint.

ONNX Runtime provides a mobile model usability checker and supports reduced-size ORT-format builds; use those checks before investing heavily in a model architecture.

## External engineering references

Useful methodology references (not Thai-checkers rules sources):

- Stockfish Fishtest: distributed head-to-head engine testing with Elo/statistical analysis and SPRT concepts: https://github.com/official-stockfish/fishtest
- ONNX Runtime Mobile deployment flow: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime mobile model usability checker: https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html

## How to resume in a new ChatGPT chat

Tell the new chat:

> Work on `saranaauttama-afk/makhos-expo`, branch `engine-mainline`. Read `ENGINE_START_HERE.md`, `docs/ENGINE_ROADMAP.md`, and `docs/ENGINE_EXPERIMENT_LOG.md` first. Inspect source before changes. Do not change engine strength without benchmark evidence.

Then continue from the first unchecked roadmap item or the latest experiment-log entry.
