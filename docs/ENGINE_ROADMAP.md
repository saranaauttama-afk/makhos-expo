# Makhos Competitive Engine Roadmap

Status legend: `[ ]` not started, `[-]` in progress, `[x]` complete and evidenced.

## North-star target

A Thai-checkers engine that is strong enough for competitive play, approaches optimal play in tractable positions, provides a trustworthy teacher for ML, and can ultimately be deployed on mobile.

No phase is considered complete because code exists. Completion requires measurements and recorded evidence.

---

# Phase 0 — Establish a trustworthy baseline

Goal: know exactly what `engine-mainline` can do before changing strength.

- [x] Confirm branch/source checkpoint and inventory active engine modules.
- [x] Verify build/typecheck on the installed clean dependency tree.
- [x] Run `npm run test:rules`.
- [x] Run `npm run test:perft`.
- [x] Run `npm run test:tactical`.
- [x] Run `npm run bench:ai:fresh` and relevant full/teacher variants.
- [x] Run `npm run regression:harness` / existing AI gates.
- [x] Make puzzle suite runnable from a stable npm command if currently missing.
- [x] Validate puzzle fixtures against their original/authoritative solutions where possible.
- [x] Record puzzle first-move accuracy, full-line accuracy where available, depth, nodes, time.
- [x] Record machine, Node version, CPU, runtime settings and commit SHA.
- [x] Verify whether `timeManager.ts` is integrated into production iterative deepening or is only a scaffold.
- [x] Create `Teacher Baseline v0` record in the experiment log.

**Exit gate:** correctness suites pass, measurements are reproducible, baseline commit and test protocol are frozen.

---

# Phase 1 — Rules/search correctness hardening

Goal: remove subtle search bugs before tuning evaluation.

## Rules and state

- [x] Re-audit Thai capture rules, multi-capture, promotion timing, flying king behavior, mandatory capture, no-legal-move result, repetition and inactivity draw. Phase 1C follow-up removed the incorrect majority-capture rule: any complete capture sequence may be chosen. `docs/THAI_RULES_SPEC.md` separates externally supported Makhos rules from unresolved 16/32-ply project draw policy.
- [x] Add targeted regression fixtures for every discovered rule bug. Phase 1C found and fixed two correctness bugs: incorrect global majority/max-capture filtering in move generation, and unsafe exact claims in production small-endgame probing caused by incomplete history/cache/deadline behavior.
- [x] Verify position hashing includes every state component that can change the game-theoretic result. Phase 1A split board/repetition identity from TT search-state keys and covered inactivity clock plus full repetition context.

## Search correctness

- [x] Audit quiescence search under mandatory-capture rules; Phase 1A proves stand-pat is used only at quiet nodes and adds repetition accounting throughout forced capture continuations.
- [x] Audit TT bounds, score normalization, mate/terminal score handling and collision safeguards. Phase 1B added bound/collision/replacement/reuse fixtures, key-zero occupancy tracking, mate-distance normalization and stopped-search store protection.
- [x] Audit null move, ProbCut, razoring, RFP, LMR, LMP, IID and extensions for Thai-checkers-specific tactical risk. Phase 1B recorded risks and prevents null move from crossing an imminent inactivity draw; strength/soundness ablations remain Phase 2/3 work.
- [x] Add feature flags for major pruning mechanisms so each can be ablated independently. Phase 1B exposes independent reverse-futility, razoring, null-move, ProbCut, IID, LMR, LMP and extensions flags without changing default thresholds.
- [x] Add deterministic fixed-depth and fixed-node test modes in addition to wall-clock mode. Phase 1A repeats both modes five times and compares move, score, main/q nodes, depth and PV.

**Exit gate:** no known correctness bug; pruning can be toggled and tested independently; deterministic search modes exist.

---

# Phase 2 — Measurement infrastructure

Goal: make “stronger” statistically defensible.

## Head-to-head harness

- [x] Baseline vs candidate from identical codebase with one controlled difference. Phase 2A validates `nullMove=false` plumbing without making a strength claim.
- [x] Paired colors for each opening/start position.
- [x] Opening diversity to reduce first-move/start-position bias. Phase 2A freezes a seeded legal measurement suite; it is not an authoritative opening book.
- [x] Support fixed time, fixed nodes and fixed depth.
- [x] Record W/D/L, score %, average nodes, depth, time, NPS.
- [x] Estimate Elo difference and confidence interval using a deterministic pair-level bootstrap.
- [ ] Add sequential-testing/SPRT-style stopping later if useful.
- [x] Save machine-readable JSON/CSV results, not only console text.

## Position suites

- [x] Freeze evidence-gated **development** and immutable **holdout** v1 sets with a canonical content fingerprint; exact state/history leakage and default case-level holdout blinding are tested. The 14 provenance-poor historical puzzles remain diagnostic rather than truth.
- [x] Add the Phase 2B motif taxonomy without forcing uncertain tags.
- [x] Add unlabeled strategic positions from replayable deterministic legal traces, explicitly separated from verified tactical truth.
- [x] Report verified development/holdout and per-motif denominators separately from legacy/provisional diagnostics.

**Exit gate:** any future engine patch can be A/B tested without manually modifying source between opponents.

**Phase 2 status:** **READY TO CLOSE.** Phase 2A supplies paired A/B testing and
Phase 2B supplies reproducible, leakage-checked position measurement with
evidence-only W/D/L adjudication and full capture-path artifact identity. The
verified v1 corpus is intentionally small and is not a strength claim.
Sequential/SPRT and distributed execution remain later optional scale work.

---

# Phase 3 — Classical engine strength

Goal: turn the engine into a strong Teacher v1.

Work one hypothesis at a time and keep only measured wins.

## Search

- [x] Complete Phase 3A–3C extension ablation and approve Teacher v1: same-corpus robustness at 5k/20k/50k nodes plus a new disjoint-corpus 20k/100 ms confirmation support disabling only `smallEndgame`, with all correctness and verified-position gates passing.
- [x] Move-ordering ablations: TT move, captures, killers, history, countermove, recapture ordering. Phase 3E selected countermove removal mechanically, but confirmation intervals crossed zero; no production default changed.
- [x] Tune LMR based on measured strength, not only speed. Phase 3F gathered controlled screening and untouched confirmation evidence without changing production defaults.
- [ ] Tune/select pruning thresholds using tournaments and tactical regressions.
- [ ] Improve time management only after fixed-node/depth strength is understood.
- [ ] Investigate aspiration windows if absent/useful.
- [ ] Investigate singular/forced-line extensions only if tactical evidence supports them.

## Evaluation

Potential features to test individually or in small groups:

- [ ] material values and phase scaling,
- [ ] real mobility for flying kings,
- [ ] promotion distance/race and breakthrough potential,
- [ ] trapped pieces / low-liberty structures,
- [ ] back-rank structure,
- [ ] king safety and king activity,
- [ ] protected/connected men,
- [ ] tempo/opposition/endgame geometry,
- [ ] hanging-piece / forced-capture vulnerability,
- [ ] sacrifice compensation,
- [ ] simplification only when objectively favorable.

Use self-play/position data for parameter tuning where appropriate, but confirm tuned parameters in holdout games.

**Exit gate:** **MET.** PR #11 merge commit `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f` **is the frozen Teacher v1 release identity**; corpus provenance uses GitHub-resolvable PR #11 historical commit `b59570adb3e94dc629fc5a9af262b96b9d030498`.

---

# Phase 4 — Thai-specific knowledge

Goal: add knowledge search alone learns inefficiently.

## Opening

- [ ] Audit existing opening-book content and legality.
- [ ] Validate Thai opening sequences against reliable Thai sources/game records.
- [ ] Store provenance for manually entered opening lines.
- [ ] Prefer weighted/repertoire variation over a single deterministic line.
- [ ] Measure book-on vs book-off full-game strength and diversity.

## หมากกล / tactical knowledge

- [ ] Expand puzzle corpus with known solutions and provenance.
- [ ] Do not hard-code puzzle answers into evaluation.
- [ ] Use puzzles to expose generic motifs/search weaknesses.
- [ ] Require holdout puzzle improvement before claiming general tactical learning.

**Exit gate:** opening knowledge improves practical strength without hiding underlying engine weakness; puzzle improvement generalizes to holdout positions.

---

# Phase 5 — Endgame / tablebases

Goal: approach perfect play where exact solving is tractable.

- [ ] Audit existing tablebase implementation and coverage.
- [ ] Verify all stored results against exhaustive small-material search.
- [ ] Define canonical position encoding and symmetry reductions.
- [ ] Expand solved material counts progressively subject to storage/time feasibility.
- [ ] Track W/D/L agreement between engine and tablebase.
- [ ] Use tablebase distances (if available) to prefer technically correct conversions.
- [ ] Feed exact solved positions into evaluation/training datasets with explicit labels.

**Exit gate:** 100% agreement on the verified tablebase domain and reliable handoff between search and tablebase.

---

# Phase 6 — Teacher data pipeline

Goal: create data good enough to train a model that can exceed shallow search.

Data must be diverse and leakage-controlled.

Sources:

- deep classical self-play,
- opening-diverse games,
- tactical/critical positions,
- endgame tablebase positions,
- real-game positions when licensing/provenance permits,
- positions mined from losses/blunders.

For each sample, prefer retaining:

- board/state encoding,
- side to move,
- legal move set,
- searched move scores or visit/quality distribution when available,
- selected best move,
- search depth/nodes,
- game result,
- tablebase exact result where applicable,
- source/split identifier.

- [ ] Freeze data schema/version.
- [ ] Deduplicate near-identical positions where appropriate.
- [ ] Split by game/source before training to avoid leakage.
- [ ] Maintain immutable validation and test sets.
- [ ] Add data-quality checks and legality verification.

**Exit gate:** reproducible dataset generation with genuine train/validation/test separation.

---

# Phase 7 — Neural / hybrid engine

Goal: build a deployable model that improves playing strength and/or search efficiency.

Start small. Mobile compatibility is checked before expensive training.

Candidate roles:

1. **Value network** — replace/augment handcrafted leaf evaluation.
2. **Policy network** — prioritize moves and reduce effective branching factor.
3. **Policy + value** — guide Alpha-Beta/MCTS.
4. Pure policy move selection is allowed as a benchmark but is not assumed to be final architecture.

- [ ] Define feature encoding version.
- [ ] Train baseline value/policy model.
- [ ] Use true held-out validation/test data.
- [ ] Export model immediately (e.g. ONNX) and verify numerical parity.
- [ ] Run ONNX Runtime mobile usability/operator check.
- [ ] Benchmark inference latency and memory.
- [ ] NN vs Teacher tournament.
- [ ] Hybrid vs Teacher tournament.
- [ ] Hybrid vs pure NN tournament.
- [ ] Promote only statistically supported wins.

**Exit gate:** NN/hybrid engine beats the current classical baseline under comparable compute or achieves a clearly superior mobile strength/latency tradeoff.

---

# Phase 8 — Self-play improvement loop

Goal: move beyond imitation of the teacher.

Only begin after the supervised model and evaluation pipeline are trustworthy.

- [ ] Maintain opponent pool of historical models/engines.
- [ ] Use opening diversity.
- [ ] Mine losses and uncertain positions.
- [ ] Periodically regenerate/search-label difficult positions.
- [ ] Evaluate each new model against the previous champion and classical teacher.
- [ ] Require promotion gates; never auto-promote from training loss alone.

Possible approach: supervised bootstrap from strong search + self-play reinforcement/fine-tuning, rather than relying on pure self-play from random initialization.

**Exit gate:** repeated champion improvements verified by tournaments, not training metrics alone.

---

# Phase 9 — Mobile production

Goal: ship the strongest practical configuration on Android/iOS.

- [ ] Choose production architecture based on measured strength/latency, not preference.
- [ ] Define model-size and move-time budgets for representative devices.
- [ ] Test ONNX/ORT-format or selected alternative runtime on Android and iOS early.
- [ ] Verify operator compatibility (NNAPI/CoreML where beneficial).
- [ ] Quantization experiments require strength/parity checks.
- [ ] Benchmark cold-load time, warm inference, memory, app-size impact and battery/thermal behavior.
- [ ] Provide fallback engine path if model/runtime initialization fails.
- [ ] Run full rule/tactical/endgame integration tests in the actual mobile application.

**Exit gate:** production AI meets agreed competitive-strength and responsiveness targets on representative mobile hardware.

---

# Promotion gates for any new “best engine”

A candidate cannot become the new baseline unless:

1. Rule/perft tests pass.
2. No critical tactical regression is introduced.
3. Holdout puzzle performance is not materially worse.
4. Head-to-head data supports equal-or-better strength with sufficient sample size.
5. Performance cost is measured and acceptable.
6. Experiment and exact commit are recorded in `docs/ENGINE_EXPERIMENT_LOG.md`.

For risky search changes, require stronger evidence than for pure speed refactors.

---

# First concrete milestone

**Teacher Baseline v0 → Teacher v1**

Do not start a new NN training cycle until Teacher v1 exists or there is a deliberate documented decision to train an experimental baseline model.

Teacher v1 should have:

- trusted rules/perft,
- reproducible tactical+puzzle baseline,
- paired A/B tournament harness,
- verified core search behavior,
- materially improved tactical/full-game strength,
- validated endgame integration for the currently supported tablebase domain.

Once that milestone is reached, begin the data/model track in parallel rather than abandoning classical-engine development.
