# ENGINE_PHASE_F_EVAL_INSPECTION

## Scope

Phase F.0 is inspection only. No evaluation behavior was changed.

## 1. Main Evaluation Entry Points

- Primary eval entry:
  - `evaluate(p)` in `src/coreClaude/eval.ts`
- Main implementation:
  - `handEvaluate(p)` in `src/coreClaude/eval.ts`
- Search call site:
  - `_eval(pos)` in `src/coreClaude/search/alphabeta.ts`
- Eval override hook for A/B testing:
  - `setEvalFn(...)`
  - `resetEvalFn()`
  - both in `src/coreClaude/search/alphabeta.ts`

## 2. Files Involved

- `src/coreClaude/eval.ts`
  - main hand-written evaluation
- `src/coreClaude/search/alphabeta.ts`
  - search-level eval caller and override hook
- `src/coreClaude/search/endgameTablebase.ts`
  - separate root/endgame exact probing, not the main static eval
- `src/coreClaude/bitboards.ts`
  - bit iteration, movement steps, board geometry helpers used by eval
- `src/coreClaude/position.ts`
  - position structure consumed by eval
- `scripts/evalAB.ts`
  - old-vs-new eval comparison harness

## 3. Current Evaluation Terms Found

Main terms in `src/coreClaude/eval.ts`:

- material
  - men value `100`
  - king value starts at `280`
  - king value scales upward in endgame
- piece-square tables
  - men advancement
  - column/structure preference
  - king centrality
- mobility
  - cheap approximation
  - no `generateMoves()` call
  - kings use approximate ray mobility
- back-rank guard
  - rewards promotion-stopper structure
- promotion threats
  - rewards men close to promotion if a forward lane is open
- king endgame proximity
  - rewards winning kings for hunting enemy men
- all-kings endgame pressure
  - edge pressure
  - king proximity pressure
- hanging pieces penalty
  - threatened and undefended men/kings
- simplification bonus
  - rewards trading when ahead

Historical note from file header:

- older explicit `promotionProgress`, `centerControl`, and `kingCentralization` terms were replaced by PSQT
- `captureNetScore` was removed for speed

## 4. Missing Evaluation Terms

Not found as separate current eval terms:

- explicit tempo bonus
- explicit center-control term outside PSQT
- explicit structure term beyond back-rank guard / PSQT / protected-men history
- explicit trapped-piece penalty term
- explicit connectedness / chain term
- explicit passer / unstoppable-runner term beyond promotion threat
- explicit king safety term as a named full feature
- explicit threat map bonus beyond hanging-piece detection
- explicit piece-square phase interpolation tables

Also notable:

- `protectedMenBonus(...)` still exists in code comments/history, but is currently omitted from final score

## 5. Whether Eval Is Symmetric / Deterministic

Deterministic:

- yes
- no randomness in the eval path
- same input position always yields the same eval

Symmetric:

- mostly yes, by side-to-move-relative scoring
- material, PST, mobility, promotion threat, and penalties are computed from the current side’s perspective and mirrored against the opponent

Important nuance:

- the eval is side-relative rather than absolute-white/black style
- `evaluate(p)` scores from `p.side`’s point of view, which matches negamax expectations

## 6. Whether Phase / Endgame Scaling Exists

Yes.

Current scaling found:

- endgame factor `eg`
  - `0` in opening/midgame
  - ramps toward `1` in low-piece positions
- king value scaling
  - `280 -> 380`
- back-rank guard scaled down by `(1 - eg)`
- king endgame bonus scaled by `eg * 0.35`
- all-kings endgame bonus scaled by `eg * 0.5`

Also outside static eval:

- root-level small endgame tablebase probing exists in search, but it is separate from the static evaluator

## 7. Whether Mobility / Trapped-Piece Logic Exists

Mobility:

- yes
- `mobilityScore(p)` exists
- uses step-square counting for men
- uses approximate ray mobility for kings
- intentionally avoids `generateMoves()` for performance

Trapped-piece logic:

- no explicit trapped-piece term found in `eval.ts`
- closest existing term is `hangingPiecesPenalty(p)`, which detects threatened/undefended pieces
- low-mobility and safety ideas also exist in search/root override logic, but those are not part of static eval itself

## 8. Whether Promotion-Race Logic Exists

Yes, partially.

In eval:

- `promotionThreatScore(p)` rewards near-promotion men with an open lane

In search:

- quiescence has a special promotion-threat exception to avoid pruning certain promotion-race positions
- root override logic also contains an `endgame promotion race` path

Assessment:

- promotion-race awareness exists
- but it is split across eval and search rather than being a single dedicated eval subsystem

## 9. Risks / Complexity Areas

1. Eval and search responsibilities are partially mixed.
   - some tactical/promotion behavior lives in search exceptions and root overrides rather than eval alone

2. Mobility is intentionally approximate.
   - this is good for speed
   - but it may miss some true trapped/low-liberty patterns

3. Hanging-piece logic is cheap, not deep.
   - it does not perform full tactical exchange analysis
   - useful signal, but can over/underestimate complex capture sequences

4. Endgame handling is split.
   - static eval has endgame bonuses
   - exact small-endgame solving lives in tablebase probing
   - tuning one without respecting the other can be misleading

5. PSQT carries multiple concepts at once.
   - advancement
   - center/column preference
   - king centrality
   - this keeps eval fast, but makes isolated tuning harder

6. Side-relative evaluation is correct for negamax, but easy to misuse in experiments.
   - any instrumentation or offline analysis needs to remember scores are from `p.side`

7. There is already evidence of prior term churn.
   - protected men removed from final score
   - earlier center/progress terms replaced
   - this suggests caution before piling on more overlapping heuristics

## 10. Recommended Safe F Roadmap

### F.1 Instrumentation

- add passive eval breakdown reporting only
- expose per-term contributions from `handEvaluate(...)`
- keep default search behavior unchanged
- use it to identify which terms dominate tactical misses

### F.2 Mobility Experiment

- isolate mobility term measurement first
- consider small weight or shape experiments only after instrumentation
- avoid switching from cheap approximation to full movegen-based mobility in hot path

### F.3 Trapped-Piece Penalties

- add a distinct trapped-piece / low-liberty term
- keep it separate from hanging-piece logic
- gate it or benchmark it independently so it does not get confused with mobility

### F.4 Promotion Race Evaluation

- strengthen promotion-race eval in a contained way
- keep search overrides unchanged during the first eval-only pass
- focus on unstoppable runner / blocked lane / king-vs-runner patterns

### F.5 Structure / Protection Revisit

- reassess whether `protectedMenBonus(...)` should remain omitted
- if revisited, benchmark it independently from trapped-piece changes

### F.6 Endgame Scaling Review

- inspect whether current `eg` ramp and king scaling are too coarse
- keep tablebase behavior separate from static eval tuning

### F.7 Constant Tuning Later

- only after instrumentation and isolated feature experiments
- avoid multi-term tuning in one pass

## Summary

The current evaluation is a fast, deterministic, side-relative hand-written evaluator centered on:

- material
- PSQT
- mobility
- promotion threats
- hanging-piece penalties
- endgame pressure/scaling

It already has some endgame and promotion awareness, but explicit trapped-piece, tempo, and richer structure terms are either missing or intentionally omitted. The safest next step is instrumentation first, then small isolated feature experiments rather than direct retuning.
