# G1.2 Low-Liberty / Edge-Lock Heuristic Plan

## Scope

Phase G1.2 is planning only.

No runtime behavior was changed.

No eval tuning was performed.

No search behavior was changed.

This note defines the narrowest plausible future heuristic target for:

- sparse no-king low-liberty structures
- edge-packed confinement
- trapped advancement lanes
- low escape count

The goal is to avoid:

- broad mobility scoring
- generic tactical bonuses
- protected fake-sacrifice family entanglement
- quiet-hanging collateral leakage

---

## Planning Objective

We do **not** want a general "mobility is low" signal.

We want a narrow structural detector for positions that look like:

- tiny all-men clusters
- back-rank-heavy
- near one board edge or corner lane
- very few forward exits
- no flexible central release squares

This is meant to describe `low-mobility-squeeze`-type positions only.

---

## Proposed Structural Signals

### 1. Low-liberty

Use forward-step topology for men only, not full generic mobility.

Candidate local measurements for the side to move:

- `blockedMen`
  - number of men with `0` legal forward step squares
- `crampedMen`
  - number of men with exactly `1` legal forward step square
- `freeMen`
  - number of men with `2+` legal forward step squares
- `totalForwardSteps`
  - total number of legal forward step squares across all side-to-move men
- `uniqueForwardTargets`
  - count of distinct forward destination squares

Interpretation:

- low-liberty target shape should have:
  - `freeMen = 0`
  - `totalForwardSteps <= 3`
  - `uniqueForwardTargets <= 2`

This is intentionally tighter than ordinary low mobility.

### 2. Edge-packed confinement

We need a structural edge signal, not just low move count.

Candidate measurements for the side to move:

- `nearEdgeMen`
  - men on files `c <= 2` or `c >= 5`
- `hardEdgeMen`
  - men on files `c <= 1` or `c >= 6`
- `backBandMen`
  - for P1: rows `5..7`
  - for P2: rows `0..2`
- `sameFlankSpan`
  - horizontal span of side-to-move men is small and edge-anchored

Interpretation:

- target shape should require:
  - all side-to-move men in the back band
  - at least `2/3` side-to-move men near the same edge flank
  - at least `1` man on the hard edge

This is what separates the squeeze family from the quieter hanging-piece family.

### 3. Trapped advancement lanes

We need to know whether legal forward squares are genuinely funnelled.

Candidate measurements:

- `forwardTargetCompression`
  - multiple men share the same one or two escape squares
- `edgeLaneOnly`
  - all legal forward targets remain on edge-adjacent files
- `noCentralExit`
  - no legal forward target lands in the central files

Interpretation:

- target shape should usually have:
  - compressed forward targets
  - no clean central breakout square

### 4. Low escape count

This should be explicit and strict.

Preferred activation threshold:

- side-to-move legal moves `<= 3`

If a future eval-only helper avoids full move generation, the cheap proxy should be:

- `totalForwardSteps <= 3`
- `uniqueForwardTargets <= 2`

If both are available:

- use cheap topology as the main helper
- use `legalMoves <= 3` only as an additional debug / benchmark-side confirmation

---

## Strict Activation Guards

These are the proposed hard guards for any future experiment.

### Required guards

1. no kings
   - `p1Kings === 0`
   - `p2Kings === 0`

2. sparse material
   - total pieces `<= 6`

3. side-to-move piece count small
   - side-to-move men `<= 3`

4. back-band confinement
   - all side-to-move men are on the back 3 rows relative to that side

5. edge-heavy structure
   - at least `2` side-to-move men on near-edge files
   - at least `1` side-to-move man on a hard-edge file

6. low-liberty threshold
   - `freeMen = 0`
   - `totalForwardSteps <= 3`
   - `uniqueForwardTargets <= 2`

### Optional stricter guards

Only if future activation still leaks:

7. no central anchor
   - no side-to-move man on the two most central playable files

8. compressed horizontal span
   - side-to-move men occupy at most a 3-file span

9. opponent close enough to matter
   - opponent has at least `2` men in the forward-contact zone

The optional guards should be considered only after basic instrumentation evidence.

---

## Positions Expected To Trigger

### Should trigger

#### `low-mobility-squeeze`

Why:

- no kings
- total pieces = `6`
- side to move has `3` men
- all side-to-move men are in the back band
- side-to-move structure is edge-leaning and compressed
- legal moves = `3`
- forward targets compress into `22, 22, 23`

This is the primary target case.

#### `low-mobility-squeeze-p2`

Why:

- no kings
- total pieces = `6`
- side to move has `3` men
- all side-to-move men are in the back band
- side-to-move structure is even more edge-packed than the P1 version
- legal moves = `3`
- forward targets compress into `10, 13, 14`

This should trigger as the mirrored secondary target.

---

## Positions Explicitly Protected From Triggering

These positions should **not** trigger under the future narrow detector.

### `quiet-hanging-piece-p1`

Why it must not trigger:

- although it is sparse and no-king
- it is not edge-heavy enough
- its main issue is quiet tactical looseness, not back-band edge-lock confinement

Main exclusion features:

- no hard-edge side-to-move man
- only weak near-edge concentration
- structure climbs through central files rather than an edge corridor

### `quiet-hanging-piece-p2`

Why it must not trigger:

- sparse and no-king alone is not enough
- shape is diagonal/central rather than edge-locked
- this is a hanging / tactical-liability family, not a squeeze-family exemplar

### `sac-two-win-three-p1`

Why it must not trigger:

- contains kings
- total pieces exceed the sparse no-king target
- legal move count is far too high
- belongs to the protected fake-sacrifice / trap family

### `sac-two-win-three-p2`

Why it must not trigger:

- contains kings
- belongs to the protected trap family
- not a low-liberty edge-lock structure

### `opening-bait-double-recapture-p1`

Why it must not trigger:

- too many pieces
- opening trap family
- back-rank-heavy but not sparse enough

### `opening-bait-double-recapture-p2`

Why it must not trigger:

- too many pieces
- opening trap family
- not a narrow squeeze target

### Any king-containing endgame fixture

Why it must not trigger:

- king mobility semantics are completely different
- the target motif is explicitly no-king only

---

## Proposed Feature Shape

### Recommended shape

Do **not** modify `mobilityScore(...)`.

Instead, if Phase G later moves to implementation, use a distinct helper such as:

- `lowLibertyEdgeLockSignal(p)`

That helper should:

1. early-return `0` unless all hard guards pass
2. compute only men-only local topology
3. measure:
   - blocked / cramped / free men
   - compressed target count
   - edge concentration
   - back-band confinement
4. return a very small signed value or debug-only breakdown field

### Recommended first scoring shape

If the first experiment goes beyond instrumentation:

- prefer a tiny discrete banded signal, not a smooth mobility term

Example conceptual shape:

- `0` unless hard guards pass
- then:
  - `+X` if opponent is more confined
  - `-X` if side to move is more confined

Important:

- no scaling by generic move count
- no adding to the main mobility formula
- no using capture depth or tactical SEE-like logic

---

## Expected Benefit

Likely benefit:

- better recognition of true squeeze / no-escape shapes in tiny no-king structures
- more direct coverage of `low-mobility-squeeze`
- cleaner structural signal than broad crowding or generic mobility penalties

Most realistic near-term benefit:

- better instrumentation / explanation first
- possible later improvement on:
  - `low-mobility-squeeze`
  - maybe `low-mobility-squeeze-p2`

---

## Expected Collateral Risk

### Main risks

1. quiet-hanging leakage
   - sparse no-king positions can look superficially similar
   - this is why edge-heavy and hard-edge guards are mandatory

2. protected trap-family interference
   - broad "cramped = bad" logic could distort tactical trap choices
   - this is why king and total-piece guards are mandatory

3. mirrored weakness overfitting
   - an overly literal edge rule may hit only one flank or one side correctly
   - mirrored target case `low-mobility-squeeze-p2` should be part of planning validation

4. broad mobility backdoor
   - if the helper starts rewarding any low move count, it becomes a disguised mobility rewrite

### Risk rating

- expected benefit: moderate
- expected collateral risk with hard guards: moderate but manageable
- expected collateral risk without hard guards: high

---

## Safest Implementation Location Later

Safest future runtime location:

- `src/coreClaude/eval.ts`

Safest placement inside that file:

- next to `lowMobilityResearchSignal(...)`
- as a separate dedicated helper
- surfaced through `EvalBreakdown`

Why this is safest:

- it keeps the work outside search
- it avoids `alphabeta.ts`
- it makes the signal inspectable in debug output
- it avoids contaminating the generic `mobilityScore(...)`

Do **not** start in:

- `alphabeta.ts`
- root overrides
- quiescence ordering
- generic move-order bonuses

---

## Safest OFF-by-Default Experiment Structure Later

### Step 1. Instrumentation only

Recommended first runtime experiment:

- add a breakdown/debug-only field:
  - `lowLibertyEdgeLock`
- keep contribution to final eval at `0`

Recommended flags:

- reuse `MAKHOS_ENABLE_EVAL_EXPERIMENTS=1`
- add a narrow flag such as:
  - `MAKHOS_ENABLE_G12_LOW_LIBERTY_EDGE_LOCK`

### Step 2. Tiny scored experiment

Only if instrumentation is clean:

- same narrow helper
- same hard guards
- tiny fixed score
- OFF by default

### Required benchmark protection order

When scored later, compare in this order:

1. `sac-two-win-three-p2`
2. `sac-two-win-three-p1`
3. `quiet-hanging-piece-p1`
4. `low-mobility-squeeze`
5. `low-mobility-squeeze-p2`

Reason:

- the target case must never outrank protected regression safety

---

## Current Planning Conclusion

The narrowest safe future path is:

- **a dedicated sparse no-king edge-lock helper**
- **not a mobility rewrite**
- **not a search change**
- **not a trap-family bonus**

The best trigger candidates are:

- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`

The main protected non-trigger set is:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `quiet-hanging-piece-p1`
- `quiet-hanging-piece-p2`
- opening trap-family benchmarks

If Phase G continues after planning, the safest next step is still:

- instrumentation first
- score later only if the trigger set is clean
