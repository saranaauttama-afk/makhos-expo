# Tactical Pattern Catalog

## Scope

Phase G1.1 is research/docs only.

No runtime behavior was changed.

No eval tuning was performed.

No search behavior was changed.

This catalog is based on direct inspection of these benchmark cases:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `low-mobility-squeeze`
- `quiet-hanging-piece-p1`

The goal is to identify recurring tactical motifs before encoding any heuristic.

---

## Case Snapshot

### `sac-two-win-three-p1`

- current role:
  - protected tactical guardrail
  - baseline-unstable
  - oracle-sensitive
- recurring pattern:
  - forced-recapture / fake-sacrifice family with promotion and king-entry options
- current engine profile:
  - structurally understands that the position is winning often enough to stay inside the BD4 accepted band
  - but exact preferred move is unstable under quick oracle fallback / override interaction

### `sac-two-win-three-p2`

- current role:
  - protected tactical guardrail
  - stable clean case
- recurring pattern:
  - same broad trap family as `p1`, but in a cleaner and more stable form
- current engine profile:
  - handled well
  - useful as the stable exemplar for this tactical family

### `low-mobility-squeeze`

- current role:
  - monitored tactical weakness
  - unstable / override-sensitive
- recurring pattern:
  - edge squeeze with very low liberty and a forced tactical ladder after the only viable move
- current engine profile:
  - base eval under-explains it
  - lower-strength practical handling has historically depended on override/search behavior more than direct eval understanding

### `quiet-hanging-piece-p1`

- current role:
  - monitored collateral-sensitive case
  - tactical / positional hybrid
- recurring pattern:
  - quiet loose-piece structure where the punishment is tactical but not fully explained by the current static hanging term
- current engine profile:
  - usually solved by search
  - fragile as a collateral target when unrelated tactical heuristics move

---

## Motif Catalog

| Motif | Structural description | Likely tactical meaning | Current engine handling | Stability / sensitivity | Focus cases |
|---|---|---|---|---|---|
| Forced recapture trap | Sparse position where one side can make a move that leaves the opponent with one forced capture or one narrow reply family, after which the tactical balance swings hard | Bait the opponent into an exchange ladder that improves the initiator's material, promotion path, or king activity | Mixed. The engine already has some search/override awareness for this family, but `p1` shows unstable move selection while `p2` is clean | Split: stable in `p2`, oracle-sensitive in `p1` | `sac-two-win-three-p1`, `sac-two-win-three-p2`, `low-mobility-squeeze` |
| Fake sacrifice structure | A move looks like a concession or loose drop, but the opponent's best-looking capture or continuation is tactically poisoned | "Take and lose" pattern; tactical bait rather than true blunder | Partial. The engine can often find it, but broad encoding risks overfitting protected cases | High risk when tied to protected cases; often search-sensitive and sometimes oracle-sensitive | `sac-two-win-three-p1`, `sac-two-win-three-p2` |
| King-entry race | Promotion or near-promotion move changes the tactical race immediately, often turning a loose-looking line into a winning king tempo | Promotion is not just material gain; it is a forcing tactical tempo and mobility swing | Partial to good in stable cases, but exact ordering is noisy in `p1` because multiple king-entry continuations are close | Oracle-sensitive when many winning king-entry lines are near-equal | `sac-two-win-three-p1`, lightly `sac-two-win-three-p2` |
| Mobility collapse / low-liberty squeeze | Tiny all-men cluster near the edge/back rank where legal moves exist but most are bad, funnelled, or tactically forced | Positional crowding has immediate tactical consequences; low liberty becomes a forcing tactical sequence, not just a static inconvenience | Poorly explained by the base eval; practical handling has depended on overrides or search | Override-sensitive; structurally stable enough to study, but risky to score broadly | `low-mobility-squeeze` |
| Edge-lock shape | Men are packed on or near an edge/double-corner lane, with one side owning the escape squares and the other side forced into a narrow corridor | Spatial clamp that converts into forced capture ladders or promotion races | Weak in direct static explanation; tends to be discovered tactically rather than understood positionally | Moderate structural stability, but can interact badly with broad mobility heuristics | `low-mobility-squeeze`, secondarily `quiet-hanging-piece-p1` |
| Quiet hanging tactical punishment | A piece is not obviously hanging by the current local static detector, but one quiet prep move or one narrow tactical reply makes the loose piece punishable | "Quietly doomed" piece; tactical liability hidden behind one tempo or one forced capture choice | Usually solved by search; static hanging term under-explains it | Collateral-sensitive / noisy rather than oracle-driven | `quiet-hanging-piece-p1` |

---

## Grouped By Phase

### Opening

- No pure opening-specific exemplar comes directly from the four G1.1 focus cases.
- The forced-recapture / fake-sacrifice family clearly has opening relatives elsewhere in the benchmark set:
  - `opening-bait-double-recapture-p1`
  - `opening-bait-double-recapture-p2`
- For this catalog, opening classification should be treated as a family extension, not a direct focus-case result.

### Midgame

- forced recapture trap
- fake sacrifice structure
- quiet hanging tactical punishment

### Endgame

- king-entry race
- mobility collapse / low-liberty squeeze
- edge-lock shape

### Tactical-only

- forced recapture trap
- fake sacrifice structure
- quiet hanging tactical punishment

---

## Stable vs Sensitive Read

### Stable motifs

- `sac-two-win-three-p2` shows that the broad forced-recapture / fake-sacrifice family is real and engine-solvable when the continuation is cleaner
- the low-liberty edge squeeze in `low-mobility-squeeze` is structurally crisp even though the engine does not score it directly

### Oracle-sensitive motifs

- the `sac-two-win-three-p1` cluster is not a clean single-move lesson
- it is better treated as:
  - a family of near-equivalent winning trap / promotion continuations
  - plus oracle fallback instability
- this makes it unsafe as the first direct encoding target

### Collateral-sensitive motifs

- `quiet-hanging-piece-p1` is the main collateral warning pattern
- it is tactically meaningful, but broad tactical adjustments can move it unintentionally even when the main target is elsewhere

---

## Recommended Future Encoding Order

### Safest first candidate motif

Safest first candidate later:

- **narrow low-liberty / edge-lock detection in sparse no-king positions**

Why this is the safest first candidate:

- it is the most geometrically crisp motif in the current focus set
- it is less oracle-sensitive than `sac-two-win-three-p1`
- it is easier to scope narrowly:
  - sparse material
  - no kings
  - edge-packed formations
  - low legal-move count
- existing docs already show that the current base eval under-explains this shape, so the signal gap is real

Important caution:

- this does **not** mean "add a broad mobility bonus/penalty"
- prior tiny mobility tuning already showed protected-case risk
- if this motif is ever encoded, it should start as:
  - instrumentation only
  - or an extremely narrow guarded signal

### G1.2 planning refinement

The current narrow planning direction is now more specific:

- target only sparse no-king edge-lock structures
- do **not** treat this as a generic low-mobility signal
- require:
  - no kings
  - total pieces `<= 6`
  - side-to-move men `<= 3`
  - all side-to-move men in the back band
  - strong edge concentration
  - no free men
  - compressed forward escape squares

Planned trigger set:

- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`

Planned protected non-trigger set:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `quiet-hanging-piece-p1`
- `quiet-hanging-piece-p2`
- opening trap-family cases

Safest implementation direction later:

- add a dedicated helper in `src/coreClaude/eval.ts`
- keep it separate from `mobilityScore(...)`
- expose it in breakdown / instrumentation first

### Motifs too risky to encode first

- **protected-case fake-sacrifice / forced-recapture family**
  - too entangled with `sac-two-win-three-p1` oracle instability
  - too easy to overfit to one protected benchmark family

- **broad mobility-collapse scoring**
  - prior tiny mobility changes already caused protected tactical regressions
  - the shape should only be approached through narrow structural guards

- **generic quiet-hanging tactical scoring**
  - current static notion of "hanging" is incomplete for this case
  - broadening it too early risks collateral movement on non-target cases

- **generic king-entry race bonuses**
  - tempting because `p1` contains promotion race energy
  - but the winning continuations are clustered and unstable enough that a direct bonus would likely overfit

---

## Practical Takeaways

1. The focus set is not random noise; it contains a coherent tactical family:
   - trap / recapture bait
   - sparse edge squeeze
   - quiet loose-piece punishment

2. The cleanest stable exemplar is not `sac-two-win-three-p1`; it is `sac-two-win-three-p2`.

3. The cleanest structural signal gap is not the protected trap family; it is the sparse low-liberty edge squeeze.

4. The main collateral guardrail for future work is `quiet-hanging-piece-p1`.

5. The safest next runtime step, if Phase G later moves past docs, is still:
   - narrow instrumentation around low-liberty / edge-lock structure
   - not broad tactical heuristics
   - and not protected-case trap bonuses

6. The G1.2 planning conclusion is that low-liberty should be encoded, if at all, as a strict sparse edge-lock detector with explicit non-trigger guards for trap-family and quiet-hanging positions.
