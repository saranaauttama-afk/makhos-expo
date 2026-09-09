# Phase 3E move-ordering ablation protocol

## Frozen identity and separation

Phase 3E starts at canonical `engine-mainline` tip
`07a3f052904a4e72095b694d4e546f16c4162afa` and measures frozen Teacher v1
`b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`. The production defaults remain
`smallEndgame=false` and `soundForcedTrap=true`. The config pins every search,
extension, and ordering flag, so later defaults cannot silently change replay.
No evaluation, reduction, pruning, extension, TT replacement, or qsearch rule is
changed; only qsearch ordering is in scope.

## Current-code audit (before outcomes)

`orderMoves` assigns additive scores and uses stable descending JavaScript
sorting. Ties retain generator order. The six requested candidates exist as
these distinct mechanisms:

| Candidate | Exact effect on score/order and state |
|---|---|
| TT move | `orderMoves`: matching `(from << 5) | to` gets +2,000,000. The key comes from the current search-state TT probe, including a move populated by IID. This flag does not disable TT probing, bounds, storage, or replacement. |
| Capture ordering | `orderMoves`: every capture gets +100,000 plus +10,000 per captured piece. In quiescence, capture moves are secondarily sorted by descending capture length. This does not alter mandatory-capture legality or which qsearch moves are searched. |
| Killer moves | Two quiet beta-cutoff move keys per ply receive +8,000 and +7,000. A quiet cutoff shifts/records them. They are cleared at every search. |
| History heuristic | The 1,024-entry from/to key receives its current value as a score. Quiet beta cutoffs add `depth²`, capped at 30,000. Deterministic searches clear it; production timed searches age it by halving. The tournament measurement isolation also clears it before each move. |
| Countermove heuristic | The quiet move most recently producing a beta cutoff in reply to `prevMoveKey` receives +6,000. The reply table is updated at quiet cutoffs and cleared at every search. |
| Recapture ordering | Quiescence only: when the prior qsearch capture supplies a square, captures landing on that square sort before other captures; descending capture length breaks ties. Despite the historical comment, the propagated value is the prior capturing move's `from` square. There is no separate main-search recapture bonus. |

Two additional order influences are deliberately not candidates: root policy
hints add rounded hint ×120,000, and quiet non-kings receive a ±20-per-row
forward-advance score. ProbCut's top-three capture-length scan is pruning
candidate selection, not the common move sorter, and is left unchanged.
Disabling killer/history/countermove also disables that mechanism's learning;
disabling TT ordering leaves TT correctness semantics intact. Instrumentation
records score matches, first-after-ordering events, and updates where relevant.

## Frozen corpus and partitions

The unchanged `xorshift32/legal-sorted/diverse-plies-2-17/v1` generator uses
seed `0x07a3f052` (128184402), the first eight hex digits of the canonical tip.
After complete final states colliding with any retained Phase 3A, 3B, 3C, or 3D
corpus are removed, the first 64 states form
`makhos-phase3e-move-ordering-starts-v1`, SHA-256
`00db4d9cedfe76202dae15373d33bfdc89f31726ad2b9b4e0fe3438701e97091`.
The suite/version/fingerprint and rules below were frozen before outcomes.

Starts 1–32 are screening-only: paired colors, 5,000 nodes/move, depth cap 64,
and 160 plies. Each candidate differs from the explicit baseline in exactly one
false ordering flag. Select exactly one candidate: the inventory-order-stable
highest candidate score strictly above 50%. If none exceeds 50%, record
inconclusive and stop. Screening never promotes a default.

Only if selected, use untouched starts 33–64 for independent confirmation at
20,000 nodes/move, 50,000 nodes/move, and 100 ms/move, paired colors, depth cap
64, and 160 plies. The time run is descriptive because wall time is not the
canonical deterministic comparison.
