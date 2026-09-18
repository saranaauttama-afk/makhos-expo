# Phase 5A exact endgame audit

Phase 5A is an evidence-only foundation. It does not alter production search,
evaluation, or `probeSmallEndgame()`.

## Source audit

The production probe in `search/endgameTablebase.ts` first applies the existing
`canProbe()` material filter (all positions with at most three pieces, and
four-piece positions with at most two men). It then proves only inactivity or
threefold draws, no-legal-move losses, and directly verified mate in one. The
`maxMs` argument is compatibility-only. The legacy recursive routines and
shared memo remain isolated and are not exact evidence: their key has only the
current repetition count rather than the full history, depth is capped at 60,
and historical DFS back-edges/depth exhaustion were treated as draws.

`probeSmallEndgameDeterministic()` is safer regression infrastructure. It keys
the complete repetition-count map, has explicit node/depth limits, and returns
incomplete for a non-rule cycle, node limit, or depth horizon. A back-edge is
not a draw. It can still be incomplete and is not the canonical solver.

The audited rules are: captures are mandatory; every complete capture sequence
is legal (there is no global majority-capture filter); kings fly on unobstructed
diagonal rays; men move and capture forward; and `applyMove()` promotes a man
immediately at its destination on squares 0–3 for P1 or 28–31 for P2. No legal
canonical state therefore retains such an unpromoted man. With no legal move,
the side to move loses. The project draw policy is three occurrences of the
board hash, 32 plies without capture generally, or 16 plies without capture
when every remaining piece is a king. The halfmove clock resets only on a
capture (not promotion).

## Exact domain and representation

The **board-theoretic canonical tablebase** exhaustively enumerates all
non-overlapping, promotion-normalized two- and three-piece positions with at
least one piece per player, for both sides to move. Production `generateMoves`
and `applyMove` are the only movement authority. It adds the closed lower-count
terminal sinks produced by captures: one or two pieces belonging to the player
who just moved with the now-empty opponent to move. Two synthetic empty-board
states (one per side to move) define zero-piece encoding; they are documented
unreachable no-move LOSS sinks, not asserted to be reachable game positions.
Overlaps, invalid sides, and men stranded on promotion ranks are rejected.

No symmetry reduction is used. A reflection that preserves a man's promotion
direction is plausible, and 180-degree rotation combined with color/side swap
is plausible, but Phase 5A intentionally does not rely on either without a
separate exhaustive move-equivariance proof.

The graph is fully enumerated before classification. No-move LOSS nodes seed
retrograde propagation; a predecessor becomes WIN on a LOSS child and LOSS
only after every child is WIN. Remaining nodes after the complete fixed point
are DRAW. WIN minimizes DTM, LOSS maximizes DTM, and DRAW has no invented DTM.
All equally DTM-optimal moves are stored in deterministic order.

## Fresh-history current-rule tablebase

The original retrograde result is retained and named **board-theoretic only**:
it resets the clock in its board graph and is not current-rule truth. Phase 5A
now also builds a distinct current-rule tablebase for the specified root
semantics: the current board occurs once and `halfmoveClock = 0`.

The current-rule dynamic program carries every live clock value. Quiet moves
preserve material and increment the clock; captures strictly reduce material
and reset the clock. This gives an acyclic dependency order: increasing
material count and, within one count, decreasing clock. A state at clock 32 is
DRAW, or at clock 16 when all pieces are kings, before no-legal-move LOSS is
considered. WIN minimizes decisive DTM, LOSS maximizes it, and DRAW has no DTM.

Threefold repetition does not require multiplying this fresh-root table by
history. A board/side repeat can contain no capture, because captures strictly
reduce material and material can never return. It is therefore a quiet cycle
that returns to exactly the same legal choices with a strictly worse inactivity
clock. Repeating cannot unlock a continuation. Any decisive strategy containing
such a segment can delete it and choose the identical continuation earlier; if
a player can perpetuate it, the clock DP already classifies that strategy as
DRAW at 16/32. Therefore threefold changes only *when* that quiet-cycle draw is
adjudicated, not fresh-root W/D/L or DTM of a decisive result. The explicit
history fixtures independently exercise second/third occurrences and child
adjudication order.

## Live-game history boundary

`probeHistoryAwareExact()` is separate. It returns exact rule draws at the
current inactivity threshold or third repetition, exact terminal loss, and
exact directly verified mate in one. Otherwise it returns `UNKNOWN`. This is
deliberately conservative: canonical WIN/LOSS/DRAW is never copied into a live
game where the halfmove clock or repetition multiset could intervene. Setting
the root clock to zero does not make Makhos board-Markovian. UNKNOWN and search
back-edges are never converted to DRAW.

## Verification and feasibility

The dedicated tablebase gate checks Bellman and DTM invariants over every
stored state and validates every best move against production move generation.
The board-only Bellman test remains a structural check. A second verifier first
derives current-rule truth with a separate top-down memoized clock/material
solver without reading candidate outcomes, DTM, or moves. Only after derivation
it compares W/D/L, DTM, outcome-preserving moves, and DTM-optimal moves for all
6,976 requested two-piece states, a deterministic 100,000 three-piece sample,
and every in-scope existing tiny-endgame fixture. One tablebase test invocation
also performs two complete current-rule rebuilds and requires identical
fingerprints. History fixtures cover both clock thresholds and their preceding
ply, second/third repetitions, ordinary cycles, invalid boards, genuine mate in
one, and apparent mate overridden by inactivity or third repetition.

The audit artifact records measured build/storage and a deterministic 4,096-position Teacher sample. Its score-to-W/D/L statistic is explicitly heuristic; outcome-preserving and decisive-only DTM metrics are reported separately. Its fingerprint excludes timing and memory estimates. The simple Map
and object graph is suitable for verification, not mobile bundling. Phase 5B
should evaluate a compact indexed static file with lazy loading; build-time
generation is preferable to reconstructing this verification graph on-device.

