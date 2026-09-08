# Phase 2B position-suite protocol

`makhos-position-suite-v1` is a frozen measurement corpus using
`makhos-position-case-schema-v1`. It is measurement infrastructure, not an
engine-strength change. The canonical command is `npm run baseline:position-suite`;
fixed nodes (5,000 maximum combined main/qsearch nodes per case) are canonical,
while `--depth=N` is diagnostic. JSON and CSV artifacts are written below
`.tmp/position-suite/` and are not version controlled. The frozen v1 manifest
has SHA-256 fingerprint
`f2d5d8d41275836904750825b60230f283724ac3d97e18305157169d03d5aa30`.

## Schema and validation

Every case has a stable ID, complete `Position` (including side and inactivity
clock), root-inclusive repetition history, optional replayable opening moves,
motifs, optional evidence-backed difficulty, split, provenance, validation tier,
and a typed expectation. Expectations support a proven unique/legal-only move,
multiple acceptable moves, W/D/L, optional PV, or no label.

Validation tiers are deliberately not interchangeable:

* **verified-exact** — a rule-derived terminal/forced continuation or a bounded,
  deterministic exhaustive result without an unresolved horizon.
* **verified-provenance** — an externally sourced position and solution with a
  stable, reviewable citation. V1 currently has none.
* **provisional-engine** — useful engine/self-play data with no objective label.
* **legacy-unverified** — historical assertions lacking enough evidence.

“A deep engine chose it” is never exact evidence. External cases must preserve
the source, publication/record locator, position transcription, solution and
applicable ruleset. Difficulty is omitted unless its source or an explicit
measurement supports it.

V1 intentionally contains only five verified cases: two development, two
holdout, and one history-sensitive diagnostic draw. The four scored move cases
are elementary rule proofs: every accepted move captures the final opposing
piece; three have exactly one legal move, and one has two equivalent immediate
wins. The fifth is an explicit threefold root under documented project policy.
This small count is honest; it must not be inflated with guessed labels.

## Development, holdout, and leakage

Development cases may be inspected during Phase 3 debugging/tuning. Holdout v1
is immutable for an experiment series and must not select parameters. Default
console output reports only aggregate holdout accuracy, never expected holdout
moves. Changing membership, labels, generator, or evidence creates a new suite
version. The automated leakage gate compares board, side, halfmove clock, and
the complete ordered history vector, and rejects exact dev/holdout duplicates.
Stable IDs and version constants are regression tested.

The fingerprint covers, in suite order, stable ID, split, every position field,
ordered history, validation tier, and the complete expected label/accepted
moves. Tests recompute it from live content. Any membership, split, state,
history, tier, or label edit therefore fails until reviewers explicitly create
a new suite version and freeze its new fingerprint; changing only a version
string cannot disguise content drift.

Normal runner output is tuning-blind: holdout case rows are omitted from both
JSON and CSV, while aggregate verified holdout accuracy remains available.
Development and diagnostic rows remain inspectable. `--reveal-holdout` is an
explicit evaluation-only escape hatch; the frozen initial baseline command uses
it, but Phase 3 parameter-selection runs must not.

The diagnostic split contains the 14 historical handcrafted puzzles and four
unlabeled positions replayed from the deterministic Phase 2A legal traces.
These strategic positions measure move stability, disagreement and depth/node
sensitivity; their selected moves are not truth labels.

## Legacy 14-puzzle audit

All 14 positions are structurally legal under the current representation. Only
four recorded expected from/to moves are currently legal:
`trap-01-bait-sacrifice`, `trap-02-double-bait`, `promo-01-race-to-king`, and
`endgame-04-triangulation`. The other ten expected moves are illegal at their
roots under current move generation. None has authoritative provenance, a
proof of bestness, or a proof of uniqueness, so all 14 remain
`legacy-unverified` diagnostics even when the recorded move is legal.

No root presents legal captures of differing lengths, so the removed global
maximum-capture filter does not alter the audited root legal sets. This does
not prove that the old assumption had no effect deeper in a claimed solution;
the prose lines are insufficient to reconstruct and audit those continuations.
No expected move was silently changed. `PUZZLE_BASELINE.md` remains historical
archive and is not a current accuracy baseline.

## Motifs and scoring

The taxonomy supports: forced capture, multi-capture, sacrifice, promotion
race, king technique, tempo, defense, trap, draw/repetition, endgame conversion,
and low mobility. Tags are omitted when unclear. A taxonomy entry need not be
represented by verified v1 evidence.

Pass/fail exists only for verified cases. A move expectation passes if the
chosen move belongs to its full accepted set. W/D/L is adjudicated only by an
explicit repetition/inactivity rule state, terminal elimination/no-move state,
or an actually resolved deterministic exact oracle. An ordinary finite search
score—positive, zero, or negative—is never W/D/L proof; without suitable
evidence the case remains unscored. Reports calculate development verified accuracy and holdout
verified accuracy separately. Motif denominators contain verified cases only;
legacy and provisional rows remain visible as diagnostics with no pass/fail.
No aggregate denominator crosses validation tiers.

JSON stores the complete chosen `Move`; CSV uses a canonical serialization of
`from`, `to`, ordered `captured`, ordered `path`, and `promote`. Capture lines
sharing endpoints therefore remain distinct measurement results.
