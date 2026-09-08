# Makhos / Thai Checkers Rules Specification

This is the canonical rule contract for the engine. Implementation is tested
against this document; implementation alone is not evidence that a rule is
correct. “Authoritative” below means supported by an external federation or
Thai competition source. “Project policy” means retained for compatibility
while external competition evidence remains incomplete.

## Evidence hierarchy

1. Published Thai competition rules and federation-family rule surveys.
2. Multiple independent Thai-rule descriptions that agree.
3. Explicit project policy, clearly marked and regression-tested.
4. Existing code and historical fixtures are implementation history only.

The Phase 1C follow-up used the FMJD *A Guide to Checkers Families and Rules*
Thai/Makhos entry and Thai Sports Association material identified in review.
The FMJD guide describes capture priority as free choice between capture
sequences, provided the chosen piece continues until it has no capture. It does
not assign Thai draughts the majority-capture rule. The Thai material likewise
describes compulsory capture and free choice among the hos/king's available
capture paths. Network access to retrieve archival copies was unavailable in
the audit environment; these references and that access limitation are
recorded rather than replacing them with inference from source code.

External rule-page references used by this audit:

- FMJD, *A Guide to Checkers Families and Rules*, “Thai draughts (Makhos)”
  entry, capture-priority table and Thai-rules section:
  `https://www.fmjd.org/downloads/variants/Checkers_families_and_rules.pdf`.
- Thai Sports Association, *กติกาการเล่นหมากฮอสไทย* (“Thai Checkers
  Rules”), rule-page section **การกิน** (“Capturing”), including the hos/king
  multiple-capture free-choice paragraph. This is the exact Thai rule-page
  reference identified in the Phase 1C review; the audit environment could not
  retrieve the external page to verify a stable public URL, so no guessed URL
  is recorded.

## Board, pieces, and movement

| Rule | Engine contract | Evidence status |
|---|---|---|
| Board/setup | 8×8 board, 32 playable dark squares, eight men per side | Established Makhos rule; executable initial-position/perft fixture |
| Man quiet movement | One diagonal square forward | Established Makhos rule |
| Man capture direction | Forward only | Externally supported Makhos-family rule; mirrored regression fixtures |
| Promotion | A man reaching the opponent's last rank is crowned after the complete move; it does not become a hos midway through a capture sequence | Externally supported Makhos-family rule; regression fixture |
| Hos quiet movement | Any unobstructed distance diagonally | Established Makhos rule; regression fixture |
| Hos capture landing | The first empty square immediately beyond the captured piece | Externally supported Makhos rule; regression fixture |

## Capture priority and continuation

Capture is mandatory whenever any piece of the player to move can capture.
Quiet moves are then illegal. The player may choose **any** capturing piece and
any complete capture path; there is no global requirement to choose the path
that captures the greatest number of pieces. Once a path is chosen, that same
piece must continue capturing until it has no further legal capture.

**Status: externally supported and resolved in Phase 1C.** This supersedes the
repository's former global maximum-capture filter and all historical notes that
called it a Thai rule. An executable fixture offers one complete one-piece path
and one complete two-piece path and requires both to be legal.

## Ending and draw state

| Rule | Engine contract | Evidence status |
|---|---|---|
| Piece elimination | A side with no pieces loses | Established |
| No legal move | A side with no legal move loses | Established |
| Threefold repetition | Third occurrence of the same board and side-to-move is an automatic draw | **Project policy / variant-dependent.** Retained and history-keyed; authoritative Thai competition support not established by this audit |
| General inactivity | 32 plies without a capture is an automatic draw | **Project policy / unresolved.** Retained for compatibility; do not describe as a universal Thai rule |
| All-hos inactivity | 16 plies without a capture when no men remain is an automatic draw | **Project policy / unresolved.** Retained for compatibility; do not describe as a universal Thai rule |
| Inactivity clock | Capture resets to zero; every quiet move, including promotion, increments by one | Project state semantics supporting the retained draw policy |

The 16/32 values were already embedded in engine/search state, but this audit
found no authoritative source sufficient to promote them to universal Thai
competition rules. They remain explicit, testable project policy until a
citable governing competition rule resolves the variant. Changing them would
alter game-theoretic results and requires a separate evidence-backed rules PR.

## Fixture and data compatibility

- Existing opening-book moves selected under majority filtering remain legal:
  removing capture priority only adds alternative complete captures and does
  not make a formerly complete capture illegal. Book coverage may be narrower,
  which is a later measurement issue, not a correctness defect.
- Tactical benchmark labels referring to “max capture” are historical dataset
  names, not rule assertions. Their expected answers were not changed or tuned.
- Puzzle expected moves remain subject to the existing provenance limitations.
  Any longer capture formerly expected remains legal, but newly legal shorter
  alternatives mean uniqueness must not be inferred from old move generation.
- Search orders captures by length as a heuristic. Ordering does not remove
  shorter legal captures and therefore is not capture priority.
