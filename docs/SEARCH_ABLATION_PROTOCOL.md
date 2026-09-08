# Phase 3A search-feature ablation protocol

## Purpose and frozen controls

Phase 3A measures the contribution of existing search features; it does not
change production defaults, thresholds, evaluation weights, or Thai rules.
Baseline A uses every `DEFAULT_SEARCH_FEATURES` value. Candidate B differs by
exactly one boolean set to `false`. Equal fixed-node work is canonical strength
evidence; fixed time is prohibited for strength decisions. Configuration is
frozen in `config/search-ablation-v1.json`, and output conforms to
`schemas/search-ablation-summary-v1.schema.json`.

The `makhos-search-ablation-starts-v1` corpus contains 64 deterministic legal
positions generated with xorshift32 seed `0x3341424c`. Each stores its complete
legal 2–17-ply replay sequence, is deduplicated by complete position state, and
is played once with each engine as each color. It is generated experiment data,
not a Thai opening book and makes no opening-theory claim.

Screen all eight flags on at least 32 paired starts at 5,000 nodes per move.
Confirm only one or two hypotheses on all 64 paired starts at equal 5,000 or
10,000 nodes. The runner records actual completed pairs, unresolved games, and
errors; incomplete pairs are excluded from score statistics rather than treated
as draws. Raw JSON/CSV goes under ignored `.tmp/search-ablation/`.

## Feature audit (defaults are all enabled)

| Flag | Actual trigger and window/depth restriction | Capture, draw, and safety interaction | Intended purpose |
|---|---|---|---|
| `reverseFutility` | non-root, unrepeated, quiet node; depth ≤4; static evaluation minus `120 × depth` reaches beta | Never at a forced capture; inactivity/repetition draws are returned before pruning | Fail-high pruning for clearly favorable quiet nodes |
| `razoring` | non-root, unrepeated, quiet node; depth ≤2; static evaluation plus 350/550 is below alpha, then a null-window qsearch also fails low | Never at a forced capture; qsearch preserves mandatory captures and repetition state | Avoid full shallow search of likely fail-low nodes |
| `nullMove` | non-root, unrepeated, quiet node; depth ≥3, non-mate beta, static evaluation ≥ beta, recursive null allowed | Never passes a mandatory capture; disabled when `halfmoveClock + 2` approaches the applicable 16/32-ply inactivity limit; synthetic pass is not inserted into repetition history | Fail-high pruning when the side can effectively lose a tempo |
| `probCut` | non-root, unrepeated forced-capture node; depth ≥5; at most the three longest captures searched at depth−4 against beta+200 | Deliberately capture-only; real child hashes are pushed/popped for repetition and root draw checks precede it | Shallow evidence for a deep beta cutoff |
| `iid` | PV/full-window node, no TT move, depth ≥5, ply >0, recursion guard set | Applies to quiet or capture positions; child search uses normal draw/capture rules; it changes ordering only | Obtain a TT move when deep PV ordering has none |
| `lmr` | fourth or later quiet move with remaining depth ≥2, at root and interior nodes | Disabled for forced-capture moves, single legal moves, and quiet sacrifices that leave an opponent forced capture; reduced moves are re-searched at full depth when they improve alpha | Spend less depth on late, apparently weak quiet moves |
| `lmp` | non-root quiet node at depth 1/2 after move index 6/10 once alpha is non-mated | Disabled for forced captures and single legal moves; ordinary draw checks occur on entered nodes | Stop searching very late quiet moves at shallow depth |
| `extensions` | groups single-legal-move, ≤5-piece, tactical/capture, low-root-mobility, sound forced-trap, root verification, and multi-capture extensions | Forced lines and captures are a primary reason for extension; normal node draw guards and `MAX_PLY` bound growth | Search forced/tactical/endgame lines beyond nominal depth |

No dead flag or broken override plumbing was found: the tournament harness merges
each engine override with defaults immediately before every search, and existing
TT tests exercise each one-at-a-time disable. Grouped `extensions` attribution
is intentionally coarse and may justify a later sub-feature experiment.

## Statistics and interpretation

Candidate score, W/D/L, pair-aware deterministic bootstrap 95% confidence
interval (20,000 resamples), and Elo transformed from score are primary.
Report average completed depth, main nodes, qnodes, NPS, unresolved, and errors
as diagnostics. The verdict describes the *enabled feature*: **likely
beneficial** only when the disabled candidate's score CI is wholly below 50%;
**likely harmful** only when wholly above 50%; otherwise **inconclusive**. Do
not call an interval crossing 0 Elo significant, and never remove a feature
from a small screening result.

Fixed-depth runs may be made for the one or two most interesting candidates to
explain main/q-node, depth, and branching effects; they never replace Elo
evidence. Selection may inspect development and diagnostic position outputs,
but must not use `--reveal-holdout` or holdout case details. A Phase 3B proposal
must also pass rules, perft, tactical, search determinism, TT correctness,
tablebase determinism, Phase 1C, tournament, position-suite, ablation, and type
checks. Any tactical regression is recorded even if tournament score improves.

## Commands

```sh
npm run experiment:search-ablation
npm run experiment:search-ablation -- --features=nullMove,lmr --pairs=64 --nodes=10000
npm run experiment:search-ablation -- --features=extensions --pairs=32 --depth=5 --output=.tmp/search-ablation-depth
npm run test:search-ablation
```
