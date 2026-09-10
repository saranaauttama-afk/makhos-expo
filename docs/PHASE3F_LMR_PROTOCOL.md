# Phase 3F LMR schedule tuning protocol

Phase 3F starts from canonical `engine-mainline` tip `8208f6f68bc16d0df00e68691e8e1b319db6cf00`; Frozen Teacher v1 remains `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`. This file, the config, corpus identity, candidate inventory, and selection rule were frozen before tournament outcomes were read.

## Source audit and invariant surface

Production uses the table `min(3, floor(0.5 + log(moveIndex+1) * log(depth+1) / 2.2))`. It reduces only with LMR enabled, zero-based move index at least 3, remaining depth at least 2, a quiet move, more than one legal move, and no forced opponent capture after the move. A reduced null-window search which raises alpha without reaching beta is re-searched at full depth. Both root and interior search use these rules. These guards and re-search policy are frozen.

Only the following predeclared profiles may be examined, in tie-break order: `off` (zero reduction), `gentler` (`max(0,current-1)`), `delayed` (current table but index threshold 4), and `aggressive` (`min(3,current+1)`). Baseline is explicit `current`. The production default remains `current` regardless of outcomes.

## Corpus and partition

`makhos-phase3f-lmr-starts-v1` uses seed `0x8208f6f6` (`2181625590`) with the audited `xorshift32/legal-sorted/diverse-plies-2-17/v1` generator. Its canonical replay fingerprint is `b22433396fe82ec54461f449944cb780fedca7caa4f5fab4a440a0e980972f89`. Complete state comprises side, all four piece bitboards, and halfmove clock. Tests prove 64 unique states and disjointness from every retained Phase 3A-3E corpus. Starts 1-32 are screening; starts 33-64 are untouched confirmation.

## Tournament and decision

Screen all four profiles against `current`, paired colors, 32 paired starts (64 games/profile), 5,000 nodes/move, depth cap 64, and 160 plies. Select exactly the highest candidate score strictly over 50%; exact score ties follow the inventory order above. If none qualifies, stop. Screening only selects; it cannot promote.

Confirm only the selected profile on starts 33-64 at 20,000 and 50,000 nodes/move and 100 ms/move, each with 32 paired starts, paired colors, depth cap 64, and 160 plies. Fixed-node evidence records W/D/L, score, Elo, pair-bootstrap 95% intervals, depth, main nodes/qnodes, elapsed/NPS, LMR counters, unresolved, and errors. The equal-time run collects no LMR counters.

Baseline and candidate explicitly pin all search, extension, and move-ordering flags from `config/phase3f-lmr-v1.json`; `smallEndgame=false` and `soundForcedTrap=true`. They differ only by LMR profile. Evaluation, LMP, null move, ProbCut, IID, razoring/reverse futility, extensions, move ordering, TT, and qsearch are not tuning surfaces.

LMR counters are opt-in and otherwise neither allocated nor incremented: eligible/reduced moves, reduced plies, histogram 0/1/2/3, reduced-search full-depth re-searches, and root/interior partitions. “Eligible” applies the selected profile's move-index threshold plus every frozen tactical guard; `off` therefore records eligible moves at the current threshold with reduction zero.
