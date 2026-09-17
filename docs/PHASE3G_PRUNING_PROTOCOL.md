# Phase 3G — pruning threshold selection

Experiment `EXP-2026-013` measures whether one current pruning threshold family is too aggressive. Production remains `pruningProfile=current`; evaluation, LMR (`current`), ordering, extensions, TT, qsearch, time management, and opening logic are frozen.

## Frozen source audit and profiles

The canonical `fa6106a6850dc16fd4ea0a071b7f956317bd0952` source matched the requested audit: RFP is quiet, depth <=4, `se - 120*depth >= beta`; razoring is quiet, depth <=2 with 350/550 margins and qsearch verification; guarded null move begins at depth 3 with R=2 through depth 5 and R=3 thereafter; capture-only ProbCut begins at depth 5, uses `beta+200`, `depth-4`, and at most three longest captures; quiet LMP at depth <=2 starts at move indexes 6/10. Existing non-root, repetition, draw, tactical, mate-window, and alpha guards remain in force.

The immutable candidate order is: `rfp-conservative` (120→160 only), `razoring-conservative` (350/550→450/700 only), `null-delayed` (minimum depth 3→4 only), `probcut-conservative` (200→300 only), and `lmp-conservative` (6/10→8/12 only).

## Frozen corpus and protocol

Suite `makhos-phase3g-pruning-starts-v1` uses the audited xorshift32/legal-sorted/diverse-plies-2-17/v1 generator, seed `0xfa6106a6` (4200662694), 64 complete states, and SHA-256 `8bc0ad868e0ffa58d8449dad8c5428f7dc7ddd81af2ecc497faeb31754844967`. Tests prove complete-state uniqueness and disjointness from every retained Phase 3A–3F suite. Starts 1–32 screen all five profiles at 5,000 nodes/move; starts 33–64 remain untouched until mechanical selection.

Baseline explicitly pins every search flag true, Teacher v1 extensions (`singleLegalMove`, `tacticalCapture`, `multiCapture`, `opponentForcedCapture`, `singleCaptureRecapture`, `rootLowMobility`, `soundForcedTrap` true; `smallEndgame` false), every ordering flag true, LMR `current`, and pruning `current`. Fixed-node games opt both engines into pruning counters; equal-time games do not construct counters.

Selection is the highest point score strictly above 50%, ties in frozen order. Only it is confirmed at 20,000 nodes, 50,000 nodes, and 100 ms on starts 33–64. Holdout identity stays blinded until results and decision are frozen, then serves only as a regression gate. Promotion requires the 20k and 100 ms Elo CIs wholly above zero and no tactical/holdout regression; mixed or zero-crossing evidence is **NEEDS MORE DATA**, repeated negative evidence rejects. No outcome changes production defaults.
