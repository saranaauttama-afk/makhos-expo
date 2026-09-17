# Phase 3H — Aspiration-window tuning protocol

## Frozen identities and source audit (before outcomes)

Canonical starting tip: `297105dd7ce17d4815e1805d3edbcadeb96a0e57`. Frozen Teacher v1 identity: `b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`.

The root loop completes depth 1 with `[-INF,+INF]`. Later iterations center on the previous **completed** score with half-width 150. A result at or below the lower bound fails low; a result at or above the upper bound fails high. Either failure doubles the half-width, expands only the failed side, and retries the same nominal depth. Each attempt retains existing root PVS: the first ordered move uses the root alpha-beta window; later moves use a null window and are re-searched at full depth with the root window after an in-window improvement. No PVS, extension, pruning, evaluation, TT, qsearch, or root tactical behavior is tuned.

## Frozen profiles

Inventory order is `off`, `narrow-75`, `narrow-100`, `wide-300`. `current` (baseline) uses ±150 and the audited doubling behavior. `off` uses a full window at every depth and never retries. The narrow profiles use initial half-width 75 or 100, and `wide-300` uses 300; all retain the current retry logic. Production defaults remain `current` regardless of results.

The baseline explicitly pins every production search flag true; Teacher extensions to `singleLegalMove=true`, `smallEndgame=false`, `tacticalCapture=true`, `multiCapture=true`, `opponentForcedCapture=true`, `singleCaptureRecapture=true`, `rootLowMobility=true`, `soundForcedTrap=true`; every ordering flag true; and `lmrProfile=current`, `pruningProfile=current`. Tournament root diversification is false. Evaluation, TT, qsearch, and tactical overrides are unchanged.

Aspiration instrumentation is independently opt-in and default-off. It records completed iterations, aspiration iterations, fail-low/high, retries and per-depth histogram/max, failed-attempt main/qnodes, and initial/final half-width histograms. It allocates no statistics object and performs no counter updates in ordinary production searches. Fixed-node measurements enable it for both sides; 100 ms measurements disable it.

## Frozen corpus and partition

`makhos-phase3h-aspiration-starts-v1` uses the audited `xorshift32/legal-sorted/diverse-plies-2-17/v1` legal-playout family, seed `0x297105dd` (`695272925`), 64 complete-state-unique legal starts, and SHA-256 fingerprint `18c227cee03578f11f4b9ecfa86899bbbc00344abe8c84aff2832551bead0098`. Complete state includes side, four bitboards, and halfmove clock. It is disjoint from all retained Phase 3A–3G corpora. Starts 1–32 are screening; starts 33–64 remain untouched until selection is frozen.

## Screening and mechanical selection

Each candidate independently faces current on starts 1–32: 32 color-swapped pairs / 64 games, 5,000 nodes/move, maxDepth 64, maxPlies 160. Record W/D/L, score, Elo, deterministic pair-bootstrap 95% CI, depths, nodes/qnodes, elapsed/NPS, aspiration counters/cost, unresolved/errors. Select exactly the highest point score strictly above 50%; ties follow inventory order. If none exceeds 50%, stop inconclusive.

## Confirmation and decision

Run only the frozen selection on untouched starts 33–64 at 20,000 nodes, 50,000 nodes, and 100 ms per move, each 32 paired starts / 64 games, maxDepth 64, maxPlies 160. Explain retry, wasted-node, and completed-depth changes for fixed-node runs. After freezing the decision, reveal the verified position-suite holdout and compare baseline to selected at 5,000 nodes; aspiration profile is the only difference.

Promotion interpretation: 20k **and** 100 ms Elo confidence intervals wholly above zero, with no tactical/holdout regression, is a strong promotion candidate. Mixed or zero-crossing evidence is **NEEDS MORE DATA**; repeated negative evidence is reject. Even strong evidence does not change the production default in this PR.
