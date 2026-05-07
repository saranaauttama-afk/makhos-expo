# Engine Phase A Inspection

This document records a read-only inspection of risky root-level heuristic and override logic on branch `tuneCodex`.

## Scope

- Phase A inspection only
- No source files modified
- No engine behavior changed
- No optimization performed

## Latest Measured Baseline

- Perft: `8/8 passed`
- Quick tactical:
  - `easy`: `92/8`
  - `normal`: `97/3`
  - `hard`: `95/3`
  - `expert`: `97/3`

## Files Inspected

- `src/coreClaude/search/alphabeta.ts`
- `src/coreClaude/search/levelPolicy.ts`
- `src/ui/useCodexEngine.ts`
- `src/coreClaude/search/openingBook.ts`
- `src/ui/HumanVsCodexArenaScreen.tsx`
- `src/coreClaude/search/repetition.ts`
- `src/workers/engineWorker.ts`
- `scripts/aiBenchmark.ts`

## Risky Logic Found

| Item | File | Can override search result? | Fabricates scores? | Introduces randomness? | Ordering only or changes final move? | Recommendation |
|---|---|---|---|---|---|---|
| Opening diversification via `pickDiversifiedRoot` | `src/coreClaude/search/alphabeta.ts:73`, `src/coreClaude/search/alphabeta.ts:880`, `src/ui/useCodexEngine.ts:132` | Yes | No | Yes, uses `Math.random()` | Changes final selected move | `flag-gate later` |
| Forced recapture trap override via `pickSoundForcedTrap` | `src/coreClaude/search/alphabeta.ts:255`, `src/coreClaude/search/alphabeta.ts:889` | Yes | Yes, helper fabricates `bestScore - 180` if candidates are absent | No | Changes final selected move | `inspect deeper` |
| Low-mobility recapture override via `pickLowMobilityRecaptureCandidate` | `src/coreClaude/search/alphabeta.ts:299`, `src/coreClaude/search/alphabeta.ts:896` | Yes | Yes, helper fabricates `bestScore - 180` if candidates are absent | No | Changes final selected move | `disable later` |
| Endgame promotion override via `pickEndgamePromotionCandidate` | `src/coreClaude/search/alphabeta.ts:317`, `src/coreClaude/search/alphabeta.ts:903` | Yes | Yes, helper fabricates `bestScore - 120` if candidates are absent | No | Changes final selected move | `flag-gate later` |
| Root tactical safety override via `pickSaferRootCandidate` | `src/coreClaude/search/alphabeta.ts:334`, `src/coreClaude/search/alphabeta.ts:913` | Yes | No | No | Changes final selected move | `inspect deeper` |
| Absolute anti-hang safety via `pickAbsoluteAntiHangMove` | `src/coreClaude/search/alphabeta.ts:367`, `src/coreClaude/search/alphabeta.ts:931` | Yes | No | No | Changes final selected move | `inspect deeper` |
| Root candidate verification pass before overrides | `src/coreClaude/search/alphabeta.ts:116`, `src/coreClaude/search/alphabeta.ts:843` | Yes, by re-searching near-top candidates and updating `best` | No | No | Changes final selected move, but still through search | `keep` |
| Fallback to first legal move if no root result exists | `src/coreClaude/search/alphabeta.ts:923` | Yes, but only as fail-safe | No | No | Changes final selected move only on incomplete search result | `keep` |
| Easy/normal post-search move selection via `selectStrictLevelMove` | `src/coreClaude/search/levelPolicy.ts:108`, `src/ui/useCodexEngine.ts:144` | Yes | No | No true randomness, but deterministic non-best selection | Changes final selected move | `flag-gate later` |
| Opening-book weighted random pick | `src/coreClaude/search/openingBook.ts:969`, `src/coreClaude/search/openingBook.ts:998` | Yes, if this path is used | No | Yes, uses `Math.random()` | Changes final selected move | `inspect deeper` |
| Opening-book root hints passed into `orderMoves` | `src/ui/useCodexEngine.ts:117`, `src/coreClaude/search/alphabeta.ts:67`, `src/coreClaude/search/alphabeta.ts:393` | No direct override | No | No | Ordering only | `keep` |
| UI-level safer fallback move after engine result | `src/ui/HumanVsCodexArenaScreen.tsx:378`, `src/ui/HumanVsCodexArenaScreen.tsx:752` | Yes, outside core engine | No | No | Changes final selected move | `inspect deeper` |

## Notes Per Item

- `pickDiversifiedRoot` is a true root override. It samples among near-best quiet moves and can replace the highest-scoring root move after search is complete.
- `pickSoundForcedTrap`, `pickLowMobilityRecaptureCandidate`, and `pickEndgamePromotionCandidate` all contain helper-level fallback score fabrication when `rootCandidates` are missing. In the current main path they are called with `lastRootCandidates`, but the fabricated-score branch still exists in code and is therefore a risk surface.
- `pickSaferRootCandidate` and `pickAbsoluteAntiHangMove` do not fabricate scores, but both can replace the searched best move with a risk-based alternative.
- `selectStrictLevelMove` is outside the core search loop, but it is still a post-search final-move override for `easy` and `normal`.
- `lookupOpeningBook` is random, but the current strict engine path in `useCodexEngine.ts` uses `lookupOpeningBookCandidates` plus root hints instead of directly calling the weighted random picker.
- `HumanVsCodexArenaScreen.tsx` contains a second safety override path in UI code, which means the final played move can differ from the engine result even when the engine itself is unchanged.

## Repetition, Loop, And Contempt Findings

- `src/coreClaude/search/repetition.ts` is deterministic bookkeeping only.
- The inspected root-selection path does use repetition state during search, but no separate root-level repetition override was found in this pass.
- No explicit `contempt` logic was found in the inspected files for this branch.
- No explicit loop-breaking root override was found beyond standard threefold repetition handling.

## Phase A Conclusion

- The highest-risk file is `src/coreClaude/search/alphabeta.ts`.
- The most important risk pattern is post-search root replacement after root candidates have already been scored.
- The main categories of risk are:
  - direct root overrides
  - helper-level fabricated fallback scores
  - opening or level-based move diversification
  - UI-layer safety replacement outside the core engine

## Source Modification Status

- Source files modified: `none`
- Documentation files modified in this task:
  - `ENGINE_PHASE_A_INSPECTION.md`
