# Engine Phase A Plan

This document converts the Phase A inspection into an implementation plan without changing current Phase 4 baseline behavior.

## Ground Rules

- Keep Phase 4 baseline behavior unchanged until implementation is explicitly approved.
- Do not propose broad root overrides.
- Do not fabricate candidate scores.
- Prefer flag-gating over deletion where possible.
- Do not touch TT, opening book, repetition, incremental hash, or move ordering yet.
- Do not edit source as part of this document task.

## SAFE / Keep Now

### Root candidate verification pass before overrides

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: re-searches a narrow window of near-top root candidates and updates the best move only through another real search result
- Affects final move selection: yes, but only by deeper verification rather than a heuristic replacement
- Risk level: `low`
- Recommended next action: keep as-is for now

### Fallback to first legal move if no root result exists

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: returns a legal move if search exits without a selected best move
- Affects final move selection: yes, but only as a fail-safe when search did not finish cleanly
- Risk level: `low`
- Recommended next action: keep as-is for now

### Opening-book root hints passed into ordering

- File path: `src/ui/useCodexEngine.ts`, `src/coreClaude/search/alphabeta.ts`
- What it does: passes book-derived root hints into move ordering only
- Affects final move selection: no direct override, ordering only
- Risk level: `low`
- Recommended next action: keep as-is and do not touch in Phase B

## MUST DISABLE Before Phase B

### Low-mobility recapture override

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: post-search root override that prefers a low-mobility forced-recapture idea over the searched best move
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: disable before Phase B experiments, ideally by introducing a narrow flag first and then switching that flag off only after approval

## NEEDS FLAG Before Experiments

### Opening diversification

- File path: `src/coreClaude/search/alphabeta.ts`, `src/ui/useCodexEngine.ts`
- What it does: randomly chooses among near-best quiet root moves in opening play
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: add an explicit flag so experiments can run with deterministic root choice when needed; keep current default unchanged until approved

### Endgame promotion override

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: post-search root override that prefers a promotion candidate in very small endgames
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: put behind a dedicated flag before experiments; do not remove outright

### Easy/normal post-search move selection

- File path: `src/coreClaude/search/levelPolicy.ts`, `src/ui/useCodexEngine.ts`
- What it does: replaces the searched best move with a weaker deterministic alternative for easier difficulty levels
- Affects final move selection: yes
- Risk level: `medium`
- Recommended next action: flag-gate separately from core search experiments so engine experiments can test raw root output without deleting difficulty shaping

## NEEDS DEEPER REVIEW

### Forced recapture trap override

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: post-search root override that chooses a quiet move if it predicts a sound forced recapture trap
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: review first, because the helper still contains a fabricated fallback-score branch if `rootCandidates` are absent; if retained later, it should only operate on real root candidates and likely be flag-gated

### Root tactical safety override

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: replaces the current best move with a lower-risk root candidate when immediate capture risk looks meaningfully better
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: inspect thresholds and behavior more deeply before any disable or flag decision

### Absolute anti-hang safety

- File path: `src/coreClaude/search/alphabeta.ts`
- What it does: final post-search safety replacement that can override the selected move with the safest legal move
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: inspect deeply, because it sits after the main root-selection chain and can still replace the final move

### Opening-book weighted random pick

- File path: `src/coreClaude/search/openingBook.ts`
- What it does: randomly selects among weighted book moves
- Affects final move selection: yes, if this path is used
- Risk level: `medium`
- Recommended next action: deeper review only for now; do not touch in Phase B because opening book is explicitly out of scope

### UI-level safer fallback move after engine result

- File path: `src/ui/HumanVsCodexArenaScreen.tsx`
- What it does: replaces the engine-returned move with a safer UI-level fallback based on immediate capture risk
- Affects final move selection: yes
- Risk level: `high`
- Recommended next action: deeper review before experiments, because it can hide or distort core-engine behavior during manual testing

## Phase B Implementation Proposal

### Exact Files Likely To Change

- `src/coreClaude/search/alphabeta.ts`
- `src/coreClaude/search/levelPolicy.ts`
- `src/ui/useCodexEngine.ts`
- `src/ui/HumanVsCodexArenaScreen.tsx`
- `src/workers/engineWorker.ts` only if any future flag plumbing must cross the worker boundary

### Safest First Change

- First add a small, explicit root-behavior flag surface in `src/coreClaude/search/alphabeta.ts` for post-search overrides only
- Keep all flags defaulted to current baseline behavior so Phase 4 stays unchanged
- Thread those flags through `src/ui/useCodexEngine.ts` only if needed for controlled experiments
- The first behavior to isolate after approval should be `pickLowMobilityRecaptureCandidate`, because it is both high-risk and already classified as the first must-disable candidate

### Validation Commands After The Change

```bash
npm run test:perft
npm run test:tactical
npm run bench:ai:fresh
```

### Validation Expectation

- Perft must remain `8/8 passed`
- No experiment should proceed if root behavior changes are not benchmarked immediately after the change
- Do not run long benchmarks as the first validation pass

## Summary

- The first Phase B target should be narrow and reversible
- The first non-default experiment should isolate one override at a time
- Low-mobility recapture is the clearest first candidate to disable later
- Trap and safety overrides need deeper review before implementation, not broad removal
