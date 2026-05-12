# Phase AL Tactical Extension Research

## Scope

Research only. No runtime behavior changes.

Goal:

- assess feasibility of a narrow tactical search extension experiment
- keep any future experiment `OFF` by default
- avoid global depth, budget, search, or eval rewrites

## Current Search Architecture Summary

Primary search file:

- `src/coreClaude/search/alphabeta.ts`

Core flow:

1. `iterativeDeepening(...)`
   - root entry point
   - iterative depth loop with aspiration windows
   - root move ordering and root candidate collection
2. `negamax(...)`
   - main recursive alpha-beta search
   - TT probe/store
   - pruning, LMR, tactical depth adjustment
3. `quiesce(...)`
   - terminal/horizon continuation
   - capture-only continuation when captures are forced

Move generation:

- `src/coreClaude/movegen.ts`
- `generateMoves(...)` calls `generateCapturesInto(...)` first and returns captures immediately if any exist
- Thai forced-capture and max-capture filtering are enforced in move generation

## Where Depth Is Reduced Or Extended

Main recursive depth reduction:

- `negamax(...)` starts each move at `d = depth - 1`
  - `src/coreClaude/search/alphabeta.ts:724`

Existing extensions already present:

- single legal move extension
  - `src/coreClaude/search/alphabeta.ts:725`
- endgame extension when total pieces `<= 5`
  - `src/coreClaude/search/alphabeta.ts:726`
- tactical extension via `extendTacticalDepth(...)`
  - function definition: `src/coreClaude/search/alphabeta.ts:172`
  - application in node search: `src/coreClaude/search/alphabeta.ts:731`
  - application at root: `src/coreClaude/search/alphabeta.ts:875`
- root-only low-mobility extension
  - `src/coreClaude/search/alphabeta.ts:844-847`
  - added on top of root depth only at low-mobility roots
- root-only forced-trap extension
  - `src/coreClaude/search/alphabeta.ts:877`
- root candidate re-check extension logic
  - `src/coreClaude/search/alphabeta.ts:951-968`

Existing reductions/pruning:

- LMR in node search
  - `src/coreClaude/search/alphabeta.ts:733-735`
- LMR at root
  - `src/coreClaude/search/alphabeta.ts:879-880`
- LMP at shallow quiet nodes
  - `src/coreClaude/search/alphabeta.ts:738-739`
- reverse futility, razoring, null move, probcut
  - `src/coreClaude/search/alphabeta.ts:630-690`

## Where Captures Are Detected

Move generation / forced capture:

- `generateMovesInto(...)` returns captures first
  - `src/coreClaude/movegen.ts:131-147`
- `generateCapturesInto(...)` filters to maximum capture length
  - `src/coreClaude/movegen.ts:150-180`

Search-side capture checks:

- quiet vs tactical node split:
  - `const isQuiet = moves[0].captured.length === 0`
  - `src/coreClaude/search/alphabeta.ts:625`
- opponent immediate capture availability:
  - `hasCapturesAvailable(child)`
  - `src/coreClaude/search/alphabeta.ts:730`
- root tactical checks also use `hasCapturesAvailable(child)`
  - `src/coreClaude/search/alphabeta.ts:874`

## Does Quiescence Exist?

Yes.

Quiescence entry:

- `src/coreClaude/search/alphabeta.ts:527`

Behavior:

- if `depth <= 0`, `negamax(...)` falls into `quiesce(...)`
  - `src/coreClaude/search/alphabeta.ts:608`
- if the position has no forced captures, quiescence stand-pats on eval
  - `src/coreClaude/search/alphabeta.ts:543-565`
- if captures are forced, quiescence searches capture continuations only
  - `src/coreClaude/search/alphabeta.ts:540-583`
- recaptures are explicitly prioritized by `lastCapSquare`
  - `src/coreClaude/search/alphabeta.ts:568-583`

Conclusion:

- the engine already has capture-only continuation after the nominal horizon
- this is not a no-quiescence searcher

## Existing Tactical Extension / Override Features

Already present:

- capture extension
  - yes, through `extendTacticalDepth(...)`
- recapture-sensitive extension
  - partially, through `opHasCaptures` and quiescence recapture ordering
- forced tactical line extension
  - yes, root forced-trap extension and low-mobility root extension
- quiescence after captures
  - yes
- extension budget / guard
  - partial only

Current guards:

- `MAX_PLY = 64`
  - `src/coreClaude/search/alphabeta.ts:65`
- `extendTacticalDepth(...)` stops extending when `ply > 18` or `depth < 2`
  - `src/coreClaude/search/alphabeta.ts:173`
- extension cap is bounded by `depth + 1` or `depth + 2`
  - `src/coreClaude/search/alphabeta.ts:184-185`

What is missing:

- no explicit per-line extension budget counter
- no dedicated node-count guard for extensions
- no OFF-by-default experiment flag for tactical extensions

## Current Search Experiment Flag Structure

Search-side env flag currently visible:

- `MAKHOS_ENABLE_LOW_MOBILITY_EXACT_TIEBREAK`
  - `src/coreClaude/search/alphabeta.ts:72-73`

Important note:

- current tactical depth adjustments are not behind a dedicated experiment flag
- `ENABLE_LOW_MOBILITY_RECAPTURE_OVERRIDE = true`
  - `src/coreClaude/search/alphabeta.ts:944-945`
  - this is a baked-in behavior, not an experiment toggle

## Smallest Safe Future Hook Point

Best future hook:

- inside `negamax(...)`, immediately after existing `extendTacticalDepth(...)`
  - `src/coreClaude/search/alphabeta.ts:724-735`

Why this hook is safest:

- all needed context already exists there:
  - current `depth`
  - current `ply`
  - `move.captured.length`
  - `opHasCaptures`
  - whether the node is quiet
- it affects recursive search uniformly without touching eval
- it can be made a no-op when flag is `OFF`
- it sits before LMR, so extension intent is clear and then any reduction logic can remain explicit

Secondary hook to keep aligned later:

- root loop after `extendTacticalDepth(...)`
  - `src/coreClaude/search/alphabeta.ts:870-880`

Reason:

- root and non-root behavior should stay conceptually aligned if a new extension is added

## Recommended First Experiment Design

Recommended experiment:

- narrow near-horizon capture extension
- max `+1` ply only
- explicit `OFF`-by-default env flag
- explicit extension budget of one extra tactical ply per line

Design sketch:

- new flag name:
  - `MAKHOS_ENABLE_PHASE_AL_CAPTURE_EXTENSION`
- new rule:
  - only eligible when `depth <= 2`
  - only if `move.captured.length > 0`
  - only if `ply <= 12`
  - only if extension budget for the line is still available
  - grant at most `+1` ply
- do not activate on quiet moves
- do not stack beyond the existing cap repeatedly in one line

Why this is safer than broader alternatives:

- narrower than changing global depth
- narrower than changing quiescence semantics
- narrower than recapture-only pattern detection at first pass
- narrower than touching root overrides or tie-break logic

Not recommended for first try:

- changing quiescence rules
- adding extension to quiet threat moves first
- adding `+2` tactical extension
- adding extension with no line budget
- touching `sac-two-win-three-*` handling through root override logic

## Likely Files / Functions To Touch Later

Primary:

- `src/coreClaude/search/alphabeta.ts`
  - `extendTacticalDepth(...)`
  - `negamax(...)`
  - root loop in `iterativeDeepening(...)`

Possibly useful but not required:

- `scripts/repeatedRunBenchmarkSummary.ts`
  - only if later docs/tooling should label the experiment mode more clearly
- `docs/ai/EXPERIMENT_LOG.md`
  - to record the experiment result later

No need initially:

- `src/coreClaude/eval.ts`
- `src/coreClaude/search/levelPolicy.ts`
- opening book files
- root tie-break logic

## Risk Assessment

Main risks:

- this engine already has several tactical extensions and root overrides
- stacking another extension can create hidden interaction rather than isolated gain
- capture-heavy tactical cases may overfit and destabilize `sac-two-win-three-p1`
- root-only special handling already exists, so new depth changes must avoid double-counting tactical help

Specific protected risks:

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`

Risk level:

- moderate if constrained to `depth <= 2`, capture-only, `+1` max, one-budget-per-line
- high if broadened to quiet moves, recaptures everywhere, or repeated stacking

## Benchmark Plan For Later Experiment

Before any code change:

1. `npm run test:perft`
2. `npm run gate:ai:report`
3. `npm run gate:ai:repeat`

For the future OFF-by-default experiment:

1. clean OFF baseline:
   - `npm run test:perft`
   - `npm run gate:ai:report`
   - `npm run gate:ai:repeat`
2. experiment ON with explicit env flag:
   - `MAKHOS_ENABLE_PHASE_AL_CAPTURE_EXTENSION=1 npm run gate:ai:repeat`
3. compare protected cases first:
   - `sac-two-win-three-p1`
   - `sac-two-win-three-p2`
   - `low-mobility-squeeze`
   - `low-mobility-squeeze-p2`
4. reject immediately if:
   - `sac-two-win-three-p1` catastrophically regresses
   - default-OFF path changes
   - perft fails

## Recommendation

Phase AL is feasible, but only as a very narrow experiment.

Best next action:

- keep this as research only for now
- if proceeding later, implement one OFF-by-default `+1` near-horizon capture extension with a one-per-line budget in `alphabeta.ts`
- benchmark with repeated runs before considering any promotion
