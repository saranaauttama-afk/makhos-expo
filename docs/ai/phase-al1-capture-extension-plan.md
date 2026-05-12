# Phase AL.1 Capture Extension Plan

## Scope

Planning only. No runtime behavior changes.

Goal:

- design the smallest OFF-by-default `+1` capture-extension experiment
- determine whether it is actually distinct from current search behavior
- protect:
  - `sac-two-win-three-p1`
  - `sac-two-win-three-p2`

## Inputs Reviewed

- `docs/ai/phase-al-tactical-extension-research.md`
- `src/coreClaude/search/alphabeta.ts`
  - `extendTacticalDepth(...)`
  - `negamax(...)`
  - root loop inside `iterativeDeepening(...)`

## Current Tactical Extension Behavior

Current non-root flow in `negamax(...)`:

1. start with `d = depth - 1`
2. extend for:
   - single legal move
   - tiny endgames (`total <= 5`)
   - tactical conditions via `extendTacticalDepth(...)`
3. then possibly reduce with LMR on late quiet moves

Current `extendTacticalDepth(...)` behavior:

- returns immediately when:
  - `ply > 18`
  - `depth < 2`
- otherwise extends when:
  - current move is a capture
  - or child gives the opponent captures
- adds more extension for:
  - multi-capture
  - shallow quiet/single-capture recapture-risk shapes
- caps result to:
  - `depth + 1` for normal capture cases
  - `depth + 2` for multi-capture cases

## Is A Generic `+1` Capture Extension Distinct?

Short answer:

- mostly no

Why:

- at `depth >= 2`, current logic already grants capture-sensitive extra depth
- for a single capture at `depth = 2`:
  - base child depth would be `1`
  - current helper already lifts it back to `2`
  - that is already the same practical `+1 ply` idea
- for larger tactical depths, current helper can already add even more in some cases

Conclusion:

- a generic “add `+1` on capture” experiment would mostly duplicate existing behavior
- it would not be a clean isolated new signal

## Where A Distinct Experiment Still Exists

The distinct slice is:

- near horizon only
- specifically when `depth == 1`

Why this slice is distinct:

- current `extendTacticalDepth(...)` bails out when `depth < 2`
- so at `depth == 1`, no tactical extension is currently applied there
- child search drops directly to `depth = 0`, then quiescence

What a distinct AL.1 experiment would do:

- preserve one extra full negamax ply at `depth == 1`
- only for narrow tactical cases
- before quiescence takes over

That is meaningfully different from current behavior because:

- it inserts one extra full search ply where none currently exists
- quiescence alone is capture-only and narrower than full negamax

## Recommended AL.1 Experiment Shape

Recommended experiment target:

- near-horizon capture extension only

Recommended exact flag:

- `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`

Recommended behavior when flag is ON:

- apply only in `negamax(...)`
- apply only when `depth == 1`
- apply only when extension budget for the line is still available
- apply only for one of:
  - `move.captured.length > 0`
  - or `opHasCaptures === true`
- grant at most `+1` ply
- do not stack more than once per line

Recommended behavior when flag is OFF:

- exact no-op
- no runtime change

## Exact Gating Conditions

Recommended gating:

1. experiment flag:
   - `process.env.MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION === '1'`
2. near-horizon only:
   - `depth === 1`
3. tactical trigger only:
   - `move.captured.length > 0`
   - or `opHasCaptures === true`
4. max `+1` ply:
   - child depth may increase from `0` to `1` only
5. extension budget guard:
   - one extra AL.1 extension per line maximum
6. ply guard:
   - retain conservative guard such as `ply <= 12`
7. node explosion guard:
   - if no dedicated node guard is added, do not proceed past planning without at least the one-per-line budget

## Extension Budget Guard Recommendation

Current engine does not have a dedicated extension budget counter.

Recommended future addition:

- pass a small integer budget through `negamax(...)`
- initial value:
  - `1` when experiment flag is ON
  - `0` when OFF
- consume it the first time the AL.1 extension fires
- do not refresh the budget deeper in the same line

Why this matters:

- prevents repeated near-horizon re-extension
- avoids accidental overlap with existing tactical extension logic
- gives a hard cap even if tactical shapes recur

## Node Explosion Guard

Current search has:

- time deadline
- `TC_MASK` periodic time check
- `MAX_PLY`

Current search does not have:

- an extension-specific node budget

Planning recommendation:

- rely on the one-per-line extension budget first
- do not add a separate node budget in AL.1 unless the first code draft proves the one-budget guard is insufficient

Reason:

- adding both a new extension and a new node-budget mechanism would broaden the experiment unnecessarily

## Recommended Hook Points

Primary future edit point:

- `src/coreClaude/search/alphabeta.ts`
  - `negamax(...)`
  - immediately after current `d = extendTacticalDepth(...)`

Secondary alignment point:

- `src/coreClaude/search/alphabeta.ts`
  - root loop in `iterativeDeepening(...)`
  - only if we decide root should mirror the same AL.1 horizon behavior

Recommended default:

- start in `negamax(...)` only
- avoid root-specific duplication unless benchmarking shows the recursive hook alone is insufficient

## Future Files / Functions To Edit

Primary:

- `src/coreClaude/search/alphabeta.ts`
  - `negamax(...)`
  - possibly helper extraction for AL.1 eligibility

Optional only if needed for cleanliness:

- `src/coreClaude/search/alphabeta.ts`
  - function signature for `negamax(...)` to carry extension budget
  - root loop in `iterativeDeepening(...)` if budget must be initialized explicitly there

Not needed for first implementation:

- `src/coreClaude/eval.ts`
- `src/coreClaude/search/levelPolicy.ts`
- `src/coreClaude/movegen.ts`
- opening book files
- root tie-break logic

## Benchmark Plan

Clean OFF baseline:

1. `npm.cmd run test:perft`
2. `npm.cmd run gate:ai:report`
3. `npm.cmd run gate:ai:repeat`

Experiment ON repeated check:

1. `$env:MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION='1'`
2. `npm.cmd run test:perft`
3. `npm.cmd run gate:ai:report`
4. `npm.cmd run gate:ai:repeat`
5. `Remove-Item Env:MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`

Required case review order:

1. `sac-two-win-three-p1`
2. `sac-two-win-three-p2`
3. `low-mobility-squeeze`
4. `low-mobility-squeeze-p2`
5. `quiet-hanging-piece-p1`
6. `small-piece-king-vs-men`
   - informational only
   - treat as noisy/oracle-probe case

Reject immediately if:

- `test:perft` fails
- default-OFF path changes
- `sac-two-win-three-p1` catastrophically regresses
- `sac-two-win-three-p2` materially regresses across repeats

Interpretation guidance:

- if only noisy cases move:
  - do not claim tactical gain
- if low-mobility improves but protected tactical cases regress:
  - reject
- if protected tactical cases stay clean and low-mobility improves repeatedly:
  - classify as promising only, not promoted

## Risk Level

Risk level:

- moderate

Why not low:

- the engine already has tactical extensions
- the experiment overlaps conceptually with current behavior
- near-horizon tweaks can still interact with quiescence and root tactical overrides

Why not high:

- the proposed distinct slice is very narrow
- OFF-by-default flag keeps default runtime unchanged
- one-per-line budget sharply bounds the blast radius

## Recommendation

Best Phase AL.1 conclusion:

- do not implement a generic capture `+1` extension
- it would mostly duplicate existing tactical extension logic
- if Phase AL.1 proceeds, implement only the distinct `depth == 1` near-horizon slice behind:
  - `MAKHOS_ENABLE_PHASE_AL1_NEAR_HORIZON_CAPTURE_EXTENSION`
- keep it one-per-line, `+1` max, and benchmark protected cases first
