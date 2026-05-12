# CURRENT_TASK

## Current Safe Loop

- Status: post-revert safe baseline
- Goal: preserve the restored runtime baseline and make the next tiny loop resumable.
- Do not promote any experiment behavior by default.

## Inferred Current State

- Root status docs remain the source of truth:
  - `ENGINE_STATUS.md`
  - `PROJECT_HANDOFF_ENGINE_STATUS.md`
- `src/coreClaude/eval.ts` was reverted to HEAD after the unsafe blocked-asymmetry runtime experiment.
- Runtime is clean again; no `MAKHOS_*` env flags are currently set.
- Latest verification state:
  - `npm run test:perft` passed
  - `npm run gate:ai:report` = `WARN`, `fatalReasons=(none)`
  - `npm run gate:ai:repeat` = `WARN/WARN/WARN`
  - `sac-two-win-three-p1` stayed clean in all 3 clean-OFF repeat runs
- Conclusion:
  - do not tune
  - do not promote
  - runtime baseline restored

## Protected Cases

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`
- `low-mobility-squeeze`
- `low-mobility-squeeze-p2`
- `quiet-hanging-piece-p1`
- `small-piece-king-vs-men`

## Next Safe Action

- Continue with docs/tooling only, or choose one new isolated experiment later.
- Keep any future runtime experiment behind explicit `OFF` flags with immediate repeat-run verification.
