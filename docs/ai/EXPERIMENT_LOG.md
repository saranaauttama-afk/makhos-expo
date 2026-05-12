# AI Experiment Log

Use this as a short append-only log for tiny research loops.

Primary references:

- `../../CURRENT_TASK.md`
- `../../ENGINE_STATUS.md`
- `../../PROJECT_HANDOFF_ENGINE_STATUS.md`

## 2026-05-12

- Loop type: docs/bootstrap
- Change: add missing task/log docs without altering runtime behavior
- Decision: safe to keep; no experiment promoted
- Next: resume with one debug/tooling or guarded experiment-only loop

## 2026-05-12 - Post-revert baseline

- Loop type: cleanup verification
- Change: reverted `src/coreClaude/eval.ts` to HEAD; kept docs stubs
- Verification:
  - `npm run test:perft` passed
  - `npm run gate:ai:report` = `WARN`, `fatalReasons=(none)`
  - `npm run gate:ai:repeat` = `WARN/WARN/WARN`
  - `sac-two-win-three-p1` stayed clean in all 3 clean-OFF repeat runs
- Decision: do not tune or promote; runtime baseline restored
- Next: continue with docs/tooling only or choose a new isolated experiment later
