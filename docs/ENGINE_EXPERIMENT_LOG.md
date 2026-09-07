# Makhos Engine Experiment Log

This is the permanent evidence log for engine-strength work on `engine-mainline`.

Do not erase failed experiments. A failed idea can prevent future chats/agents from repeating the same work.

## Rules for logging

For every change intended to affect playing strength or search performance, record:

- date,
- baseline commit,
- candidate commit,
- hypothesis,
- exact changed files/features,
- test environment,
- correctness results,
- puzzle/tactical results,
- head-to-head results,
- performance results,
- conclusion: KEEP / REVERT / NEEDS MORE DATA.

Do not promote from one good puzzle or a few self-play games.

---

# Canonical baselines

## Teacher Baseline v0

**Status:** NOT YET RECORDED

**Branch:** `engine-mainline`

**Starting checkpoint:** `db28e143bb06fb8a35a643877a055c5df9770020`

Documentation commits after checkpoint do not count as engine-strength changes.

### Environment

- Date:
- OS:
- CPU:
- RAM:
- Node:
- Commit tested:
- Search/time settings:

### Correctness

| Test | Result | Notes |
|---|---|---|
| `npm run test:rules` | TODO | |
| `npm run test:perft` | TODO | |
| `npm run test:tactical` | TODO | |
| regression harness | TODO | |

### Search benchmark

| Metric | Result |
|---|---:|
| avg nodes | TODO |
| avg depth | TODO |
| avg NPS | TODO |
| avg think time | TODO |

### Puzzle baseline

Historical report: `PUZZLE_BASELINE.md` says 1/14 expected moves correct (7%), but this must be re-run and puzzle answers validated before becoming the canonical v0 number.

| Dataset | Correct | Total | Full-line solved | Avg depth | Avg nodes | Avg time |
|---|---:|---:|---:|---:|---:|---:|
| development | TODO | TODO | TODO | TODO | TODO | TODO |
| holdout | TODO | TODO | TODO | TODO | TODO | TODO |

### Head-to-head

Not applicable until the reproducible A/B harness is established.

### Notes

- Verify `src/coreClaude/search/timeManager.ts` integration before recording time-management behavior.
- Record any suspicious rule/search behavior discovered during baseline runs as issues/tests before strength tuning.

---

# Experiment template

Copy this section for each experiment.

## EXP-YYYY-NNN — Short title

**Status:** PLANNED / RUNNING / KEEP / REVERT / NEEDS MORE DATA

**Date:** YYYY-MM-DD

**Baseline commit:** `<sha>`

**Candidate commit:** `<sha>`

### Hypothesis

One clear statement. Example: “Using full flying-king ray mobility in evaluation improves positional strength without tactical regressions.”

### Change

- files:
- feature/parameter:
- feature flag if applicable:

### Environment

- OS:
- CPU:
- Node:
- mode: fixed-depth / fixed-nodes / fixed-time
- limits:

### Correctness gates

| Gate | Baseline | Candidate | Pass? |
|---|---|---|---|
| Rules | | | |
| Perft | | | |
| Tactical core | | | |
| Tablebase agreement | | | |

### Puzzle results

| Set | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Development | | | |
| Holdout | | | |

List newly solved positions and newly regressed positions separately.

### Head-to-head

- games:
- paired openings:
- W/D/L candidate perspective:
- score %:
- estimated Elo delta:
- confidence interval / sequential-test result:

### Performance

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| nodes/move | | | |
| depth | | | |
| NPS | | | |
| think time | | | |

### Analysis

Why did the result happen? Note tactical regressions, speed-vs-strength tradeoffs, and any suspicious overfitting.

### Decision

**KEEP / REVERT / NEEDS MORE DATA**

Reason:

---

# Historical context worth preserving

These are historical observations from earlier branches/reports and must not be treated as freshly verified results:

- Low-mobility override work around commit `9f7eee3` reported expert benchmark solve 100% and blunder 0%, after fixing a regression caused by a too-aggressive fixed threshold.
- Historical `PUZZLE_BASELINE.md` reported only 1/14 expected puzzle moves correct.
- Earlier NN experiments reported impressive validation/benchmark numbers, but at least one training setup reused the same data for train/validation and some match samples were small. Do not use those numbers as promotion evidence for future models.
- `tuneNN` contains later NN/training/mobile work and some classical changes; recover tools selectively instead of merging the branch wholesale.

The purpose of this section is to preserve context while forcing all future claims through the new baseline/testing protocol.
