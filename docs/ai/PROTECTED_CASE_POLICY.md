# Protected Case Policy

This file defines how protected benchmark cases should be interpreted during engine research.

Use this together with:

- `ENGINE_MASTER_ROADMAP.md`
- `docs/ai/EXPERIMENT_RESULTS.md`

## Core Rule

Protected cases remain higher priority than secondary benchmark improvements.

If a change improves secondary cases but clearly worsens a protected case, reject the change.

## Primary Protected Cases

- `sac-two-win-three-p1`
- `sac-two-win-three-p2`

## `sac-two-win-three-p2`

Classification:

- protected
- tactical
- stable

Policy:

- continue treating `sac-two-win-three-p2` as a normal hard-stop protected case
- a clear repeated regression is a direct reject signal

## `sac-two-win-three-p1`

Classification:

- protected
- tactical
- repeat-required
- oracle-sensitive
- benchmark-policy special case

### Why `p1` is special

BD1-BD3 diagnostics showed:

- the quick oracle fallback path is unstable
- larger oracle budgets alone do not stabilize it
- quick root fallback and root override do not reliably track the strongest independent reference ordering
- stronger independent scoring shows a stable high-confidence winning band rather than one single quick-oracle move that should be treated as absolute truth

### Accepted Reference Band

Provisional accepted band for `sac-two-win-three-p1`:

- `7->3K`
- `8->4K`
- `20->16`
- `31->26`
- `7->2K`

Interpretation:

- `7->3K` is the strongest current reference move
- the full band should be treated as the accepted winning cluster for benchmark judgment
- quick-oracle drift within this band is not, by itself, evidence of an engine regression

### Warning Rules For `p1`

Treat `p1` as `warning-only` for a single run when:

- the quick oracle move changes
- but the final selected move still stays inside the accepted reference band

Also treat as warning-only when:

- repeated runs show band-internal drift without a clear increase in catastrophic outcomes

### Fail Rules For `p1`

Treat `p1` as a fail signal when:

- an experiment pushes the final selected move outside the accepted reference band repeatedly
- catastrophic outcomes increase clearly across repeated runs
- the move leaves the accepted band and the stronger diagnostic reference still supports the band

### Repeat Requirement

Never judge `p1` from one quick run alone.

Minimum policy:

- use repeated-run confirmation
- compare band membership, not just one oracle move string
- compare catastrophic rate, not just one label

### What Not To Do

- do not pin a single oracle move in engine runtime
- do not treat a single quick oracle move as authoritative
- do not remove `p1` from protected status
- do not resume autonomous queue execution just because one quick run is clean

## Manual Experiment Guidance

After BD4, manual isolated experiments may resume cautiously if:

- `test:perft` passes
- `sac-two-win-three-p2` remains clean
- `sac-two-win-three-p1` is reviewed using this band-aware repeated-run policy
- autonomous execution remains paused until tooling/harness policy catches up

## Autonomous Loop Guidance

Autonomous mode should stay paused until benchmark tooling explicitly understands:

- `p1` accepted-band logic
- repeated-run confirmation requirements
- the distinction between warning-only band drift and true protected-case failure
