# newAz

Split training pipeline for Makhos AlphaZero:

- `Colab` does self-play, training, checkpoint saving, and only a light quick-eval
- `Local machine` does the heavy full evaluation
- both sides communicate through shared files inside the same Drive folder

This folder is intentionally outside `colab/` so the original training setup stays intact.

---

## Super Simple Usage

If you want the shortest possible version:

1. Colab runs `train_colab.py`
2. Local machine runs the watcher
3. Local machine writes `result` + `decision`
4. Colab picks those decisions up later

### What you should run on your local machine

PowerShell:

```powershell
.\newAz\run_local_watch.ps1 -DriveDir "D:\GoogleDrive\makhos_az_v5"
```

Check current status:

```bash
python newAz/check_status.py --drive-dir "D:\GoogleDrive\makhos_az_v5"
```

If you see pending requests but no results, your local evaluator is not running yet.

---

## Why This Exists

The previous loop became bottlenecked by evaluation.

Typical pain point:
- training on Colab is still fast enough
- heavy evaluation against deep minimax plus opening suite can take `15-16 hours`
- during that time, the expensive GPU runtime is mostly waiting on CPU-heavy eval logic

So the idea here is:

1. keep Colab focused on producing checkpoints
2. move expensive validation to the local machine
3. let both sides exchange state only through safe artifact files

This avoids corrupting the training state while still letting us evaluate every important checkpoint.

---

## Folder Contents

| File | Purpose |
|---|---|
| `train_colab.py` | Colab training loop in external-eval mode |
| `eval_local.py` | Full local evaluator for saved checkpoints |
| `check_status.py` | Easy status summary for requests/results/decisions |
| `run_local_watch.ps1` | Windows helper to start the local watcher quickly |
| `battle_verify.ts` | Optional TypeScript final verify against the app-side engine |
| `makhos_engine.py` | Copied engine dependency |
| `network_az.py` | Copied network dependency |
| `mcts_az.py` | Copied MCTS dependency |

---

## Core Concept

### Colab owns training state

These files belong to the training loop and should only be written by Colab:

- `models/latest.pt`
- `models/train_state.pt`
- `replay_buffer.npy`
- `training_log.jsonl`

### Local machine owns full evaluation artifacts

These files are produced by the local evaluator:

- `external_eval/results/*.result.json`
- `external_eval/decisions/*.decision.json`
- `external_eval/target_status.json`
- `external_eval/external_eval_log.jsonl`

### Shared handshake

Colab writes evaluation requests:

- `external_eval/requests/iter_XXXX.request.json`

Local machine reads those requests and writes:

- `external_eval/results/iter_XXXX.result.json`
- `external_eval/decisions/iter_XXXX.decision.json`
- `external_eval/target_status.json` when a new target is accepted

Colab can then read the `decision` files on the next iterations and optionally promote:

- `best.pt`
- `target_best.pt`

without giving the local evaluator permission to touch `latest.pt` or `train_state.pt`.

---

## Drive Layout

Recommended shared folder:

```text
MyDrive/makhos_az_v5/
  makhos_engine.py
  network_az.py
  mcts_az.py
  train_colab.py
  models/
    latest.pt
    best.pt
    target_best.pt
    iter_0009.pt
    iter_0019.pt
    train_state.pt
  replay_buffer.npy
  training_log.jsonl
  external_eval/
    requests/
      iter_0009.request.json
    results/
      iter_0009.result.json
    decisions/
      iter_0009.decision.json
    target_status.json
    external_eval_log.jsonl
```

---

## How `train_colab.py` Works

Compared with the heavier in-loop setup, this version is adjusted for split evaluation:

- saves checkpoint every `EVAL_INTERVAL`
- writes an external eval request JSON for each checkpoint
- runs only a quick sanity eval on Colab
- does not promote `best.pt` from quick eval by default
- reads local decisions and can auto-adjust loss-mining policy from them
- can read local `decision` files and promote `best.pt` / `target_best.pt`

In short:

- Colab is now a checkpoint producer
- local machine is the final judge of practical strength

---

## How `eval_local.py` Works

The local evaluator runs the expensive parts:

- `candidate vs best`
- `candidate vs random`
- `candidate vs minimax-3/5/7/9/11`
- opening suite evaluation
- target comparison logic

But it now uses gated depth escalation:

- always evaluate `mm3`, `mm5`, `mm7`
- only continue to `mm9` if `wr_vs_mm7 >= 0.35`
- only continue to `mm11` if `wr_vs_mm9 >= 0.45`
- only run opening suite if the checkpoint is at least competitive at `mm7`

This is deliberate: if a model is still clearly failing at `mm7`, spending hours on `mm9` and `mm11` usually adds very little decision value.

It also uses two deployment-focused helpers:

- reads `target_status.json` first, so the current target baseline usually does not need to be recomputed
- can optionally run a final TypeScript battle verify before trusting a checkpoint as the new target

It then writes:

1. a detailed result JSON
2. a machine-readable decision JSON
3. an append-only eval log

Decision logic:

- `promote_best` is based on `candidate vs best`
- `promote_target` is based on `mm11 + opening suite + side floor`
- if no prior `target_best.pt` exists, it falls back to absolute thresholds
- it also emits a recommended loss-mining policy for Colab

### Auto Loss-Mining Policy

The local evaluator now recommends whether Colab should mine losses:

- if `mm5 < 0.20` -> keep loss mining off
- if `mm5 >= 0.20` but `mm7 < 0.35` -> light mining at depth 5
- if `mm7 >= 0.35` but `mm9 < 0.45` -> medium mining at depth 7
- if `mm9 >= 0.45` but `mm11 < 0.50` -> deeper mining at depth 9
- if the model is already competitive late-stage -> keep only light maintenance mining

Colab stores the active mining policy in `train_state.pt` and also reloads it from `target_status.json` when available.

---

## Suggested Workflow

### Step 1: Upload files to Drive for Colab

Upload these from `newAz/` into your shared training folder:

- `train_colab.py`
- `makhos_engine.py`
- `network_az.py`
- `mcts_az.py`

### Step 2: Run Colab training

In Colab, open `train_colab.py` and run the notebook cells as usual.

Colab will:

- train
- save `iter_XXXX.pt`
- create `external_eval/requests/iter_XXXX.request.json`

### Step 3: Sync the same Drive folder to your local machine

Your local machine needs access to the same directory tree.

Examples:

- Google Drive Desktop synced folder
- manually copied mirror of the training folder
- mounted cloud drive path

### Step 4: Run local full evaluation

One-shot evaluation:

```bash
python newAz/eval_local.py --drive-dir "D:\\GoogleDrive\\makhos_az_v5" --checkpoint iter_0049
```

Watch mode:

```bash
python newAz/eval_local.py --drive-dir "D:\\GoogleDrive\\makhos_az_v5" --watch
```

Watch mode with final TypeScript verify:

```bash
python newAz/eval_local.py --drive-dir "D:\\GoogleDrive\\makhos_az_v5" --watch --ts-verify
```

Single pending request then exit:

```bash
python newAz/eval_local.py --drive-dir "D:\\GoogleDrive\\makhos_az_v5" --once
```

### Step 5: Let Colab pick up decision files

`train_colab.py` polls `external_eval/decisions/` during training.

If a new decision says:

- `promote_best = true`
- `promote_target = true`

then Colab can load the chosen checkpoint and update:

- `best.pt`
- `target_best.pt`

without interrupting replay buffer or optimizer state.

---

## File Contract

### Request JSON

Example:

```json
{
  "iter": 49,
  "checkpoint": "/content/drive/MyDrive/makhos_az_v5/models/iter_0049.pt",
  "created_at": 1760000000,
  "status": "pending"
}
```

### Result JSON

Example fields:

```json
{
  "iter": 49,
  "checkpoint": ".../iter_0049.pt",
  "wr_vs_best": 0.567,
  "wr_vs_random": 1.0,
  "wr_vs_minimax3": 0.875,
  "wr_vs_minimax5": 0.625,
  "wr_vs_minimax11": 0.375,
  "wr_opening_suite": 0.531,
  "wr_opening_floor": 0.500,
  "gated_at_depth": null
}
```

If deeper stages are skipped, fields such as `wr_vs_minimax11` or `wr_opening_suite` may be absent or `null`, and `gated_at_depth` will show where the evaluator stopped.

### Decision JSON

Example fields:

```json
{
  "iter": 49,
  "checkpoint": ".../iter_0049.pt",
  "promote_best": true,
  "promote_target": false,
  "ts_verify_win_rate": 0.583,
  "summary": "best=0.567 mm11=0.375 open=0.531 floor=0.500 ..."
}
```

Optional TS verify block in result JSON:

```json
{
  "ts_verify": {
    "win_rate": 0.583,
    "games": 12,
    "depth": 11,
    "sims": 1600
  }
}
```

---

## Safety Rules

These are important:

1. Local evaluator should not overwrite `latest.pt`
2. Local evaluator should not overwrite `train_state.pt`
3. Colab should remain the only writer of replay buffer state
4. Promotion decisions should happen through JSON handoff, not direct mutation from the local machine
5. Quick Colab eval should not be trusted as the final deployment judge

This keeps race conditions much lower.

---

## Practical Tradeoff

This setup improves wall-clock efficiency, but with a tradeoff:

- Colab is no longer doing the full target-selection loop immediately at every checkpoint
- decisions may be applied with a delay of one or more iterations

That is usually acceptable, because:

- GPU time is more valuable than immediate deep eval
- external eval is still the stronger signal for deployment decisions
- training no longer stalls for many hours on the Colab side

---

## Good Next Steps

If this pipeline works well, the next upgrades I would consider are:

1. separate quick Colab metrics from deployment metrics more clearly in logs
2. cache previous full-eval summaries for `best.pt` too
3. tighten TS verify thresholds once enough empirical results exist
4. move email credentials out of the script into environment variables

---

## Session Notes (2026-04-08)

### What We Set Up

- Created `train_colab.py` + `eval_local.py` for split pipeline
- Converted `train_colab.py` → `train_colab.ipynb` (8-cell notebook) and uploaded to Drive
- Uploaded `makhos_engine.py`, `network_az.py`, `mcts_az.py` to Drive folder
- Started v5 training fresh from iter 0 on T4 GPU

### Current Config Choices

**Loss Mining: AUTO FROM LOCAL DECISIONS**

Current behavior:

- base config still starts with `ENABLE_LOSS_MINING = False`
- `AUTO_LOSS_MINING_FROM_DECISIONS = True`
- `eval_local.py` now recommends whether mining should be off / light / medium / deeper
- `train_colab.py` reads those decision files and updates the current mining policy automatically on resume / next loop

Current recommendation policy:

- if `mm5 < 0.20` -> keep mining off
- if `mm5 >= 0.20` but `mm7 < 0.35` -> stronger light mining at depth 5
- if `mm7 >= 0.35` but `mm9 < 0.45` -> medium mining at depth 7
- if `mm9 >= 0.45` but `mm11 < 0.50` -> deeper mining at depth 9
- otherwise keep only maintenance mining to avoid late-stage overfitting

This means Colab is still primarily a checkpoint producer, but it can now adapt mining budget using the external evaluator's frontier signal.

**Resume LR: REDUCED FOR ROUND 2**

- `train_colab.py` now resumes with `LR = 1e-4`
- this is intentional after the first plateau around `mm7`
- the goal is to keep learning moving while reducing late-stage overshoot

**Run Policy: 20-Iter Blocks**

- Colab now runs in blocks of `20` iterations at a time
- checkpoints are still saved every `10` iterations
- this keeps checkpoint cadence intact while making it easier to stop, review, and adjust without burning too much Colab time

**Heavy Eval Policy**

- `mm9` can still run whenever the `mm7` gate is passed
- `mm11` now runs only every `20` iterations
- opening suite is paired with those heavy `mm11` checkpoints only
- practical verify `mm7 @ 800 sims` stays available between heavy checkpoints

**Opening Suite: NOT run during training**

Also by design. `eval_local.py` runs the opening suite as part of full evaluation. It does not feed results back into training.

### Training Status

| Iter | mm3 | mm5 | mm7 | Notes |
|---|---|---|---|---|
| 9 | 50% | 0% | 0% | First checkpoint, gated at mm7, mining should remain off |

### Loss Mining: When To Enable

No manual toggle is required anymore if you use the updated files:

- local eval writes a mining recommendation into each `decision.json`
- Colab picks that up automatically
- you only need to restart / rerun the notebook to load the updated code, not restart training from scratch

For the current checkpoint trajectory, iter `0009` is still too weak for mining. Let the next 1-2 checkpoints land first unless the model is obviously stuck at the same frontier.

---

## Claude Handoff Summary

If you are reviewing this system quickly, these are the important facts:

- `newAz/` is a split pipeline created to keep GPU-heavy Colab training separate from CPU-heavy evaluation
- `train_colab.py` is the Colab source of truth, and `train_colab.ipynb` is generated from it for direct use from Drive
- the same updated `train_colab.py` and `train_colab.ipynb` were synced to `MyDrive/makhos_az_v5/`
- local evaluation is gated: always `mm3/mm5/mm7`, then only continue to `mm9` and `mm11` if the checkpoint is competitive enough
- Colab now runs in `20`-iteration blocks instead of one long uninterrupted run
- `mm11` is no longer run at every checkpoint; it is now a sparse heavy eval that only fires every `20` iterations
- `ALLOW_QUICK_PROMOTE_BEST = False`, so quick Colab eval is informational only
- `eval_local.py` emits promotion decisions plus loss-mining recommendations
- `train_colab.py` now applies those decisions and auto-adjusts mining policy
- the round-2 config lowers resume LR to `1e-4` and increases the light mining tier to `2 games / 6 positions / max 16 samples`
- `eval_local.py` now also runs an automatic practical verify at `mm7 @ 800 sims` for checkpoints that already look strong enough at the main gate
- Colab also resolves Windows-style checkpoint paths from local decisions back into Drive paths before applying them
- current known eval result: `iter_0009` reached `mm3 = 50%`, `mm5 = 0%`, `mm7 = 0%`, so mining should still stay off for now
- invariant to preserve: `Colab owns training state; local machine owns heavy evaluation output`

## Review Notes For Claude

This setup is intentionally biased toward robustness over elegance.

Questions worth reviewing:

- whether `promote_target` logic should require direct comparison with the previous target every time
- whether `promote_best` should also require optional TypeScript verification
- whether local watch mode should also mark processed request files
- whether TS verify should run only for `promote_target` candidates or for `promote_best` too

If a reviewer wants to simplify the system, the main invariant to preserve is:

`Colab owns training state; local machine owns heavy evaluation output.`
