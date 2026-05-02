# Teacher Training (Makhos AZ)

This guide is for the teacher/supervised phase after self-play regression.

Goal:
- start from the strongest known baseline, currently `iter_0094`
- fix tactical blunders using minimax-labeled data
- use soft root-candidate policy targets, not only one-hot best moves
- re-enter long self-play only after passing clear external eval gates

## 1) Build Teacher Dataset

Run from repo root:

```powershell
python newAz/build_teacher_data.py `
  --output "G:\My Drive\makhos_az_v_teacher\teacher_data\teacher_v2_soft_d7_from_0094.npz" `
  --samples 8000 `
  --minimax-depth 7 `
  --playout-minimax-depth 5 `
  --policy-mode soft `
  --policy-top-k 4 `
  --policy-margin 80 `
  --policy-temp 35 `
  --profile hard `
  --model-checkpoint "G:\My Drive\makhos_az_v_teacher\models\iter_0094.pt" `
  --resume `
  --save-every 200
```

What it does:
- samples positions from mixed sources:
  - random playout
  - minimax-guided playout
  - forced-capture positions
  - model-vs-minimax disagreement positions
- labels policy with minimax root candidates as a soft distribution
- labels value with normalized minimax score (`tanh(score / value_scale)`)
- balances side-to-move (`P1/P2`) as samples are collected
- saves:
  - dataset `.npz`
  - metadata `.json` (same basename)

Recommended starting settings:
- use `--minimax-depth 7` first
- then try depth 9 only on smaller datasets if compute budget allows
- for recovery experiments, use `--profile hard` + smaller samples first (e.g. 2000-3000)

Optional email notification:
- set env vars before running:
  - `NOTIFY_EMAIL_TO`
  - `NOTIFY_EMAIL_FROM`
  - `NOTIFY_EMAIL_PASSWORD` (app password/token)
- both `build_teacher_data.py` and `train_teacher.py` will send completion email automatically if these are set.

Resume behavior:
- `build_teacher_data.py --resume`:
  - writes partial checkpoint every `--save-every` samples
  - rerun same command to continue from last partial progress after disconnect
- `train_teacher.py --resume --save-every-epoch`:
  - writes training state every epoch
  - rerun same command to continue from next epoch

## 2) Train Teacher Model

```powershell
python newAz/train_teacher.py `
  --dataset "G:\My Drive\makhos_az_v_teacher\teacher_data\teacher_v2_soft_d7_from_0094.npz" `
  --baseline-checkpoint "G:\My Drive\makhos_az_v_teacher\models\iter_0094.pt" `
  --output-dir "G:\My Drive\makhos_az_v_teacher\models" `
  --epochs 6 `
  --batch-size 256 `
  --lr 3e-5 `
  --anchor-weight 0.08 `
  --anchor-value-weight 0.35 `
  --resume `
  --save-every-epoch
```

Output:
- `teacher_iter0094_d7_YYYYMMDD.pt`
- `teacher_iter0094_d7_YYYYMMDD.trainlog.json`

Loss terms (logged separately each epoch):
- `policy_ce` (teacher move imitation)
- `value_loss` (teacher value target)
- `anchor_loss` (stability regularization against baseline checkpoint)

Total optimized loss:
- `policy_ce + value_weight * value_loss + anchor_weight * anchor_loss`

## Colab Email Setup Example

In Colab, run this once before build/train:

```python
import os, getpass
os.environ["NOTIFY_EMAIL_TO"] = "you@example.com"
os.environ["NOTIFY_EMAIL_FROM"] = "your_sender@gmail.com"
os.environ["NOTIFY_EMAIL_PASSWORD"] = getpass.getpass("Gmail app password: ")
```

Then run the same build/train commands as normal. The scripts will notify when done.

## 3) Evaluate Teacher Checkpoint

Use the existing local evaluator:

```powershell
python newAz/eval_local.py `
  --drive-dir "G:\My Drive\makhos_az_v_teacher" `
  --checkpoint "teacher_iter0079_d7_YYYYMMDD.pt"
```

If you copy/rename a teacher checkpoint to iter-style for watcher compatibility:

```powershell
python newAz/eval_local.py --drive-dir "G:\My Drive\makhos_az_v_teacher" --checkpoint iter_0079_teacher_d7.pt
```

## 4) Eval Gate Policy (Before Long Self-Play)

Do not start at mm11.

Gate order:
1. tactical fixed suite
2. vs random
3. vs mm3
4. vs mm5
5. vs mm7
6. vs mm9
7. mm11 only when mm7/mm9 are clearly stable

Decision rules:
- fail mm5 -> discard candidate
- mm7 < 50% -> do not spend on mm9/mm11
- mm9 not stable at >= 50% -> skip mm11
- always inspect both P1/P2 split (not only aggregate)

## 5) Suggested Next Loop

1. Build depth-7 teacher dataset.
2. Train from `iter_0079` with anchor regularization.
3. Evaluate with gates through mm7/mm9.
4. Only if stable, run sparse mm11.
5. Then resume self-play from the teacher checkpoint.
