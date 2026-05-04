# Makhos AlphaZero V6 - Setup Guide

## Goal
Beat MM depth-11 with 400 Colab units (~21 days)

## V6 Key Improvements

### 1. **Efficient Evaluation** (Save ~150 units!)
- **Before:** 60-100 games per eval
- **After:** 16 games (8 openings × 2 sides)
- **Savings:** 6x faster eval → can eval every 3 iterations!

### 2. **Tactical Focus**
- Loss mining from **MM11** (not MM7)
- Higher SIMS: **500** (better tactical vision)
- More mining samples: **96** per iteration

### 3. **No MM Self-play**
- **0%** games vs MM during training
- **75%** games vs opponent pool (diversity!)
- **25%** games vs self

### 4. **Progressive Target**
- **iter 95-105:** Focus on MM9 (get to 90%+)
- **iter 106+:** Attack MM11 (get to 55%+)

### 5. **Adaptive SIMS**
- **iter <110:** 500 sims (best quality)
- **iter 110-120:** 300 sims (faster)
- **iter 120+:** 250 sims (even faster)

---

## Google Drive Setup

### Step 1: Create Folder Structure

On Google Drive, create:
```
MyDrive/
  └── makhos_az_v6/
      ├── makhos_engine.py
      ├── network_az.py
      ├── mcts_az.py
      ├── train_az.py
      └── models/          (will be auto-created)
```

### Step 2: Upload Files

Upload these 4 files from `makhos_az_v6/` folder:
1. ✅ `makhos_engine.py`
2. ✅ `network_az.py`
3. ✅ `mcts_az.py`
4. ✅ `train_az.py`

### Step 3: Upload Checkpoint (if resuming)

If resuming from iter_0094:
1. Get `iter_0094.pth` from v5
2. Upload to `makhos_az_v6/models/`
3. In train_az.py, set:
   ```python
   FORCE_BASELINE_ITER = 94
   ```

---

## Colab Notebook Setup

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Import train_az
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/makhos_az_v6')
from train_az import *
```

### Cell 3: Run Training
```python
# Training will run for 10 iterations, then stop
# Review logs, then run again to continue
```

---

## Expected Timeline

### **Phase 1: iter 95-105 (150 units)**
**Goal:** MM9 > 90%

Eval every 3 iterations:
- iter 95: Baseline check
- iter 98: Should see MM9 ~80%+
- iter 101: Should see MM9 ~85%+
- iter 104: Should see MM9 ~90%+ ✅

**Budget:** ~15 units/iter × 10 = 150 units

---

### **Phase 2: iter 106-120 (150 units)**
**Goal:** MM11 > 55%

Progressive difficulty:
- iter 106-110: MM11 ~30-40%
- iter 111-115: MM11 ~45-50%
- iter 116-120: MM11 ~55%+ ✅

**Budget:** ~10 units/iter × 15 = 150 units

---

### **Phase 3: iter 121-140 (150 units) [BONUS]**
**Stretch Goal:** MM11 > 65%

If units allow:
- iter 121-130: MM11 ~60%
- iter 131-140: MM11 ~65%+ 🎯

**Budget:** ~10 units/iter × 15 = 150 units

---

## Monitoring Progress

### Check training_log.jsonl:
```python
!tail -20 /content/drive/MyDrive/makhos_az_v6/training_log.jsonl
```

### Look for:
```json
{
  "iteration": 98,
  "mm9_rate": 0.812,    // Good! Climbing
  "mm11_rate": 0.312,   // Too early
  "loss": 0.145,        // Should decrease slowly
  "lr": 4.5e-5,         // Will decay gradually
  "best_replaced": true // Yay!
}
```

---

## Rollback Strategy

If regression occurs:
1. **Automatic rollback** if performance drops >12%
2. **LR decay** to 70% after 3 plateau iterations
3. **Manual rollback:** Set `FORCE_BASELINE_ITER = X`

---

## Success Criteria

### **Minimum (MVP):**
- ✅ iter 105: MM9 ≥ 90%
- ✅ iter 120: MM11 ≥ 55%

### **Stretch:**
- 🎯 iter 140: MM11 ≥ 65%

---

## Troubleshooting

### Q: Training is slow?
**A:** Check if using GPU:
```python
print(f'Device: {DEVICE}')  # Should show 'cuda'
```

### Q: Out of memory?
**A:** Reduce batch size:
```python
BATCH_SIZE = 128  # from 256
```

### Q: Regression happening?
**A:**
1. Check if LR too high → will auto-decay
2. Check replay buffer → should be >50k
3. May need to rollback → auto after 3 iterations

### Q: Not beating MM11?
**A:**
1. Check if LOSS_MINING_DEPTH = 11 ✅
2. Check if N_SIMS = 500 ✅
3. May need more iterations (up to 140)

---

## File Descriptions

### `makhos_engine.py`
- Board representation (bitboards)
- Move generation
- Position evaluation (hand_eval)
- Feature extraction for NN

### `network_az.py`
- Neural network architecture (256-dim, 6 ResBlocks)
- Policy head (1024 moves)
- Value head (scalar output)
- Inference wrapper

### `mcts_az.py`
- Monte Carlo Tree Search
- PUCT formula
- Dirichlet noise
- Backpropagation

### `train_az.py`
- Training loop
- Self-play
- Replay buffer
- Eval vs MM
- Loss mining
- Checkpointing

---

## Contact

If stuck, check:
1. training_log.jsonl for errors
2. models/ folder for checkpoints
3. Colab session didn't timeout

Good luck! 🎯
