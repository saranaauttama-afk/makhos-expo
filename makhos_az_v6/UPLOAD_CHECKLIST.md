# 📦 Upload Checklist for Google Drive

## ✅ Required Files (MUST upload these 4)

### 1. Core Engine Files
```
☐ makhos_engine.py  (23 KB)  - Game rules & eval
☐ network_az.py     (4.9 KB) - Neural network
☐ mcts_az.py        (7.0 KB) - MCTS algorithm
☐ train_az.py       (7.7 KB) - Training loop (V6 config!)
```

### 2. Google Drive Structure
Create this folder on your Google Drive:
```
MyDrive/
  └── makhos_az_v6/          ← Create this folder
      ├── makhos_engine.py   ← Upload here
      ├── network_az.py      ← Upload here
      ├── mcts_az.py         ← Upload here
      ├── train_az.py        ← Upload here
      └── models/            ← Will auto-create
```

---

## 🔄 Optional: Resume from iter_0094

If you want to continue training from checkpoint 94:

```
☐ Get iter_0094.pth from makhos_az_v5/models/
☐ Upload to makhos_az_v6/models/iter_0094.pth
☐ Edit train_az.py:
    FORCE_BASELINE_ITER = 94  (change from None)
```

---

## 📖 Documentation Files (Optional - for reference)

These are for your reference, don't need to upload:
```
☐ README_SETUP.md    (4.8 KB) - Detailed guide
☐ QUICK_START.txt    (3.8 KB) - Quick reference
☐ UPLOAD_CHECKLIST.md (this file)
```

---

## ✅ Verification Steps

After uploading to Google Drive:

### Step 1: Check folder structure
```python
# In Colab:
!ls /content/drive/MyDrive/makhos_az_v6/
```
Expected output:
```
makhos_engine.py
mcts_az.py
network_az.py
train_az.py
models/  (if resuming)
```

### Step 2: Check file sizes
```python
!ls -lh /content/drive/MyDrive/makhos_az_v6/*.py
```
Expected:
```
makhos_engine.py  ~23K
network_az.py     ~5K
mcts_az.py        ~7K
train_az.py       ~8K
```

### Step 3: Test import
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/makhos_az_v6')
from makhos_engine import initial_position
print("✅ Import successful!")
```

---

## 🚀 Start Training

Once verified, run:

```python
# In Colab Cell 1:
from google.colab import drive
drive.mount('/content/drive')

# In Colab Cell 2:
import sys
sys.path.insert(0, '/content/drive/MyDrive/makhos_az_v6')
exec(open('/content/drive/MyDrive/makhos_az_v6/train_az.py').read())

# Training will start automatically!
```

---

## 📊 Expected First Output

After starting, you should see:
```
Device: cuda
Setup OK
════════════════════════════════════════════
Config V6: Optimized for MM11 with 400 units
════════════════════════════════════════════
  self-play: 120 games × 500 sims (adaptive)
  training : 250 steps × batch 256
  replay   : 600,000 positions (6x larger!)
  eval     : every 3 iterations, 16 games
  mining   : depth-11 (MM11!), 96 samples
  target   : MM9 (iter<105) → MM11 (iter>=105)
  mix      : 0% MM, 25% anchor, 75% pool
  lr       : 5.0e-05 → 1.0e-05
════════════════════════════════════════════

Starting iteration 95...
```

---

## ⚠️ Common Issues

### Issue: "File not found"
**Fix:** Check folder name is exactly `makhos_az_v6`

### Issue: "No module named makhos_engine"
**Fix:** Check sys.path.insert() points to correct folder

### Issue: "CUDA out of memory"
**Fix:** In train_az.py, change:
```python
BATCH_SIZE = 128  # from 256
```

### Issue: "No GPU available"
**Fix:** Runtime → Change runtime type → T4 GPU

---

## 📍 Current Location

These files are ready at:
```
d:\My App\makhos-v2\makhos_az_v6\
```

Just drag & drop to Google Drive! 🎯
