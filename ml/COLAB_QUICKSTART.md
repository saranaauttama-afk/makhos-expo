# 🚀 Makhos ML - Google Colab Quick Start

**⚡ TypeScript version เร็วกว่า Pure Python 10 เท่า!**

---

## 📋 Cell 1: Setup (Mount Drive + Install Dependencies)

```python
from google.colab import drive
drive.mount('/content/drive')

# สร้างโฟลเดอร์
!mkdir -p /content/drive/MyDrive/makhos_ml
!mkdir -p /content/drive/MyDrive/makhos_ml/checkpoints

# Install PyTorch
!pip install torch numpy -q

# Install Node.js
!apt-get update > /dev/null 2>&1
!apt-get install -y nodejs npm > /dev/null 2>&1
!npm install -g tsx > /dev/null 2>&1

print("✓ Setup complete!")
```

---

## 📋 Cell 2: Clone Repository & Install

```python
import os

# Clone repo (เปลี่ยน YOUR_USERNAME)
!rm -rf /content/makhos-expo
!git clone https://github.com/YOUR_USERNAME/makhos-expo.git

%cd /content/makhos-expo
!npm install > /dev/null 2>&1

print("✓ Repository ready!")
```

---

## 📋 Cell 3: Generate Training Data (Quick Test - 100 games)

```python
import sys
sys.path.insert(0, '/content/makhos-expo/ml')

from gen_data import generate_data

# Quick test: 100 games (~5-10 minutes)
generate_data(
    total_games=100,
    batch_size=100,
    time_per_move=500,
    output="/content/drive/MyDrive/makhos_ml/training_data_100.npz"
)

print("\n✓ Quick test data saved to Google Drive!")
```

---

## 📋 Cell 4: OR Generate Full Dataset (5,000 games - ใช้แทน Cell 3)

```python
import sys
sys.path.insert(0, '/content/makhos-expo/ml')

from gen_data import generate_data

# Full dataset: 5,000 games (~2-3 hours)
generate_data(
    total_games=5000,
    batch_size=1000,
    time_per_move=1000,
    output="/content/drive/MyDrive/makhos_ml/training_data.npz"
)

print("\n✓ Full dataset saved to Google Drive!")
```

---

## 📋 Cell 4: Inspect Data

```python
import numpy as np

# เลือกไฟล์ที่ต้องการดู
data_file = "/content/drive/MyDrive/makhos_ml/training_data.npz"  # หรือ training_data_100.npz

data = np.load(data_file)
states = data['states']
policy_targets = data['policy_targets']
legal_masks = data['legal_masks']
values = data['values']

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"File: {data_file}")
print(f"\nStates: {states.shape}")
print(f"Policy targets: {policy_targets.shape}")
print(f"Legal masks: {legal_masks.shape}")
print(f"Values: {values.shape}")

wins_p1 = (values == 1).sum()
draws = (values == 0).sum()
wins_p2 = (values == -1).sum()
total = len(values)

print(f"\nValue distribution:")
print(f"  P1 wins: {wins_p1:,} ({wins_p1/total*100:.1f}%)")
print(f"  Draws:   {draws:,} ({draws/total*100:.1f}%)")
print(f"  P2 wins: {wins_p2:,} ({wins_p2/total*100:.1f}%)")

print(f"\nAvg legal moves: {legal_masks.sum(axis=(1,2)).mean():.1f}")
print("=" * 60)
```

---

## 📋 Cell 5: Train Model

**Copy ทั้งหมดจาก `ml/COLAB_TRAIN.py` มาวางใน cell นี้**

จากนั้นเพิ่มคำสั่งนี้ท้าย cell:

```python
# Quick test: Simple model, 30 epochs (~30-60 นาที)
train_makhos_model(
    data_path="/content/drive/MyDrive/makhos_ml/training_data.npz",
    model_type="simple",
    hidden_size=512,
    epochs=30,
    batch_size=64,
    lr=0.001,
    output_dir="/content/drive/MyDrive/makhos_ml/checkpoints"
)

print("\n✓ Training complete! Models saved to Google Drive!")
```

**สำหรับ production:** ใช้ ResNet model และ 50 epochs:
```python
train_makhos_model(
    data_path="/content/drive/MyDrive/makhos_ml/training_data.npz",
    model_type="resnet",
    num_channels=128,
    num_res_blocks=6,
    epochs=50,
    batch_size=32,
    output_dir="/content/drive/MyDrive/makhos_ml/checkpoints"
)
```

---

## 📋 Cell 6: Test Trained Model

**หมายเหตุ:** ใช้ฟังก์ชัน `create_model()` ที่โหลดจาก Cell 5 แล้ว

```python
import torch
import numpy as np

# Load model (ใช้ create_model จาก Cell 5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = create_model("simple", hidden_size=512)
checkpoint = torch.load('/content/drive/MyDrive/makhos_ml/checkpoints/best_model.pt',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✓ Model loaded!\n")

# Load test data
data = np.load('/content/drive/MyDrive/makhos_ml/training_data.npz')
states = data['states']
policy_targets = data['policy_targets']
legal_masks = data['legal_masks']
values = data['values']

# Test on random position
idx = np.random.randint(len(states))
state = states[idx]
true_policy = policy_targets[idx]
legal_mask = legal_masks[idx]
true_value = values[idx]

# Predict
state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
with torch.no_grad():
    policy_logits, value_pred = model(state_tensor)

    # Apply legal mask
    logits_flat = policy_logits.view(-1)
    mask_flat = torch.from_numpy(legal_mask.reshape(-1)).to(device)
    masked = torch.where(mask_flat > 0, logits_flat, torch.tensor(-1e9).to(device))

    policy_probs = torch.softmax(masked, dim=0).cpu().numpy()
    value_pred = value_pred.item()

# Top-5 moves
legal_indices = np.where(legal_mask.reshape(-1) > 0)[0]
legal_probs = policy_probs[legal_indices]
top_5 = legal_indices[np.argsort(legal_probs)[-5:][::-1]]

print("=" * 60)
print("MODEL PREDICTION TEST")
print("=" * 60)
print("Top 5 predicted moves:")
for i, move_idx in enumerate(top_5, 1):
    from_sq = move_idx // 32
    to_sq = move_idx % 32
    prob = policy_probs[move_idx]
    print(f"  {i}. {from_sq} → {to_sq}  (p={prob:.4f})")

# True move
true_idx = np.argmax(true_policy.reshape(-1))
true_from = true_idx // 32
true_to = true_idx % 32

print(f"\nTrue move: {true_from} → {true_to}")
print(f"\nValue prediction:")
print(f"  Predicted: {value_pred:+.3f}")
print(f"  True:      {true_value:+.0f}")
print("=" * 60)
```

---

## 📋 Cell 7: Download from Drive (Optional)

หากต้องการ download มาเก็บที่เครื่อง:

```python
from google.colab import files

# Download dataset
files.download('/content/drive/MyDrive/makhos_ml/training_data.npz')

# Download models
files.download('/content/drive/MyDrive/makhos_ml/checkpoints/best_model.pt')
files.download('/content/drive/MyDrive/makhos_ml/checkpoints/final_model_scripted.pt')

print("✓ Download complete!")
```

---

## 🎯 Quick Summary

**ไม่ต้อง clone GitHub! แค่ copy-paste 2 ไฟล์:**

1. **Cell 1**: Setup Drive + PyTorch (1 นาที)
2. **Cell 2**: Copy `COLAB_PURE_PYTHON.py` → Gen 100 games (5-10 นาที)
   - หรือ **Cell 3**: Gen 5,000 games (2-3 ชั่วโมง)
4. **Cell 4**: Check data
5. **Cell 5**: Copy `COLAB_TRAIN.py` → Train model (30-60 นาที)
6. **Cell 6**: Test model
7. **Cell 7**: Download (optional)

---

## 📁 ไฟล์ที่ต้อง copy

- **`ml/COLAB_PURE_PYTHON.py`** → สำหรับ gen data (Cell 2/3)
- **`ml/COLAB_TRAIN.py`** → สำหรับ train model (Cell 5)

---

## ⚠️ หมายเหตุ

- **ไม่ต้อง clone GitHub!** แค่ copy-paste 2 ไฟล์ข้างบน
- **Colab disconnect**: ไฟล์ปลอดภัย (เก็บใน Drive แล้ว)
- **GPU**: Enable GPU ที่ Runtime → Change runtime type → GPU (training เร็วขึ้น 5-10 เท่า!)
- **Quick test**: Cell 2 (100 games) + Cell 5 (30 epochs) ใช้เวลารวม 30-40 นาที
- **Production**: Cell 3 (5000 games) + Cell 5 (50 epochs) ใช้เวลารวม 3-4 ชั่วโมง