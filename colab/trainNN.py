"""
Makhos Neural Network Trainer — Google Colab
=============================================

Architecture : 128 → 256 → 128 → 1  (ReLU, Dropout 0.1)
Input        : 4-channel one-hot board (my_men, my_kings, enemy_men, enemy_kings)
Teacher      : hand-crafted eval scores (centipawns)
GPU required : Yes (T4 on free Colab is fine)

Steps:
  1. Upload data/positions.jsonl to Google Drive
  2. Open this file in Colab (Runtime → Change runtime type → GPU)
  3. Run all cells in order
  4. Download nn_weights.json from Drive → run importNNWeights.ts on PC
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Config (edit paths if needed)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = '/content/drive/MyDrive/makhos/positions.jsonl'
WEIGHTS_PATH = '/content/drive/MyDrive/makhos/nn_weights.json'

EPOCHS      = 300
BATCH_SIZE  = 512
LR          = 3e-4
WEIGHT_DECAY= 1e-4

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load data
# ─────────────────────────────────────────────────────────────────────────────
import json, torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

print("Loading data...")
xs, ts = [], []
with open(DATA_PATH) as f:
    for line in f:
        obj = json.loads(line)
        xs.append(obj['x'])
        ts.append(obj['t'])

X = torch.tensor(xs, dtype=torch.float32)
T = torch.tensor(ts, dtype=torch.float32)
print(f"  {len(X)} positions  |  target mean={T.mean():.1f}cp  std={T.std():.1f}cp")

# Train / val split (90/10)
n_val   = int(len(X) * 0.1)
n_train = len(X) - n_val
ds_train, ds_val = random_split(TensorDataset(X, T), [n_train, n_val])
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Define model
# ─────────────────────────────────────────────────────────────────────────────
class NNEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
model = NNEval().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Train
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn   = nn.MSELoss()

best_val_rmse = float('inf')
best_state    = None

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    for xb, tb in train_loader:
        xb, tb = xb.to(device), tb.to(device)
        optimizer.zero_grad()
        loss_fn(model(xb), tb).backward()
        optimizer.step()
    scheduler.step()

    # Validate
    if epoch % 50 == 0 or epoch == EPOCHS:
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for xb, tb in val_loader:
                xb, tb = xb.to(device), tb.to(device)
                val_mse += loss_fn(model(xb), tb).item() * len(xb)
        val_rmse = (val_mse / n_val) ** 0.5
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:3d}/{EPOCHS}  val RMSE={val_rmse:.1f}cp  (best={best_val_rmse:.1f}cp)")

model.load_state_dict(best_state)
print(f"\nBest val RMSE: {best_val_rmse:.1f} cp")

# R² on full dataset
model.eval()
with torch.no_grad():
    preds = model(X.to(device)).cpu()
ss_res = ((preds - T) ** 2).sum().item()
ss_tot = ((T - T.mean()) ** 2).sum().item()
print(f"R² on full data: {1 - ss_res/ss_tot:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Export weights to Google Drive
# ─────────────────────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

def to_list(t): return t.detach().cpu().numpy().tolist()

weights = {
    "arch": [128, 256, 128, 1],
    "rmse": round(best_val_rmse, 2),
    "W1": to_list(model.net[0].weight),   # [256, 128]
    "b1": to_list(model.net[0].bias),     # [256]
    "W2": to_list(model.net[3].weight),   # [128, 256]
    "b2": to_list(model.net[3].bias),     # [128]
    "W3": to_list(model.net[6].weight),   # [1, 128]
    "b3": to_list(model.net[6].bias),     # [1]
}

with open(WEIGHTS_PATH, 'w') as f:
    json.dump(weights, f)

print(f"Weights saved to {WEIGHTS_PATH}")
print(f"File size: {os.path.getsize(WEIGHTS_PATH)/1024:.0f} KB")
print("\nNext step: download nn_weights.json from Drive, then run:")
print("  npx tsx --tsconfig tsconfig.test.json scripts/importNNWeights.ts")
