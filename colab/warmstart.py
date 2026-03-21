"""
warmstart.py — Pre-train value head จาก positions.jsonl ก่อน self-play
=======================================================================

วางเป็น Cell ใหม่ระหว่าง Cell 5 (eval helper) กับ Cell 6 (training loop)

สิ่งที่ทำ:
  1. โหลด positions.jsonl จาก Drive  (19k positions + hand-crafted scores)
  2. แปลง centipawn score → -1..+1  ด้วย tanh(t / 400)
  3. Train value head 50 epochs บน supervised data
  4. sync best_net ให้ตรงกับ current_net

หลังรัน cell นี้ → รัน Cell 6 (self-play loop) ต่อได้เลย
network จะรู้จักบอร์ดก่อน ทำให้ self-play data มีคุณภาพขึ้นมาก
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5.5 — Warm-start: supervised pre-training บน positions.jsonl
# ─────────────────────────────────────────────────────────────────────────────
import json, math, random
import numpy as np
import torch
import torch.nn.functional as F

POSITIONS_PATH  = f'{DRIVE_DIR}/positions.jsonl'
WARMSTART_EPOCHS = 50
WARMSTART_BATCH  = 512
WARMSTART_LR     = 1e-3
CP_SCALE         = 400.0   # tanh(t/400): ±400cp → ±0.76,  ±800cp → ±0.96

print('── Warm-start: loading positions.jsonl ──────────────────────────────')

if not os.path.exists(POSITIONS_PATH):
    print(f'ERROR: ไม่เจอ {POSITIONS_PATH}')
    print('Upload data/positions.jsonl ไปที่ Drive/makhos_az/ ก่อนครับ')
else:
    # ── โหลด data ────────────────────────────────────────────────────────────
    ws_xs, ws_zs = [], []
    with open(POSITIONS_PATH) as f:
        for line in f:
            obj  = json.loads(line)
            x    = np.array(obj['x'], dtype=np.float32)   # 128-dim
            t    = float(obj['t'])
            z    = math.tanh(t / CP_SCALE)                # -1..+1
            ws_xs.append(x)
            ws_zs.append(z)

    ws_xs = np.stack(ws_xs)                               # (N, 128)
    ws_zs = np.array(ws_zs, dtype=np.float32)            # (N,)
    N     = len(ws_xs)
    print(f'  Loaded {N:,} positions  |  z mean={ws_zs.mean():.3f}  std={ws_zs.std():.3f}')

    # ── Train ─────────────────────────────────────────────────────────────────
    from network import DEVICE
    ws_opt = torch.optim.Adam(current_net.model.parameters(), lr=WARMSTART_LR)

    idx = list(range(N))
    print(f'  Training {WARMSTART_EPOCHS} epochs (batch={WARMSTART_BATCH}, value head only)...')

    for epoch in range(1, WARMSTART_EPOCHS + 1):
        random.shuffle(idx)
        total_loss = 0.0
        batches    = 0

        for start in range(0, N, WARMSTART_BATCH):
            batch_idx = idx[start:start + WARMSTART_BATCH]
            x_b = torch.from_numpy(ws_xs[batch_idx]).to(DEVICE)
            z_b = torch.from_numpy(ws_zs[batch_idx]).to(DEVICE)

            current_net.model.train()
            _, val = current_net.model(x_b)          # (B,)
            val_loss = F.mse_loss(val, z_b)

            ws_opt.zero_grad()
            val_loss.backward()
            ws_opt.step()

            total_loss += val_loss.item()
            batches    += 1

        if epoch % 10 == 0 or epoch == WARMSTART_EPOCHS:
            rmse = math.sqrt(total_loss / batches)
            print(f'  epoch {epoch:3d}/{WARMSTART_EPOCHS}  value RMSE={rmse:.4f}')

    # ── Sync best_net ─────────────────────────────────────────────────────────
    best_net.copy_weights_from(current_net)
    best_net.save(BEST_MODEL_PATH)
    print(f'\nWarm-start complete! best_net synced → {BEST_MODEL_PATH}')
    print('รัน Cell 6 (self-play loop) ได้เลยครับ')

    # ── ทดสอบด่วน: ดู value prediction ───────────────────────────────────────
    current_net.model.eval()
    sample_x = torch.from_numpy(ws_xs[:8]).to(DEVICE)
    with torch.no_grad():
        _, sample_v = current_net.model(sample_x)
    print(f'\nSample value predictions (expect ≈ targets):')
    for i in range(8):
        print(f'  target={ws_zs[i]:+.3f}  predicted={float(sample_v[i]):+.3f}')
