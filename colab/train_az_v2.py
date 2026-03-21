"""
train_az_v2.py — AlphaZero Mixed Training (Clean Version)
==========================================================

แก้ปัญหาจาก v1:
  ✅ Mixed buffer: supervised (positions.jsonl) + self-play ผสมกันตลอด
  ✅ Value head เรียนจาก hand-crafted scores ตลอด → ไม่ลืม
  ✅ Policy head เรียนจาก MCTS ที่ guided ด้วย value ที่ดี
  ✅ best_net update ทุก EVAL_EVERY โดยไม่มี threshold ช่วงแรก
     (เปิด threshold หลัง iter 20 เมื่อ network แข็งพอ)

วิธีใช้:
  1. Upload ไฟล์ทั้ง 5 ไปที่ Drive/makhos_az/
     (makhos_engine.py, mcts.py, network.py, self_play.py, train_az_v2.py)
  2. Upload positions.jsonl ไปที่ Drive/makhos_az/
  3. Runtime → GPU (T4) → Run all cells
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Setup
# ─────────────────────────────────────────────────────────────────────────────
import os, shutil

DRIVE_DIR  = '/content/drive/MyDrive/makhos_az'
MODELS_DIR = f'{DRIVE_DIR}/models'
os.makedirs(MODELS_DIR, exist_ok=True)

for fname in ['makhos_engine.py', 'mcts.py', 'network.py', 'self_play.py']:
    src = f'{DRIVE_DIR}/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{fname}')
        print(f'  Copied {fname}')
    else:
        print(f'  WARNING: {src} not found')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Config
# ─────────────────────────────────────────────────────────────────────────────
# ── Self-play ──────────────────────────────────────────────────────────────
NUM_ITERATIONS   = 120     # ~20 ชั่วโมงบน A100
GAMES_PER_ITER   = 50      # เพิ่มความหลากหลาย
SIMULATIONS      = 800     # A100 เร็วพอ → policy targets ดีขึ้นมาก

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE       = 512     # 256 supervised + 256 self-play
TRAIN_STEPS      = 30      # train มากขึ้นต่อ iter
LR               = 3e-4    # ลดเพิ่มเติมเพื่อ stability
SUP_RATIO        = 0.5     # สัดส่วน supervised ใน batch (0.5 = 50/50)

# ── Buffer ─────────────────────────────────────────────────────────────────
SELFPLAY_BUFFER  = 100_000 # buffer ใหญ่ขึ้น

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_EVERY       = 10      # eval ทุก 10 (แต่ละ eval ช้ากว่าเพราะ sim มาก)
EVAL_GAMES       = 50      # มากขึ้น → win rate แม่นขึ้น
WIN_THRESHOLD    = 0.0     # เริ่มด้วย 0 → update ทุกครั้ง
THRESHOLD_SWITCH = 30      # เปิด 0.52 หลัง iter นี้
SAVE_EVERY       = 10

# ── Paths ──────────────────────────────────────────────────────────────────
POSITIONS_PATH   = f'{DRIVE_DIR}/positions.jsonl'
BEST_MODEL_PATH  = f'{MODELS_DIR}/best_v2.pt'

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load supervised data (positions.jsonl)
# ─────────────────────────────────────────────────────────────────────────────
import json, math
import numpy as np

print('Loading supervised data...')
sup_xs, sup_zs = [], []

with open(POSITIONS_PATH) as f:
    for line in f:
        obj = json.loads(line)
        sup_xs.append(obj['x'])
        sup_zs.append(math.tanh(obj['t'] / 400.0))

sup_xs = np.array(sup_xs, dtype=np.float32)   # (N, 128)
sup_zs = np.array(sup_zs, dtype=np.float32)   # (N,)
N_SUP  = len(sup_xs)
print(f'  {N_SUP:,} supervised positions loaded')
print(f'  z mean={sup_zs.mean():.3f}  std={sup_zs.std():.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Init network + optimizer
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

from network import Network, DEVICE

current_net = Network()
best_net    = Network()

if os.path.exists(BEST_MODEL_PATH):
    current_net.load(BEST_MODEL_PATH)
    best_net.load(BEST_MODEL_PATH)
    print(f'Resumed from {BEST_MODEL_PATH}')
else:
    print('Starting fresh (random weights)')

optimizer = optim.Adam(current_net.model.parameters(), lr=LR, weight_decay=1e-4)

# Self-play replay buffer
sp_buffer = deque(maxlen=SELFPLAY_BUFFER)

print(f'Device: {DEVICE}')
print(f'Network params: {sum(p.numel() for p in current_net.model.parameters()):,}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Mixed training step
# ─────────────────────────────────────────────────────────────────────────────
def mixed_train_step(optimizer) -> tuple:
    """
    1 gradient step ผสม supervised + self-play.
    Returns (sup_val_loss, sp_pol_loss, sp_val_loss)
    """
    current_net.model.train()
    n_sup = int(BATCH_SIZE * SUP_RATIO)
    n_sp  = BATCH_SIZE - n_sup

    # ── Supervised batch (value only) ────────────────────────────────────────
    idx_s = np.random.choice(N_SUP, n_sup, replace=False)
    x_s   = torch.from_numpy(sup_xs[idx_s]).to(DEVICE)
    z_s   = torch.from_numpy(sup_zs[idx_s]).to(DEVICE)

    _, val_s = current_net.model(x_s)
    sup_loss = F.mse_loss(val_s, z_s)

    # ── Self-play batch (policy + value) ─────────────────────────────────────
    sp_pol_loss = torch.tensor(0.0).to(DEVICE)
    sp_val_loss = torch.tensor(0.0).to(DEVICE)

    if len(sp_buffer) >= n_sp:
        batch   = random.sample(sp_buffer, n_sp)
        x_sp    = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        pi_sp   = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEVICE)
        z_sp    = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)

        pol_sp, val_sp = current_net.model(x_sp)

        log_sm      = F.log_softmax(pol_sp, dim=-1)
        sp_pol_loss = -(pi_sp * log_sm).sum(dim=-1).mean()
        sp_val_loss = F.mse_loss(val_sp, z_sp)

    # ── Combined loss ─────────────────────────────────────────────────────────
    loss = sup_loss + sp_pol_loss + sp_val_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(current_net.model.parameters(), 1.0)
    optimizer.step()

    return (
        float(sup_loss.detach()),
        float(sp_pol_loss.detach()),
        float(sp_val_loss.detach()),
    )

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
from makhos_engine import (
    initial_position, generate_moves, apply_move,
    is_terminal, is_draw_by_inactivity, bit_count,
)
from mcts import MCTS

def evaluate_networks(new_net, old_net, num_games=EVAL_GAMES) -> float:
    mcts_new = MCTS(new_net, num_simulations=SIMULATIONS // 2)
    mcts_old = MCTS(old_net, num_simulations=SIMULATIONS // 2)
    new_wins = 0.0
    for g in range(num_games):
        new_is_p1 = (g % 2 == 0)
        pos = initial_position()
        for _ in range(200):
            if is_terminal(pos) or is_draw_by_inactivity(pos):
                break
            moves = generate_moves(pos)
            if not moves:
                break
            is_new = (pos.side == 1) == new_is_p1
            move   = mcts_new.select_move(pos, 0.0) if is_new else mcts_old.select_move(pos, 0.0)
            if move is None:
                break
            pos = apply_move(pos, move)
        p1 = bit_count(pos.p1_men | pos.p1_kings)
        p2 = bit_count(pos.p2_men | pos.p2_kings)
        if p1 == p2:
            new_wins += 0.5
        elif (p1 > p2) == new_is_p1:
            new_wins += 1.0
    return new_wins / num_games

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Training loop
# ─────────────────────────────────────────────────────────────────────────────
from self_play import run_self_play

print(f'Mixed Training v2: {NUM_ITERATIONS} iterations')
print(f'  self-play: {GAMES_PER_ITER} games × {SIMULATIONS} sim')
print(f'  train: {TRAIN_STEPS} steps/iter, batch={BATCH_SIZE} ({int(SUP_RATIO*100)}% sup)')
print(f'  eval every {EVAL_EVERY} iters, threshold opens at iter {THRESHOLD_SWITCH}')
print()

for iteration in range(1, NUM_ITERATIONS + 1):
    print(f'── Iter {iteration}/{NUM_ITERATIONS} ──────────────────────────────────')

    # ── 1. Self-play ──────────────────────────────────────────────────────
    samples = run_self_play(current_net, num_games=GAMES_PER_ITER,
                            simulations=SIMULATIONS, verbose=True)
    sp_buffer.extend(samples)
    print(f'  SP buffer: {len(sp_buffer)}/{SELFPLAY_BUFFER}')

    # ── 2. Mixed training ─────────────────────────────────────────────────
    sup_losses, pol_losses, val_losses = [], [], []
    for _ in range(TRAIN_STEPS):
        sl, pl, vl = mixed_train_step(optimizer)
        sup_losses.append(sl)
        pol_losses.append(pl)
        val_losses.append(vl)

    print(f'  Loss → sup_val={np.mean(sup_losses):.4f}  '
          f'sp_pol={np.mean(pol_losses):.4f}  sp_val={np.mean(val_losses):.4f}')

    # ── 3. เปิด threshold หลัง iter THRESHOLD_SWITCH ──────────────────────
    if iteration == THRESHOLD_SWITCH:
        WIN_THRESHOLD = 0.52
        print(f'  [threshold unlocked → {WIN_THRESHOLD}]')

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    if iteration % EVAL_EVERY == 0:
        print(f'  Evaluating ({EVAL_GAMES} games)...')
        win_rate = evaluate_networks(current_net, best_net)
        print(f'  Win rate: {win_rate:.1%}', end='')

        threshold = 0.52 if iteration >= THRESHOLD_SWITCH else 0.0
        if win_rate >= threshold:
            best_net.copy_weights_from(current_net)
            best_net.save(BEST_MODEL_PATH)
            print(f'  ✅ Best updated!')
        else:
            print(f'  (no update, threshold={threshold:.0%})')
    else:
        print(f'  [eval at iter {((iteration // EVAL_EVERY) + 1) * EVAL_EVERY}]')

    # ── 5. Snapshot ───────────────────────────────────────────────────────
    if iteration % SAVE_EVERY == 0:
        current_net.save(f'{MODELS_DIR}/v2_iter_{iteration:04d}.pt')

    print()

print('Done!')
print(f'Best model: {BEST_MODEL_PATH}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Quick eval: best_v2.pt vs Minimax-3
# ─────────────────────────────────────────────────────────────────────────────
from makhos_engine import hand_eval

def minimax(pos, depth, alpha, beta):
    if depth == 0 or is_terminal(pos) or is_draw_by_inactivity(pos):
        return float(hand_eval(pos))
    moves = generate_moves(pos)
    if not moves: return -9999.0
    best = -float('inf')
    for m in moves:
        score = -minimax(apply_move(pos, m), depth-1, -beta, -alpha)
        if score > best: best = score
        alpha = max(alpha, score)
        if alpha >= beta: break
    return best

def minimax_move(pos, depth=3):
    moves = generate_moves(pos)
    if not moves: return None
    return max(moves, key=lambda m: -minimax(apply_move(pos,m), depth-1, -float('inf'), float('inf')))

best_net.load(BEST_MODEL_PATH)
mcts_eval = MCTS(best_net, num_simulations=200)

wins = 0.0
N = 30
for g in range(N):
    nn_p1 = (g % 2 == 0)
    pos   = initial_position()
    for _ in range(200):
        if is_terminal(pos) or is_draw_by_inactivity(pos): break
        mvs = generate_moves(pos)
        if not mvs: break
        is_nn = (pos.side == 1) == nn_p1
        move  = mcts_eval.select_move(pos, 0.0) if is_nn else minimax_move(pos)
        if move is None: break
        pos = apply_move(pos, move)
    p1 = bit_count(pos.p1_men | pos.p1_kings)
    p2 = bit_count(pos.p2_men | pos.p2_kings)
    if p1 == p2: wins += 0.5
    elif (p1 > p2) == nn_p1: wins += 1.0

wr = wins / N
print(f'\nFinal: MCTS+NN vs Minimax-3 = {wr:.1%}')
if wr > 0.55:
    print('✅ NN แข็งกว่า hand-crafted eval → ใช้เป็น Expert level ได้เลย!')
elif wr > 0.40:
    print('🟡 สูสี — train ต่ออีกสักรอบ')
else:
    print('❌ ยังอ่อนอยู่')
