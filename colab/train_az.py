"""
train_az.py — AlphaZero Training Loop สำหรับ Makhos
====================================================

วิธีใช้ใน Google Colab:
  1. Upload ไฟล์ทั้งหมดใน colab/ ไปที่ Drive หรือ Colab session
  2. เปิด Runtime → Change runtime type → GPU (T4)
  3. Run all cells

Pipeline:
  iteration 1..N:
    1. Self-play → เก็บ samples ลง replay buffer
    2. Train network จาก replay buffer
    3. ทุก EVAL_EVERY iteration: ทดสอบ new vs old model
       ถ้า win_rate > WIN_THRESHOLD → update best model

ไฟล์ที่สร้าง:
  models/best.pt      — best model ล่าสุด
  models/iter_{n}.pt  — snapshot ทุก SAVE_EVERY iteration
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install deps + copy engine files
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, shutil, os

# Engine files ต้องอยู่ใน /content/ เพื่อ import ได้
DRIVE_DIR  = '/content/drive/MyDrive/makhos_az'
MODELS_DIR = f'{DRIVE_DIR}/models'
os.makedirs(MODELS_DIR, exist_ok=True)

for fname in ['makhos_engine.py', 'mcts.py', 'network.py', 'self_play.py']:
    src = f'{DRIVE_DIR}/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{fname}')
        print(f'Copied {fname}')
    else:
        print(f'WARNING: {src} not found — upload ไฟล์ไปที่ Drive ก่อน')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Config
# ─────────────────────────────────────────────────────────────────────────────
NUM_ITERATIONS  = 100    # จำนวน iteration ทั้งหมด
GAMES_PER_ITER  = 25     # self-play games per iteration
SIMULATIONS     = 200    # MCTS simulations per move
BUFFER_SIZE     = 50_000 # max samples ใน replay buffer
BATCH_SIZE      = 512
TRAIN_EPOCHS    = 5      # epochs per iteration
LR              = 1e-3
EVAL_EVERY      = 5      # ทดสอบทุกกี่ iteration
EVAL_GAMES      = 20     # เกมทดสอบต่อครั้ง
WIN_THRESHOLD   = 0.55   # win rate ต้องสูงกว่านี้ถึงจะ update best model
SAVE_EVERY      = 10     # save snapshot ทุกกี่ iteration

BEST_MODEL_PATH = f'{MODELS_DIR}/best.pt'

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Initialize
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from typing import Deque, Tuple
import random

from network   import Network
from self_play import run_self_play
from mcts      import MCTS

# Replay buffer: deque จะ pop ซ้ายอัตโนมัติเมื่อเต็ม
replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=BUFFER_SIZE)

# ─── โหลด model เดิมถ้ามี ─────────────────────────────────────────────────
current_net = Network()
best_net    = Network()

if os.path.exists(BEST_MODEL_PATH):
    current_net.load(BEST_MODEL_PATH)
    best_net.load(BEST_MODEL_PATH)
    print(f'Resumed from {BEST_MODEL_PATH}')
else:
    print('Starting from scratch (random weights)')

optimizer = optim.Adam(current_net.model.parameters(), lr=LR, weight_decay=1e-4)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
from makhos_engine import (
    initial_position, generate_moves, apply_move,
    is_terminal, is_draw_by_inactivity, bit_count,
)

def evaluate_networks(new_net: Network, old_net: Network, num_games: int = EVAL_GAMES) -> float:
    """
    ให้ new_net เล่น vs old_net แบบ greedy (temp=0).
    คืน win_rate ของ new_net (0.0–1.0).
    Draw นับเป็นครึ่ง.
    """
    mcts_new = MCTS(new_net, num_simulations=SIMULATIONS // 2)   # ลด sim เพื่อความเร็ว
    mcts_old = MCTS(old_net, num_simulations=SIMULATIONS // 2)

    new_wins = 0.0

    for g in range(num_games):
        # สลับสีทุกเกม
        new_is_p1 = (g % 2 == 0)
        pos = initial_position()

        for _ in range(200):
            if is_terminal(pos) or is_draw_by_inactivity(pos):
                break
            moves = generate_moves(pos)
            if not moves:
                break

            is_new_turn = (pos.side == 1) == new_is_p1
            mcts = mcts_new if is_new_turn else mcts_old
            move = mcts.select_move(pos, temperature=0.0)
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
# CELL 6 — Training loop
# ─────────────────────────────────────────────────────────────────────────────
print(f'Starting AlphaZero training: {NUM_ITERATIONS} iterations')
print(f'  self-play: {GAMES_PER_ITER} games × {SIMULATIONS} simulations')
print(f'  train: {TRAIN_EPOCHS} epochs, batch={BATCH_SIZE}')
print(f'  eval every {EVAL_EVERY} iters ({EVAL_GAMES} games)')
print()

for iteration in range(1, NUM_ITERATIONS + 1):
    print(f'── Iteration {iteration}/{NUM_ITERATIONS} ──────────────────────────')

    # ── 1. Self-play ──────────────────────────────────────────────────────
    print(f'  [1/3] Self-play ({GAMES_PER_ITER} games)...')
    samples = run_self_play(current_net, num_games=GAMES_PER_ITER,
                            simulations=SIMULATIONS, verbose=True)
    replay_buffer.extend(samples)
    print(f'  Buffer size: {len(replay_buffer)}/{BUFFER_SIZE}')

    # ── 2. Train ──────────────────────────────────────────────────────────
    print(f'  [2/3] Training ({TRAIN_EPOCHS} epochs, batch={BATCH_SIZE})...')
    for epoch in range(TRAIN_EPOCHS):
        batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
        xs  = np.stack([s[0] for s in batch])
        pis = np.stack([s[1] for s in batch])
        zs  = np.array([s[2] for s in batch], dtype=np.float32)

        pol_loss, val_loss = current_net.train_batch(xs, pis, zs, optimizer)

    print(f'  Loss → policy={pol_loss:.4f}  value={val_loss:.4f}')

    # ── 3. Evaluate + update best ──────────────────────────────────────────
    if iteration % EVAL_EVERY == 0:
        print(f'  [3/3] Evaluating new vs best ({EVAL_GAMES} games)...')
        win_rate = evaluate_networks(current_net, best_net)
        print(f'  Win rate: {win_rate:.1%}', end='')

        if win_rate >= WIN_THRESHOLD:
            best_net.copy_weights_from(current_net)
            best_net.save(BEST_MODEL_PATH)
            print(f'  ✓ Best model updated!')
        else:
            print(f'  (threshold={WIN_THRESHOLD:.0%}, no update)')
    else:
        print(f'  [3/3] Skipping eval (next at iter {((iteration // EVAL_EVERY) + 1) * EVAL_EVERY})')

    # ── Save snapshot ──────────────────────────────────────────────────────
    if iteration % SAVE_EVERY == 0:
        snap_path = f'{MODELS_DIR}/iter_{iteration:04d}.pt'
        current_net.save(snap_path)

    print()

print('Training complete!')
print(f'Best model saved at: {BEST_MODEL_PATH}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Export weights สำหรับ TypeScript
# ─────────────────────────────────────────────────────────────────────────────
import json, torch

best_net.load(BEST_MODEL_PATH)
m = best_net.model

def to_list(t): return t.detach().cpu().numpy().tolist()

# แปลง architecture ของ value head มาเป็น format เดิม (128→256→128→1)
# เพื่อให้ importNNWeights.ts ใช้ได้โดยไม่ต้องแก้ไฟล์ TS
# ** ใช้เฉพาะ value head สำหรับ eval ใน alpha-beta fallback **
weights = {
    'arch'  : [128, 256, 128, 1],
    'rmse'  : 0.0,   # ไม่มีค่า RMSE แบบ supervised
    'W1'    : to_list(m.fc1.weight),     # [256, 128]
    'b1'    : to_list(m.fc1.bias),       # [256]
    'W2'    : to_list(m.fc2.weight),     # [256, 256]  ← ขนาดต่างจากเดิม
    'b2'    : to_list(m.fc2.bias),       # [256]
    # value head เป็น 256→64→1 ไม่ตรง format — export แค่ policy head weights
    # สำหรับ Phase 2 เราใช้ full model (.pt) บน Colab ไม่ได้ใช้ exportNNWeights
}

out_path = f'{DRIVE_DIR}/az_model_weights.json'
with open(out_path, 'w') as f:
    json.dump(weights, f)
print(f'Weights exported → {out_path}')
print('Phase 2 complete! ใช้ best.pt เพื่อ run inference บน Colab ต่อได้เลย')
