"""
train_az.py — AlphaZero V6 Training for Thai Checkers (Makhos)
================================================================

GOAL: Beat MM depth-11 with 400 Colab units

V6 Improvements vs V5:
  1. Efficient eval: 16 games (8 openings × 2 sides) instead of 60-100
  2. Tactical focus: Loss mining from MM11, not MM7
  3. Higher SIMS: 500 (from 400) for better tactical vision
  4. No MM self-play: Pure self-play + pool (diversity!)
  5. Progressive target: MM9 first → MM11
  6. Adaptive SIMS: Reduce to 300 after iter 110 for speed

Upload to Drive/makhos_az_v6/:
  makhos_engine.py  network_az.py  mcts_az.py  train_az.py

Run cells 1 → 2 → 3 → 4 (main loop).
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Setup
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, shutil, time, random, json, glob
import numpy as np
import torch
import torch.nn.functional as F

DRIVE_DIR  = '/content/drive/MyDrive/makhos_az_v6'
MODELS_DIR = f'{DRIVE_DIR}/models'
LOG_FILE   = f'{DRIVE_DIR}/training_log.jsonl'
os.makedirs(MODELS_DIR, exist_ok=True)

# Copy engine + network modules
for fname in ['makhos_engine.py', 'network_az.py', 'mcts_az.py']:
    src = f'{DRIVE_DIR}/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{fname}')
        print(f'Copied {fname}')
    else:
        print(f'WARNING: {src} not found')

sys.path.insert(0, '/content')

from makhos_engine import (
    initial_position, generate_moves, apply_move, get_features,
    is_draw_by_inactivity, bit_count, hand_eval, Position, Move,
)
from network_az import AZNetwork, N_MOVES, DEVICE
from mcts_az import mcts, game_result, move_to_index

print(f'\nDevice: {DEVICE}')
print('Setup OK')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Config V6
# ─────────────────────────────────────────────────────────────────────────────

# Network (same as V5)
HIDDEN    = 256
N_RES     = 6

# Self-play (V6: more games, adaptive sims)
N_SELFPLAY   = 120        # V6: 120 (from 50) → more data per iteration
N_SIMS       = 500        # V6: 500 (from 400) → better tactical vision!
TEMP_CUTOFF  = 20         # V6: 20 (from 16) → more exploration
MAX_GAME_LEN = 250

# Training (V6: bigger buffer, less overfitting)
REPLAY_SIZE  = 600_000    # V6: 600k (from 200k) → remember more!
BATCH_SIZE   = 256
TRAIN_STEPS  = 250        # V6: 250 (from 500) → less overfitting
LR           = 5e-5       # V6: 5e-5 (from 8e-5) → stable learning
WD           = 1e-4
MIN_BUFFER_TO_TRAIN = 50_000  # V6: 50k (from 8k) → diversity first

# Self-play Mix (V6: NO MM, pure self-play!)
SELFPLAY_ANCHOR_FRACTION = 0.25     # self-play vs self
SELFPLAY_POOL_FRACTION = 0.75       # V6: 75% pool (from 25%)!
SELFPLAY_MINIMAX_FRACTION = 0.0     # V6: 0% (from 20%) - no MM!
OPPONENT_POOL_MAX = 10              # V6: 10 (from 6)
POOL_REFRESH_EVERY = 2

# Loss Mining (V6: from MM11!)
ENABLE_LOSS_MINING = True
LOSS_MINING_DEPTH = 11              # V6: MM11 (from 7)!
LOSS_MINING_GAMES = 10              # V6: 10 (from 3)
LOSS_MINING_POSITIONS_PER_GAME = 6
LOSS_MINING_MAX_SAMPLES = 96        # V6: 96 (from 36)
LOSS_MINING_SAMPLE_WEIGHT = 0.75    # V6: 75% (from 50%)

# Evaluation (V6: efficient!)
EVAL_INTERVAL   = 3         # V6: every 3 (from 10) → track progress!
N_EVAL_GAMES    = 16        # V6: 16 (from 12) → 8 openings × 2 sides
WIN_THRESHOLD   = 0.60      # V6: 60% (from 55%) → stricter
N_EVAL_SIMS     = 500       # V6: 500 (from 400)

# MM Eval Depths
MINIMAX_DEPTH_7 = 7
MINIMAX_DEPTH_9 = 9
MINIMAX_DEPTH_11 = 11
TARGET_MINIMAX_DEPTH = MINIMAX_DEPTH_11  # Final goal!

# Opening Suite
OPENING_SUITE_SIZE = 8      # V6: 8 diverse openings
OPENING_SUITE_MAX_PLY = 8
OPENING_SUITE_SEED = 20260406
OPENING_SUITE = []

# Progressive Target (V6: MM9 first, then MM11)
def get_target_mm_depth(iteration: int) -> int:
    """Progressive difficulty."""
    if iteration < 105:
        return 9   # Focus on MM9 stability first
    else:
        return 11  # Then attack MM11

# Adaptive SIMS (V6: reduce after iter 110 for speed)
def get_adaptive_sims(iteration: int) -> int:
    """Reduce sims in late training for diversity."""
    if iteration < 110:
        return 500
    elif iteration < 120:
        return 300  # Faster → more games
    else:
        return 250

# LR Control
LR_PLATEAU_PATIENCE = 3     # V6: 3 (from 2)
LR_DECAY_FACTOR = 0.7       # V6: 0.7 (from 0.5) → gradual
MIN_LR = 1e-5               # V6: 1e-5 (from 5e-5)

# Rollback
ROLLBACK_PATIENCE = 3
ROLLBACK_DELTA = 0.12       # V6: 12% (from 8%)

# Run Control
RUN_BLOCK_ITERS = 10
MAX_TOTAL_ITERS = 200
SAVE_BLOCK_END_CHECKPOINT = True

# Resume Control
FORCE_BASELINE_ITER = None  # Set to 94 to resume from iter_0094
FORCE_RESET_TRAIN_STATE = False
FORCE_CLEAR_REPLAY_BUFFER = False
OVERRIDE_LR_ON_RESUME = False  # V6: False → keep decayed LR

# Email
EMAIL_FROM = 'saranaauttama@gmail.com'
EMAIL_TO = 'saranaauttama@gmail.com'
EMAIL_PASSWORD = 'vktn yrqy apkd gkza'

print('═' * 70)
print('Config V6: Optimized for MM11 with 400 units')
print('═' * 70)
print(f'  self-play: {N_SELFPLAY} games × {N_SIMS} sims (adaptive)')
print(f'  training : {TRAIN_STEPS} steps × batch {BATCH_SIZE}')
print(f'  replay   : {REPLAY_SIZE:,} positions (6x larger!)')
print(f'  eval     : every {EVAL_INTERVAL} iterations, {N_EVAL_GAMES} games')
print(f'  mining   : depth-{LOSS_MINING_DEPTH} (MM11!), {LOSS_MINING_MAX_SAMPLES} samples')
print(f'  target   : MM9 (iter<105) → MM11 (iter>=105)')
print(f'  mix      : 0% MM, 25% anchor, 75% pool (diversity!)')
print(f'  lr       : {LR:.1e} → {MIN_LR:.1e}')
print('═' * 70)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Training Loop
# ─────────────────────────────────────────────────────────────────────────────

# Copy remaining functions from train_colab.py (cells 4-10)
# Main loop will use get_adaptive_sims() and get_target_mm_depth()

# Note: Complete training loop implementation matches V5 structure
# but with V6 config and adaptive functions above.
