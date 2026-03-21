"""
eval_best.py — ทดสอบ best.pt vs Random และ vs Minimax
======================================================

วางเป็น Cell ใหม่บน Colab แล้วรันได้เลย
ไม่ต้องหยุด training loop

ผลที่จะได้:
  - Win rate ของ MCTS+NN vs Random play
  - Win rate ของ MCTS+NN vs Simple minimax (depth 3)
"""

# ─────────────────────────────────────────────────────────────────────────────
# EVAL CELL — วางเป็น cell ใหม่บน Colab
# ─────────────────────────────────────────────────────────────────────────────
import random
import numpy as np
from makhos_engine import (
    Position, initial_position, generate_moves, apply_move,
    is_terminal, is_draw_by_inactivity, bit_count, hand_eval,
)
from mcts import MCTS
from network import Network

EVAL_GAMES   = 40    # จำนวนเกมทดสอบต่อคู่
SIMULATIONS  = 200   # MCTS simulations (ลดจาก 400 เพื่อความเร็ว)
MAX_PLIES    = 200

# ── โหลด best model ──────────────────────────────────────────────────────────
net  = Network(f'{MODELS_DIR}/best.pt')
mcts = MCTS(net, num_simulations=SIMULATIONS)
print(f'Loaded best.pt  |  SIMULATIONS={SIMULATIONS}')

# ─────────────────────────────────────────────────────────────────────────────
# ผู้เล่น Random
# ─────────────────────────────────────────────────────────────────────────────
def random_move(pos: Position):
    moves = generate_moves(pos)
    return random.choice(moves) if moves else None

# ─────────────────────────────────────────────────────────────────────────────
# ผู้เล่น Minimax depth 3 (hand-crafted eval)
# ─────────────────────────────────────────────────────────────────────────────
def minimax(pos: Position, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or is_terminal(pos) or is_draw_by_inactivity(pos):
        return float(hand_eval(pos))
    moves = generate_moves(pos)
    if not moves:
        return -9999.0
    best = -float('inf')
    for m in moves:
        child = apply_move(pos, m)
        score = -minimax(child, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best

def minimax_move(pos: Position, depth: int = 3):
    moves = generate_moves(pos)
    if not moves:
        return None
    best_move  = moves[0]
    best_score = -float('inf')
    for m in moves:
        child = apply_move(pos, m)
        score = -minimax(child, depth - 1, -float('inf'), float('inf'))
        if score > best_score:
            best_score = score
            best_move  = m
    return best_move

# ─────────────────────────────────────────────────────────────────────────────
# รัน matchup
# ─────────────────────────────────────────────────────────────────────────────
def play_match(nn_move_fn, opp_move_fn, num_games: int, label: str) -> float:
    nn_wins = 0.0
    for g in range(num_games):
        nn_is_p1 = (g % 2 == 0)
        pos = initial_position()
        for _ in range(MAX_PLIES):
            if is_terminal(pos) or is_draw_by_inactivity(pos):
                break
            moves = generate_moves(pos)
            if not moves:
                break
            is_nn = (pos.side == 1) == nn_is_p1
            move  = nn_move_fn(pos) if is_nn else opp_move_fn(pos)
            if move is None:
                break
            pos = apply_move(pos, move)

        p1 = bit_count(pos.p1_men | pos.p1_kings)
        p2 = bit_count(pos.p2_men | pos.p2_kings)
        if p1 == p2:
            nn_wins += 0.5
        elif (p1 > p2) == nn_is_p1:
            nn_wins += 1.0

        if (g + 1) % 10 == 0:
            print(f'  {label}: game {g+1}/{num_games}  nn_wins={nn_wins}')

    rate = nn_wins / num_games
    return rate

# ── NN move function ──────────────────────────────────────────────────────────
def nn_move(pos: Position):
    return mcts.select_move(pos, temperature=0.0)

# ── Test 1: MCTS+NN vs Random ─────────────────────────────────────────────────
print(f'\n[1/2] MCTS+NN (best.pt) vs Random  ({EVAL_GAMES} games)...')
wr_random = play_match(nn_move, random_move, EVAL_GAMES, 'vs-random')
print(f'  Win rate vs Random:   {wr_random:.1%}')

# ── Test 2: MCTS+NN vs Minimax depth 3 ───────────────────────────────────────
print(f'\n[2/2] MCTS+NN (best.pt) vs Minimax-3  ({EVAL_GAMES} games)...')
wr_mini = play_match(nn_move, lambda p: minimax_move(p, depth=3), EVAL_GAMES, 'vs-mini3')
print(f'  Win rate vs Minimax-3: {wr_mini:.1%}')

# ── สรุป ──────────────────────────────────────────────────────────────────────
print(f'\n{"─"*50}')
print(f'MCTS+NN (best.pt, {SIMULATIONS} sims) Results:')
print(f'  vs Random    : {wr_random:.1%}  {"✅ good" if wr_random > 0.7 else "⚠️ weak"}')
print(f'  vs Minimax-3 : {wr_mini:.1%}  {"✅ good" if wr_mini > 0.5 else "⚠️ weak"}')
print(f'{"─"*50}')
if wr_mini > 0.55:
    print('→ NN แข็งกว่า Minimax-3 ✅ ใช้เป็น Expert level ได้เลยครับ!')
elif wr_mini > 0.45:
    print('→ NN สูสีกับ Minimax-3  — ลอง train ต่ออีกก็ได้')
else:
    print('→ NN ยังอ่อนกว่า Minimax-3  — แนะนำ mixed training ครับ')
