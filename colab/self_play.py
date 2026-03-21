"""
self_play.py — เก็บ training data จากการให้ AI เล่นกับตัวเอง
=============================================================

สิ่งที่เก็บต่อ 1 position:
  x   : 128-dim feature vector (side-to-move relative)
  pi  : 128-dim policy target  (visit-count probabilities from MCTS)
  z   : +1 / -1 / 0  (ผลเกม จากมุมมองของฝั่งที่เดิน)

Usage:
  from self_play import run_self_play
  samples = run_self_play(network, num_games=50, simulations=200)
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from makhos_engine import (
    Position, Move,
    initial_position, generate_moves, apply_move,
    is_terminal, is_draw_by_inactivity, get_features,
)
from mcts import MCTS

MAX_PLIES        = 200    # จำกัดไม่ให้เกมยาวเกินไป
TEMP_SWITCH_PLY  = 30     # ply แรกๆ ใช้ temperature=1 (explore), หลังจากนั้นใช้ temperature=0 (greedy)

@dataclass
class Sample:
    x:    np.ndarray   # (128,)
    pi:   np.ndarray   # (128,)
    side: int          # ฝั่งที่เดิน ณ ตานี้ (+1 หรือ -1)

def run_self_play(
    network,
    num_games:   int = 50,
    simulations: int = 200,
    verbose:     bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    เล่น num_games เกม, คืน list ของ (x, pi, z).
    z คือผลเกมจากมุมมองของฝั่งที่เดิน ณ ตานั้น.
    """
    mcts    = MCTS(network, num_simulations=simulations)
    all_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for g in range(num_games):
        samples = _play_one_game(mcts)
        all_samples.extend(samples)

        if verbose:
            print(f"\r  self-play game {g+1}/{num_games}  "
                  f"total samples: {len(all_samples)}", end='', flush=True)

    if verbose:
        print()

    return all_samples


def _play_one_game(mcts: MCTS) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    เล่น 1 เกม, คืน [(x, pi, z), ...] โดยที่ z ใส่ตอนจบ.
    """
    pos     = initial_position()
    history: List[Sample] = []   # เก็บ (x, pi, side) ไว้ก่อน, ค่อยใส่ z ทีหลัง
    hashes  = set()

    for ply in range(MAX_PLIES):
        # ตรวจ terminal
        if is_terminal(pos) or is_draw_by_inactivity(pos):
            break

        # ตรวจ repetition (ง่ายๆ ด้วย set ของ hash)
        h = _pos_hash(pos)
        if h in hashes:
            break   # ถือเป็น draw
        hashes.add(h)

        moves = generate_moves(pos)
        if not moves:
            break

        # temperature: explore ช่วงต้น, greedy ช่วงหลัง
        temp = 1.0 if ply < TEMP_SWITCH_PLY else 0.0

        moves_list, pi_full = mcts.get_policy(pos, temperature=temp)

        # แปลง pi_full (prob ต่อ legal move) → 128-dim array
        pi_128 = np.zeros(128, dtype=np.float32)
        from mcts import move_to_action
        for m, p in zip(moves_list, pi_full):
            pi_128[move_to_action(m)] = p

        history.append(Sample(
            x    = get_features(pos),
            pi   = pi_128,
            side = pos.side,
        ))

        # เลือก move ตาม pi_full
        move = moves_list[np.argmax(pi_full)]
        pos  = apply_move(pos, move)

    # ── กำหนดผลเกม ──────────────────────────────────────────────────────────
    result = _game_result(pos)
    # result: +1 ถ้า P1 ชนะ, -1 ถ้า P2 ชนะ, 0 เสมอ

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for s in history:
        z = float(result * s.side)   # z = +1 ถ้าฝั่งของเราชนะ
        samples.append((s.x, s.pi, z))

    return samples


def _game_result(pos: Position) -> float:
    """
    คืน +1 ถ้า P1 ชนะ, -1 ถ้า P2 ชนะ, 0 เสมอ.
    เรียกหลังเกมจบ.
    """
    from makhos_engine import bit_count
    p1 = bit_count(pos.p1_men | pos.p1_kings)
    p2 = bit_count(pos.p2_men | pos.p2_kings)

    if p1 == 0 and p2 > 0:
        return -1.0   # P2 ชนะ
    if p2 == 0 and p1 > 0:
        return 1.0    # P1 ชนะ
    return 0.0        # เสมอ (inactivity หรือ repetition)


def _pos_hash(pos: Position) -> int:
    """Hash ง่ายๆ สำหรับตรวจ repetition."""
    return hash((pos.p1_men, pos.p1_kings, pos.p2_men, pos.p2_kings, pos.side))
