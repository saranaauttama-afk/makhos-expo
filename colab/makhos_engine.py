"""
makhos_engine.py — Thai Checkers (Makhos) game engine in Python
================================================================

Faithful port of src/coreClaude/{bitboards,position,movegen}.ts

Board layout  (8×8, dark squares only, 32 indices)
  row 0 = top (P2 starts here),  row 7 = bottom (P1 starts here)
  dark square: (r+c) & 1 == 1   → indexed row-major 0..31

  index  0: (r=0,c=1)   index  1: (r=0,c=3)  ...  index  3: (r=0,c=7)
  index  4: (r=1,c=0)   ...                       index  7: (r=1,c=6)
  ...
  index 24: (r=6,c=1)   ...                       index 27: (r=6,c=7)
  index 28: (r=7,c=0)   ...                       index 31: (r=7,c=6)

P1 (side=1)  : starts 24–31, moves UP   (UL/UR), promotes at 0–3
P2 (side=-1) : starts  0–7,  moves DOWN (DL/DR), promotes at 28–31
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Bitboard helpers
# ─────────────────────────────────────────────────────────────────────────────

MASK32 = 0xFFFFFFFF   # keep values in uint32 range

def b1(i: int) -> int:
    """Single-bit mask for square i."""
    return (1 << i) & MASK32

def bit_count(x: int) -> int:
    return bin(x & MASK32).count('1')

def bits(bb: int):
    """Yield indices of set bits (LSB first)."""
    x = bb & MASK32
    while x:
        lsb = x & (-x & MASK32)
        yield lsb.bit_length() - 1
        x = (x ^ lsb) & MASK32

# ─────────────────────────────────────────────────────────────────────────────
# Build square maps + adjacency tables
# ─────────────────────────────────────────────────────────────────────────────

# SQUARE_TO_RC[i] = (row, col)
SQUARE_TO_RC: List[Tuple[int,int]] = []
RC_TO_INDEX = [[-1]*8 for _ in range(8)]

def _build_maps():
    idx = 0
    for r in range(8):
        for c in range(8):
            if (r + c) & 1:          # dark square
                SQUARE_TO_RC.append((r, c))
                RC_TO_INDEX[r][c] = idx
                idx += 1

_build_maps()

DIRS = [('UL', -1, -1), ('UR', -1, +1), ('DL', +1, -1), ('DR', +1, +1)]

# STEPS[i] = list of (to_sq, dir_name)   — adjacent squares (step 1)
STEPS: List[List[Tuple[int,str]]] = [[] for _ in range(32)]

# NEXT_IN_DIR[i][dir] = next square in that direction (or -1)
_dir_idx = {d: k for k, (d, *_) in enumerate(DIRS)}
NEXT_IN_DIR: List[List[int]] = [[-1]*4 for _ in range(32)]

def _build_adjacency():
    for i in range(32):
        r, c = SQUARE_TO_RC[i]
        for dk, (dname, dr, dc) in enumerate(DIRS):
            r1, c1 = r + dr, c + dc
            if 0 <= r1 < 8 and 0 <= c1 < 8:
                j = RC_TO_INDEX[r1][c1]
                if j >= 0:
                    STEPS[i].append((j, dname))
                    NEXT_IN_DIR[i][dk] = j

_build_adjacency()

def _next_in_dir(sq: int, dname: str) -> int:
    dk = _dir_idx[dname]
    return NEXT_IN_DIR[sq][dk]

def _ray(sq: int, dname: str):
    """Yield squares along direction until off-board."""
    cur = sq
    dk = _dir_idx[dname]
    while True:
        nxt = NEXT_IN_DIR[cur][dk]
        if nxt < 0:
            return
        yield nxt
        cur = nxt

# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

LAST_RANK_P1 = frozenset([0, 1, 2, 3])
LAST_RANK_P2 = frozenset([28, 29, 30, 31])
DRAW_PIECE_THRESHOLD = 2

@dataclass(frozen=True)
class Position:
    side:           int   # +1 = P1 to move, -1 = P2 to move
    p1_men:         int   # bitboard uint32
    p1_kings:       int
    p2_men:         int
    p2_kings:       int
    halfmove_clock: int   # plies since last capture; reset on capture

def initial_position() -> Position:
    p2_men = 0
    for i in range(8):
        p2_men |= b1(i)
    p1_men = 0
    for i in range(24, 32):
        p1_men |= b1(i)
    return Position(side=1, p1_men=p1_men, p1_kings=0, p2_men=p2_men, p2_kings=0, halfmove_clock=0)

def occupied(p: Position) -> int:
    return (p.p1_men | p.p1_kings | p.p2_men | p.p2_kings) & MASK32

def _side_men(p: Position)   -> int: return p.p1_men   if p.side == 1 else p.p2_men
def _side_kings(p: Position) -> int: return p.p1_kings if p.side == 1 else p.p2_kings
def _opp_men(p: Position)    -> int: return p.p2_men   if p.side == 1 else p.p1_men
def _opp_kings(p: Position)  -> int: return p.p2_kings if p.side == 1 else p.p1_kings

def is_draw_by_inactivity(p: Position) -> bool:
    p1_count = bit_count(p.p1_men | p.p1_kings)
    p2_count = bit_count(p.p2_men | p.p2_kings)
    few = (p1_count <= DRAW_PIECE_THRESHOLD) and (p2_count <= DRAW_PIECE_THRESHOLD)
    return few and p.halfmove_clock >= 20

def is_terminal(p: Position) -> bool:
    my_count  = bit_count(_side_men(p) | _side_kings(p))
    opp_count = bit_count(_opp_men(p)  | _opp_kings(p))
    return my_count == 0 or opp_count == 0

# ─────────────────────────────────────────────────────────────────────────────
# Move
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Move:
    from_sq:  int
    to_sq:    int
    captured: Tuple[int, ...]  # dark-square indices of captured pieces
    promote:  bool

# ─────────────────────────────────────────────────────────────────────────────
# applyMove
# ─────────────────────────────────────────────────────────────────────────────

def apply_move(p: Position, m: Move) -> Position:
    if p.side == 1:
        my_men, my_kings, op_men, op_kings = p.p1_men, p.p1_kings, p.p2_men, p.p2_kings
    else:
        my_men, my_kings, op_men, op_kings = p.p2_men, p.p2_kings, p.p1_men, p.p1_kings

    from_bit = b1(m.from_sq)
    to_bit   = b1(m.to_sq)

    moving_king = bool(my_kings & from_bit)

    if moving_king:
        my_kings = ((my_kings & ~from_bit) | to_bit) & MASK32
    else:
        my_men   = ((my_men   & ~from_bit) | to_bit) & MASK32

    for c in m.captured:
        cb = b1(c)
        if op_men & cb:
            op_men   = (op_men   & ~cb) & MASK32
        else:
            op_kings = (op_kings & ~cb) & MASK32

    if m.promote and not moving_king:
        my_men   = (my_men   & ~to_bit) & MASK32
        my_kings = (my_kings |  to_bit) & MASK32

    new_side  = -1 if p.side == 1 else 1
    new_clock = 0 if m.captured else p.halfmove_clock + 1

    if p.side == 1:
        return Position(new_side, my_men, my_kings, op_men, op_kings, new_clock)
    else:
        return Position(new_side, op_men, op_kings, my_men, my_kings, new_clock)

# ─────────────────────────────────────────────────────────────────────────────
# Move generation
# ─────────────────────────────────────────────────────────────────────────────

def _will_promote(side: int, to_sq: int) -> bool:
    if side == 1:
        return to_sq in LAST_RANK_P1
    else:
        return to_sq in LAST_RANK_P2

def _gen_men_captures(p: Position, from_sq: int,
                      my_men0, my_kings0, op_men0, op_kings0) -> List[Move]:
    results: List[Move] = []
    path: List[int] = []
    caps: List[int] = []

    def dfs(cur, my_men, my_kings, op_men, op_kings):
        extended = False
        for (step_sq, dname) in STEPS[cur]:
            # men forward only
            if p.side == 1  and dname in ('DL', 'DR'): continue
            if p.side == -1 and dname in ('UL', 'UR'): continue

            over = step_sq
            over_bit = b1(over)
            occ_now = (my_men | my_kings | op_men | op_kings) & MASK32
            if not ((op_men | op_kings) & over_bit):
                continue   # no enemy to capture

            landing = _next_in_dir(over, dname)
            if landing < 0:
                continue
            landing_bit = b1(landing)
            if (occ_now & landing_bit):
                continue   # landing blocked

            from_bit = b1(cur)
            captured_was_king = bool(op_kings & over_bit)

            # apply capture to temporary state
            if my_kings & from_bit:
                my_menN   = my_men
                my_kingsN = ((my_kings & ~from_bit) | landing_bit) & MASK32
            else:
                my_menN   = ((my_men   & ~from_bit) | landing_bit) & MASK32
                my_kingsN = my_kings

            if captured_was_king:
                op_menN   = op_men
                op_kingsN = (op_kings & ~over_bit) & MASK32
            else:
                op_menN   = (op_men   & ~over_bit) & MASK32
                op_kingsN = op_kings

            path.append(landing)
            caps.append(over)
            dfs(landing, my_menN, my_kingsN, op_menN, op_kingsN)
            path.pop()
            caps.pop()
            extended = True

        if not extended and caps:
            last_to = path[-1] if path else cur
            promote = _will_promote(p.side, last_to)
            results.append(Move(from_sq, last_to, tuple(caps), promote))

    dfs(from_sq, my_men0, my_kings0, op_men0, op_kings0)
    return results


def _gen_king_captures(p: Position, from_sq: int,
                       my_men0, my_kings0, op_men0, op_kings0) -> List[Move]:
    # Verify the piece is actually a king
    if not (my_kings0 & b1(from_sq)):
        return []

    results: List[Move] = []
    path: List[int] = []
    caps: List[int] = []

    def dfs(cur, my_men, my_kings, op_men, op_kings):
        extended = False

        for (dname, *_) in DIRS:
            seen_enemy = False
            enemy_idx  = -1

            for sq in _ray(cur, dname):
                bit     = b1(sq)
                occ_now = (my_men | my_kings | op_men | op_kings) & MASK32
                is_mine  = bool((my_men | my_kings) & bit)
                is_enemy = bool((op_men | op_kings) & bit)
                is_empty = not bool(occ_now & bit)

                if is_mine:
                    break   # blocked by own piece

                if not seen_enemy:
                    if is_empty:
                        continue
                    elif is_enemy:
                        seen_enemy = True
                        enemy_idx  = sq
                        continue
                    else:
                        break
                else:
                    # already found one enemy; next square must be empty (landing)
                    if not is_empty:
                        break
                    landing = sq
                    from_bit    = b1(cur)
                    landing_bit = b1(landing)
                    enemy_bit   = b1(enemy_idx)
                    captured_was_king = bool(op_kings & enemy_bit)

                    # kings always stay kings
                    my_kingsN = ((my_kings & ~from_bit) | landing_bit) & MASK32
                    my_menN   = my_men

                    if captured_was_king:
                        op_kingsN = (op_kings & ~enemy_bit) & MASK32
                        op_menN   = op_men
                    else:
                        op_menN   = (op_men   & ~enemy_bit) & MASK32
                        op_kingsN = op_kings

                    path.append(landing)
                    caps.append(enemy_idx)
                    dfs(landing, my_menN, my_kingsN, op_menN, op_kingsN)
                    path.pop()
                    caps.pop()
                    extended = True
                    break   # only immediate landing is legal

        if not extended and caps:
            last_to = path[-1] if path else cur
            results.append(Move(from_sq, last_to, tuple(caps), False))

    dfs(from_sq, my_men0, my_kings0, op_men0, op_kings0)
    return results


def generate_moves(p: Position) -> List[Move]:
    occ      = occupied(p)
    empty    = (~occ) & MASK32
    my_men   = _side_men(p)
    my_kings = _side_kings(p)

    if p.side == 1:
        my_men0, my_kings0, op_men0, op_kings0 = p.p1_men, p.p1_kings, p.p2_men, p.p2_kings
    else:
        my_men0, my_kings0, op_men0, op_kings0 = p.p2_men, p.p2_kings, p.p1_men, p.p1_kings

    captures: List[Move] = []

    for from_sq in bits(my_men):
        captures.extend(_gen_men_captures(p, from_sq, my_men0, my_kings0, op_men0, op_kings0))
    for from_sq in bits(my_kings):
        captures.extend(_gen_king_captures(p, from_sq, my_men0, my_kings0, op_men0, op_kings0))

    if captures:
        return captures

    # Quiet moves
    quiet: List[Move] = []

    for from_sq in bits(my_men):
        for (step_sq, dname) in STEPS[from_sq]:
            if p.side == 1  and dname in ('DL', 'DR'): continue
            if p.side == -1 and dname in ('UL', 'UR'): continue
            if empty & b1(step_sq):
                quiet.append(Move(from_sq, step_sq, (), _will_promote(p.side, step_sq)))

    for from_sq in bits(my_kings):
        for (dname, *_) in DIRS:
            for sq in _ray(from_sq, dname):
                if occ & b1(sq):
                    break
                quiet.append(Move(from_sq, sq, (), False))

    return quiet

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (same encoding as genTrainingData.ts / nnEval.ts)
# ─────────────────────────────────────────────────────────────────────────────

def get_features(p: Position) -> np.ndarray:
    """
    128-dim one-hot feature vector (side-to-move relative).
      x[  0.. 31] = 1 if my man at square i
      x[ 32.. 63] = 1 if my king at square i
      x[ 64.. 95] = 1 if enemy man at square i
      x[ 96..127] = 1 if enemy king at square i
    """
    x = np.zeros(128, dtype=np.float32)
    my_men   = p.p1_men   if p.side == 1 else p.p2_men
    my_kings = p.p1_kings if p.side == 1 else p.p2_kings
    op_men   = p.p2_men   if p.side == 1 else p.p1_men
    op_kings = p.p2_kings if p.side == 1 else p.p1_kings

    for sq in bits(my_men):   x[sq]       = 1.0
    for sq in bits(my_kings): x[32 + sq]  = 1.0
    for sq in bits(op_men):   x[64 + sq]  = 1.0
    for sq in bits(op_kings): x[96 + sq]  = 1.0
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Minimal hand-crafted eval (for data generation sanity checks)
# ─────────────────────────────────────────────────────────────────────────────

def hand_eval(p: Position) -> int:
    """Quick material + advancement heuristic (centipawns, side-relative)."""
    if p.side == 1:
        my_men, my_kings, op_men, op_kings = p.p1_men, p.p1_kings, p.p2_men, p.p2_kings
    else:
        my_men, my_kings, op_men, op_kings = p.p2_men, p.p2_kings, p.p1_men, p.p1_kings

    score = 100 * (bit_count(my_men) - bit_count(op_men)) \
          + 200 * (bit_count(my_kings) - bit_count(op_kings))
    return score

# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pos = initial_position()
    moves = generate_moves(pos)
    print(f"Initial position — {len(moves)} legal moves")
    for m in moves:
        print(f"  {m.from_sq} -> {m.to_sq}  promote={m.promote}  captured={m.captured}")

    # Play a random game
    import random
    p = pos
    for ply in range(200):
        mvs = generate_moves(p)
        if not mvs or is_terminal(p) or is_draw_by_inactivity(p):
            break
        p = apply_move(p, random.choice(mvs))

    print(f"\nGame ended at ply {ply+1}")
    print(f"P1 pieces: {bit_count(p.p1_men|p.p1_kings)}  P2 pieces: {bit_count(p.p2_men|p.p2_kings)}")
    print(f"Features shape: {get_features(p).shape}")
    print("Engine OK")
