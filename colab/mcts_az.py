"""
mcts_az.py — Monte Carlo Tree Search for AlphaZero (Thai Checkers)
===================================================================
PUCT selection · Dirichlet noise at root · negamax backpropagation

Backprop convention
-------------------
node.W   — accumulated value from the perspective of the player whose turn
           it is AT that node (positive = that player is winning).
node.Q   = W / N   (same perspective).
puct_score() returns score from PARENT's perspective, so it negates Q:
    score = −Q + C · P · √(parent.N) / (1 + N)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from makhos_engine import (
    Position, Move,
    generate_moves, apply_move, get_features,
    is_draw_by_inactivity, bit_count,
)
from network_az import AZNetwork, N_MOVES

C_PUCT      = 1.5
DIR_ALPHA   = 0.3    # Dirichlet concentration — smaller = flatter noise
DIR_WEIGHT  = 0.25   # noise weight at root (AlphaZero default)


# ── Terminal check ────────────────────────────────────────────────────────────

def game_result(pos: Position, moves: Optional[list] = None) -> Optional[float]:
    """Value from CURRENT player's perspective, or None if game ongoing.
       +1 = I win,  -1 = I lose,  0 = draw.
    """
    if is_draw_by_inactivity(pos):
        return 0.0

    if pos.side == 1:
        my_bb  = pos.p1_men | pos.p1_kings
        opp_bb = pos.p2_men | pos.p2_kings
    else:
        my_bb  = pos.p2_men | pos.p2_kings
        opp_bb = pos.p1_men | pos.p1_kings

    if bit_count(opp_bb) == 0:
        return 1.0   # opponent has no pieces → I win
    if bit_count(my_bb) == 0:
        return -1.0  # I have no pieces → I lose

    if moves is None:
        moves = generate_moves(pos)
    if not moves:
        return -1.0  # no legal moves → I lose

    return None      # game in progress


# ── MCTS node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ('pos', 'move', 'parent', 'children', 'N', 'W', 'P', 'expanded')

    def __init__(self, pos: Position, move: Optional[Move] = None,
                 parent: Optional['MCTSNode'] = None, prior: float = 1.0):
        self.pos      = pos
        self.move     = move      # Move that led to this node (None for root)
        self.parent   = parent
        self.children: list[MCTSNode] = []
        self.N        = 0
        self.W        = 0.0
        self.P        = prior
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self) -> float:
        """Score from PARENT's perspective (negates Q since Q is from child's view)."""
        U = C_PUCT * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return -self.Q + U


# ── Main MCTS function ────────────────────────────────────────────────────────

def mcts(root_pos: Position, network: AZNetwork,
         n_sims: int, add_noise: bool = True) -> tuple[list[Move], np.ndarray]:
    """Run MCTS from root_pos and return move policy.

    Args:
        root_pos  : current position (root of search tree)
        network   : AZNetwork for policy + value inference
        n_sims    : number of simulations (tree walks)
        add_noise : whether to add Dirichlet noise at root (True during self-play)

    Returns:
        moves      : list[Move] — legal moves from root
        visit_probs: np.ndarray shape (len(moves),) — visit-count distribution
    """
    root = MCTSNode(root_pos)

    for _ in range(n_sims):
        node = root

        # ── 1. Selection ──────────────────────────────────────────────────────
        while node.expanded and node.children:
            node = max(node.children, key=lambda c: c.puct_score())

        # ── 2. Evaluate ───────────────────────────────────────────────────────
        moves  = generate_moves(node.pos)
        result = game_result(node.pos, moves)

        if result is not None:
            value = result     # terminal node
        else:
            # ── 3. Expand ─────────────────────────────────────────────────────
            features      = get_features(node.pos)
            if node.pos.side == 1:
                legal_indices = [m.from_sq * 32 + m.to_sq for m in moves]
            else:
                legal_indices = [(31 - m.from_sq) * 32 + (31 - m.to_sq) for m in moves]
            probs, value  = network.predict(features, legal_indices)

            # Dirichlet noise at root for exploration
            if node is root and add_noise and len(moves) > 1:
                noise = np.random.dirichlet([DIR_ALPHA] * len(moves))
                probs = (1 - DIR_WEIGHT) * probs + DIR_WEIGHT * noise

            for move, prob in zip(moves, probs):
                child = MCTSNode(
                    apply_move(node.pos, move),
                    move=move, parent=node, prior=float(prob),
                )
                node.children.append(child)
            node.expanded = True

        # ── 4. Backpropagation ────────────────────────────────────────────────
        # value is from node's current player's perspective; flip at each level
        v    = value
        curr = node
        while curr is not None:
            curr.N += 1
            curr.W += v
            v    = -v
            curr = curr.parent

    # ── Policy target ─────────────────────────────────────────────────────────
    if not root.children:
        return [], np.array([])

    out_moves  = [c.move for c in root.children]
    out_visits = np.array([c.N for c in root.children], dtype=np.float64)
    out_visits /= out_visits.sum()
    return out_moves, out_visits


# ── Move index helper ─────────────────────────────────────────────────────────

def move_to_index(m: Move, side: int = 1) -> int:
    if side == 1:
        return m.from_sq * 32 + m.to_sq
    else:
        return (31 - m.from_sq) * 32 + (31 - m.to_sq)
