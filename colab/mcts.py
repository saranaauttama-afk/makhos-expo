"""
mcts.py — Monte Carlo Tree Search for Makhos (AlphaZero-style)
===============================================================

UCB formula (PUCT):
  UCB(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))

Usage:
  from mcts import MCTS
  from network import Network

  net  = Network()
  mcts = MCTS(net, num_simulations=200)
  pi   = mcts.get_policy(position)   # visit-count distribution over moves
  move = mcts.select_move(position)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from makhos_engine import (
    Position, Move,
    generate_moves, apply_move,
    is_terminal, is_draw_by_inactivity,
    get_features,
)

C_PUCT = 1.5   # exploration constant (tune if needed)

# ─────────────────────────────────────────────────────────────────────────────
# Move encoding  (32 squares × 4 directions = 128 action slots)
# ─────────────────────────────────────────────────────────────────────────────

DIR_IDX = {'UL': 0, 'UR': 1, 'DL': 2, 'DR': 3}

from makhos_engine import STEPS   # STEPS[sq] = list of (to_sq, dir_name)

def move_to_action(m: Move) -> int:
    """Encode move as integer 0..127  (from_sq × 4 + dir_idx)."""
    from_sq = m.from_sq
    to_sq   = m.to_sq
    # find the direction from from_sq toward to_sq
    # for multi-step captures `to_sq` may not be adjacent — use captured chain start
    for (step_sq, dname) in STEPS[from_sq]:
        if step_sq == to_sq:
            return from_sq * 4 + DIR_IDX[dname]
    # multi-capture: find the first step direction that eventually reaches to_sq
    # we approximate with the direction of the first captured piece
    # (good enough for policy prior — exact move chosen by visit-count)
    for (step_sq, dname) in STEPS[from_sq]:
        return from_sq * 4 + DIR_IDX[dname]   # fallback: first available dir
    return from_sq * 4   # absolute fallback

def _legal_mask(moves: List[Move]) -> np.ndarray:
    """Binary mask of shape (128,) with 1 at each legal action slot."""
    mask = np.zeros(128, dtype=np.float32)
    for m in moves:
        mask[move_to_action(m)] = 1.0
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    __slots__ = ('pos', 'parent', 'move',
                 'children', 'prior',
                 'visit_count', 'value_sum',
                 'is_expanded')

    def __init__(self, pos: Position, parent: Optional['Node'], move: Optional[Move], prior: float):
        self.pos        = pos
        self.parent     = parent
        self.move       = move         # move that led here from parent
        self.children:  Dict[int, 'Node'] = {}   # action_idx → child Node
        self.prior      = prior        # P(s,a) from NN
        self.visit_count= 0
        self.value_sum  = 0.0
        self.is_expanded= False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int) -> float:
        """PUCT score."""
        exploration = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

# ─────────────────────────────────────────────────────────────────────────────
# MCTS
# ─────────────────────────────────────────────────────────────────────────────

class MCTS:
    def __init__(self, network, num_simulations: int = 200):
        """
        network : any object with a method
            policy_value(x: np.ndarray) -> (policy: np.ndarray[128], value: float)
            where x is a 128-dim feature vector.
        """
        self.network         = network
        self.num_simulations = num_simulations

    # ── Public API ────────────────────────────────────────────────────────────

    def get_policy(self, pos: Position, temperature: float = 1.0) -> Tuple[List[Move], np.ndarray]:
        """
        Run MCTS from `pos`.
        Returns (legal_moves, pi) where pi[i] is the visit-count probability
        of legal_moves[i].  Use temperature=0 for greedy (best move), 1 for
        exploration during self-play.
        """
        root = self._build_root(pos)
        for _ in range(self.num_simulations):
            self._simulate(root)

        moves = generate_moves(pos)
        visit_counts = np.array([
            root.children[move_to_action(m)].visit_count
            if move_to_action(m) in root.children else 0
            for m in moves
        ], dtype=np.float32)

        if temperature == 0:
            pi = np.zeros_like(visit_counts)
            pi[np.argmax(visit_counts)] = 1.0
        else:
            counts_t = visit_counts ** (1.0 / temperature)
            total = counts_t.sum()
            pi = counts_t / total if total > 0 else np.ones(len(moves)) / len(moves)

        return moves, pi

    def select_move(self, pos: Position, temperature: float = 0.0) -> Optional[Move]:
        """Return the best move (temperature=0 → greedy)."""
        moves, pi = self.get_policy(pos, temperature)
        if not moves:
            return None
        return moves[np.argmax(pi)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_root(self, pos: Position) -> Node:
        root = Node(pos, parent=None, move=None, prior=1.0)
        self._expand(root)
        return root

    def _simulate(self, root: Node):
        """One MCTS simulation: select → expand → backup."""
        path: List[Node] = [root]
        node = root

        # ── Select ──────────────────────────────────────────────────────────
        while node.is_expanded and node.children:
            node = self._select_child(node)
            path.append(node)

        # ── Terminal check ───────────────────────────────────────────────────
        pos = node.pos
        if is_terminal(pos) or is_draw_by_inactivity(pos):
            if is_draw_by_inactivity(pos):
                value = 0.0
            else:
                # current side has no pieces → lost
                value = -1.0
            self._backup(path, value)
            return

        # ── Expand ──────────────────────────────────────────────────────────
        value = self._expand(node)

        # ── Backup ──────────────────────────────────────────────────────────
        self._backup(path, value)

    def _select_child(self, node: Node) -> Node:
        """Pick child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        n = node.visit_count
        for child in node.children.values():
            score = child.ucb_score(n)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: Node) -> float:
        """
        Evaluate position with NN, create child nodes.
        Returns value estimate for current player (from NN).
        """
        pos   = node.pos
        moves = generate_moves(pos)

        if not moves:
            node.is_expanded = True
            return -1.0   # no moves = loss

        x = get_features(pos)
        policy_logits, value = self.network.policy_value(x)  # (128,), scalar

        # Mask illegal actions and normalize to probability
        mask = _legal_mask(moves)
        policy = np.exp(policy_logits - policy_logits.max()) * mask   # softmax + mask
        total  = policy.sum()
        if total > 0:
            policy /= total
        else:
            # all priors zero → uniform over legal moves
            policy = mask / mask.sum()

        # Add Dirichlet noise at root for exploration (only if this IS the root)
        if node.parent is None:
            alpha  = 0.3
            eps    = 0.25
            noise  = np.random.dirichlet([alpha] * int(mask.sum()))
            legal_indices = np.where(mask > 0)[0]
            for k, idx in enumerate(legal_indices):
                policy[idx] = (1 - eps) * policy[idx] + eps * noise[k]

        for m in moves:
            a = move_to_action(m)
            child_pos = apply_move(pos, m)
            node.children[a] = Node(child_pos, parent=node, move=m, prior=float(policy[a]))

        node.is_expanded = True
        return float(value)

    def _backup(self, path: List[Node], value: float):
        """
        Propagate value back up the tree.
        Value is from the perspective of the player at the LEAF node.
        Each level up we flip the sign (players alternate).
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum   += value
            value = -value   # flip for parent (opposite player)
