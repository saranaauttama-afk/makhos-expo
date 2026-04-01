// azMcts.ts — MCTS with AlphaZero network for Thai Checkers
//
// PUCT selection : -child.Q + C_PUCT * child.P * sqrt(parent.N) / (1 + child.N)
// Backprop       : v = -v going up (negamax convention)
// Move encoding  : from * 32 + to  ∈ [0, 1024)

import { applyMove, generateMoves, Move } from './movegen';
import { isDrawByInactivity, Position } from './position';
import { getFeatures } from './azFeatures';
import { azInfer } from './azNet';

const C_PUCT = 1.5;

interface Node {
  pos:      Position;
  move:     Move | null;   // move that led here (null for root)
  parent:   Node | null;
  children: Node[];
  N:        number;        // visit count
  W:        number;        // total value (current-player perspective)
  Q:        number;        // mean value = W / N
  P:        number;        // prior probability from policy head
  expanded: boolean;
}

function makeNode(pos: Position, move: Move | null, parent: Node | null, P: number): Node {
  return { pos, move, parent, children: [], N: 0, W: 0, Q: 0, P, expanded: false };
}

function terminalResult(pos: Position, moves: Move[]): number | null {
  if (isDrawByInactivity(pos)) return 0;
  if (moves.length === 0)      return -1; // current player has no moves → loses
  return null;
}

function softmaxSubset(logits: Float32Array, indices: number[]): number[] {
  let maxVal = -Infinity;
  for (const i of indices) if (logits[i] > maxVal) maxVal = logits[i];
  const exps = indices.map(i => Math.exp(logits[i] - maxVal));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

async function expand(node: Node): Promise<number> {
  node.expanded = true;
  const moves  = generateMoves(node.pos);
  const result = terminalResult(node.pos, moves);
  if (result !== null) return result;

  const { policyLogits, value } = await azInfer(getFeatures(node.pos));
  const indices = node.pos.side === 1
    ? moves.map(m => m.from * 32 + m.to)
    : moves.map(m => (31 - m.from) * 32 + (31 - m.to));
  const priors  = softmaxSubset(policyLogits, indices);

  for (let i = 0; i < moves.length; i++) {
    node.children.push(
      makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]),
    );
  }
  return value;
}

function selectChild(node: Node): Node {
  const sqrtN = Math.sqrt(node.N);
  let best = node.children[0];
  let bestScore = -Infinity;
  for (const child of node.children) {
    const score = -child.Q + C_PUCT * child.P * sqrtN / (1 + child.N);
    if (score > bestScore) { bestScore = score; best = child; }
  }
  return best;
}

function backprop(leaf: Node, value: number): void {
  let curr: Node | null = leaf;
  let v = value;
  while (curr !== null) {
    curr.N++;
    curr.W += v;
    curr.Q  = curr.W / curr.N;
    v       = -v;
    curr    = curr.parent;
  }
}

/**
 * Run AlphaZero MCTS and return the best move.
 * @param pos    current position
 * @param nSims  number of MCTS simulations (default 200)
 */
export async function azBestMove(pos: Position, nSims = 200): Promise<Move | undefined> {
  const root = makeNode(pos, null, null, 1.0);

  // Expand root
  const rootVal = await expand(root);
  backprop(root, rootVal);

  if (root.children.length === 0) return undefined;

  for (let sim = 0; sim < nSims; sim++) {
    // ── Selection ──────────────────────────────────────────────────────────
    let node = root;
    while (node.expanded && node.children.length > 0) {
      node = selectChild(node);
    }

    // ── Expansion + Evaluation ─────────────────────────────────────────────
    let value: number;
    if (!node.expanded) {
      value = await expand(node);
    } else {
      // Terminal node (expanded but no children)
      value = terminalResult(node.pos, []) ?? 0;
    }

    // ── Backpropagation ────────────────────────────────────────────────────
    backprop(node, value);
  }

  // Return move with most visits
  let best = root.children[0];
  for (const child of root.children) {
    if (child.N > best.N) best = child;
  }
  return best.move ?? undefined;
}
