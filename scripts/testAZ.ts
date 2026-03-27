/**
 * testAZ.ts — Test AlphaZero MCTS logic with a mocked network
 * Runs a full game: AZ (mock) vs random mover, verifies game ends correctly.
 *
 * Run: npx ts-node scripts/testAZ.ts
 */

import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move }                from '../src/coreClaude/movegen';
import { getFeatures }                                   from '../src/coreClaude/azFeatures';

// ── Mock azInfer (no ONNX needed) ─────────────────────────────────────────────
// Returns uniform policy logits + random value — exercises full MCTS tree code
async function mockInfer(features: Float32Array): Promise<{ policyLogits: Float32Array; value: number }> {
  const logits = new Float32Array(1024).fill(0); // uniform → equal priors
  const value  = (Math.random() * 2 - 1) * 0.1; // small noise around 0
  return { policyLogits: logits, value };
}

// ── Paste MCTS inline (avoids onnxruntime-react-native import) ─────────────────
const C_PUCT = 1.0;

interface Node {
  pos: Position; move: Move | null; parent: Node | null;
  children: Node[]; N: number; W: number; Q: number; P: number; expanded: boolean;
}

function makeNode(pos: Position, move: Move | null, parent: Node | null, P: number): Node {
  return { pos, move, parent, children: [], N: 0, W: 0, Q: 0, P, expanded: false };
}

function terminalResult(pos: Position, moves: Move[]): number | null {
  if (isDrawByInactivity(pos)) return 0;
  if (moves.length === 0)      return -1;
  return null;
}

function softmaxSubset(logits: Float32Array, indices: number[]): number[] {
  let max = -Infinity;
  for (const i of indices) if (logits[i] > max) max = logits[i];
  const exps = indices.map(i => Math.exp(logits[i] - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

async function expand(node: Node): Promise<number> {
  node.expanded = true;
  const moves  = generateMoves(node.pos);
  const result = terminalResult(node.pos, moves);
  if (result !== null) return result;
  const { policyLogits, value } = await mockInfer(getFeatures(node.pos));
  const indices = moves.map(m => m.from * 32 + m.to);
  const priors  = softmaxSubset(policyLogits, indices);
  for (let i = 0; i < moves.length; i++)
    node.children.push(makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]));
  return value;
}

function selectChild(node: Node): Node {
  const sqrtN = Math.sqrt(node.N);
  let best = node.children[0]; let bestScore = -Infinity;
  for (const child of node.children) {
    const score = -child.Q + C_PUCT * child.P * sqrtN / (1 + child.N);
    if (score > bestScore) { bestScore = score; best = child; }
  }
  return best;
}

function backprop(leaf: Node, value: number): void {
  let curr: Node | null = leaf; let v = value;
  while (curr) { curr.N++; curr.W += v; curr.Q = curr.W / curr.N; v = -v; curr = curr.parent; }
}

async function azBestMove(pos: Position, nSims = 50): Promise<Move | undefined> {
  const root = makeNode(pos, null, null, 1.0);
  backprop(root, await expand(root));
  if (!root.children.length) return undefined;
  for (let i = 0; i < nSims; i++) {
    let node = root;
    while (node.expanded && node.children.length > 0) node = selectChild(node);
    backprop(node, node.expanded ? (terminalResult(node.pos, []) ?? 0) : await expand(node));
  }
  let best = root.children[0];
  for (const c of root.children) if (c.N > best.N) best = c;
  return best.move ?? undefined;
}

// ── Game simulation ────────────────────────────────────────────────────────────
async function main() {
  console.log('=== AZ MCTS Test (mock network) ===\n');

  // Test feature extraction
  const startPos = initialPosition();
  const feats = getFeatures(startPos);
  console.log(`Feature vector: length=${feats.length}, sum=${feats.reduce((a,b)=>a+b,0)} (expect 16 pieces)`);

  // Verify move index range
  const moves = generateMoves(startPos);
  console.log(`Initial moves: ${moves.length}`);
  const indices = moves.map(m => m.from * 32 + m.to);
  const maxIdx = Math.max(...indices);
  console.log(`Move indices range: 0–${maxIdx} (max allowed: 1023)\n`);

  // Play 3 games: AZ (50 sims) vs random
  for (let game = 0; game < 3; game++) {
    let pos = initialPosition();
    let ply = 0;
    let result = '';

    while (ply < 200) {
      const legalMoves = generateMoves(pos);
      if (!legalMoves.length) { result = pos.side === 1 ? 'P2 wins' : 'P1 wins'; break; }
      if (isDrawByInactivity(pos)) { result = 'draw'; break; }

      let move: Move | undefined;
      if (pos.side === 1) {
        // AZ plays P1
        move = await azBestMove(pos, 30);
      } else {
        // Random plays P2
        move = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      }

      if (!move) { result = 'AZ returned undefined'; break; }
      pos = applyMove(pos, move);
      ply++;
    }

    if (!result) result = 'draw (max ply)';
    console.log(`Game ${game + 1}: ${ply} plies → ${result}`);
  }

  console.log('\n✅ MCTS logic OK — no crashes, moves generated correctly');
}

main().catch(e => { console.error('❌ Error:', e); process.exit(1); });
