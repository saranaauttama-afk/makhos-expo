/**
 * battleTest.ts — AZ (ONNX) vs Minimax battle test
 *
 * Run: npx tsc --module commonjs --moduleResolution node --target es2017
 *        --outDir /tmp/battle --esModuleInterop true --skipLibCheck true
 *        scripts/battleTest.ts src/coreClaude/azFeatures.ts
 *        src/coreClaude/position.ts src/coreClaude/movegen.ts
 *        src/coreClaude/bitboards.ts src/coreClaude/eval.ts
 *      && node /tmp/battle/scripts/battleTest.js
 */

import * as ort from 'onnxruntime-node';
import * as path from 'path';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move }                from '../src/coreClaude/movegen';
import { handEvaluate }                                  from '../src/coreClaude/eval';
import { getFeatures }                                   from '../src/coreClaude/azFeatures';

const MODEL_PATH = path.join(__dirname, '../../assets/models/makhos_az.onnx');
const N_GAMES    = 20;   // games per match
const AZ_SIMS    = 3200;  // MCTS simulations per move
const MM_DEPTH   = 9;     // minimax depth

// ── ONNX inference ─────────────────────────────────────────────────────────────
let session: ort.InferenceSession | null = null;
async function getSession() {
  if (!session) session = await ort.InferenceSession.create(MODEL_PATH);
  return session;
}

async function azInfer(features: Float32Array): Promise<{ policyLogits: Float32Array; value: number }> {
  const s      = await getSession();
  const tensor = new ort.Tensor('float32', features, [1, 128]);
  const res    = await s.run({ features: tensor });
  return {
    policyLogits: res['policy_logits'].data as Float32Array,
    value:        (res['value'].data as Float32Array)[0],
  };
}

// ── MCTS ───────────────────────────────────────────────────────────────────────
const C_PUCT = 1.5;

interface Node {
  pos: Position; move: Move | null; parent: Node | null;
  children: Node[]; N: number; W: number; Q: number; P: number; expanded: boolean;
}

function makeNode(pos: Position, move: Move | null, parent: Node | null, P: number): Node {
  return { pos, move, parent, children: [], N: 0, W: 0, Q: 0, P, expanded: false };
}

function terminalResult(pos: Position, moves: Move[]): number | null {
  if (isDrawByInactivity(pos)) return 0;
  if (!moves.length)           return -1;
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
  const { policyLogits, value } = await azInfer(getFeatures(node.pos));
  const indices = node.pos.side === 1
    ? moves.map(m => m.from * 32 + m.to)
    : moves.map(m => (31 - m.from) * 32 + (31 - m.to));
  const priors = softmaxSubset(policyLogits, indices);
  for (let i = 0; i < moves.length; i++)
    node.children.push(makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]));
  return value;
}

function selectChild(node: Node): Node {
  const sqrtN = Math.sqrt(node.N);
  let best = node.children[0]; let bestScore = -Infinity;
  for (const c of node.children) {
    const s = -c.Q + C_PUCT * c.P * sqrtN / (1 + c.N);
    if (s > bestScore) { bestScore = s; best = c; }
  }
  return best;
}

function backprop(leaf: Node, v: number) {
  let curr: Node | null = leaf;
  while (curr) { curr.N++; curr.W += v; curr.Q = curr.W / curr.N; v = -v; curr = curr.parent; }
}

async function azMove(pos: Position): Promise<Move | undefined> {
  const root = makeNode(pos, null, null, 1.0);
  backprop(root, await expand(root));
  if (!root.children.length) return undefined;
  for (let i = 0; i < AZ_SIMS; i++) {
    let node = root;
    while (node.expanded && node.children.length) node = selectChild(node);
    backprop(node, node.expanded ? (terminalResult(node.pos, []) ?? 0) : await expand(node));
  }
  let best = root.children[0];
  for (const c of root.children) if (c.N > best.N) best = c;
  return best.move ?? undefined;
}

// ── Minimax ────────────────────────────────────────────────────────────────────
const INF = 99999;

function minimax(pos: Position, depth: number, alpha: number, beta: number): number {
  if (isDrawByInactivity(pos)) return 0;
  const moves = generateMoves(pos);
  if (!moves.length) return -INF;
  if (depth === 0)   return handEvaluate(pos);
  let best = -INF;
  for (const m of moves) {
    const s = -minimax(applyMove(pos, m), depth - 1, -beta, -alpha);
    if (s > best) best = s;
    if (s > alpha) alpha = s;
    if (alpha >= beta) break;
  }
  return best;
}

function minimaxMove(pos: Position): Move | undefined {
  const moves = generateMoves(pos);
  if (!moves.length) return undefined;
  let best: Move = moves[0]; let bestScore = -INF;
  for (const m of moves) {
    const s = -minimax(applyMove(pos, m), MM_DEPTH - 1, -INF, INF);
    if (s > bestScore) { bestScore = s; best = m; }
  }
  return best;
}

// ── Single game ────────────────────────────────────────────────────────────────
async function playGame(azIsP1: boolean, verbose = false): Promise<'az' | 'mm' | 'draw'> {
  let pos = initialPosition();
  for (let ply = 0; ply < 250; ply++) {
    const moves = generateMoves(pos);
    if (!moves.length) {
      const loser = pos.side === 1 ? 'p1' : 'p2';
      const azSide = azIsP1 ? 'p1' : 'p2';
      if (verbose) console.log(`  ply=${ply} no moves for ${loser}`);
      return loser === azSide ? 'mm' : 'az';
    }
    if (isDrawByInactivity(pos)) {
      if (verbose) console.log(`  ply=${ply} draw-by-inactivity (halfmoveClock=${pos.halfmoveClock})`);
      return 'draw';
    }

    const isAZTurn = (pos.side === 1) === azIsP1;
    const move = isAZTurn ? await azMove(pos) : minimaxMove(pos);
    if (!move) { if (verbose) console.log(`  ply=${ply} move=undefined`); return 'draw'; }
    if (verbose && ply < 10) console.log(`  ply=${ply} side=${pos.side} agent=${isAZTurn?'AZ':'MM'} from=${move.from}→${move.to} cap=${move.captured.length}`);
    pos = applyMove(pos, move);
  }
  if (verbose) console.log(`  ply=250 max-ply draw`);
  return 'draw';
}

// ── Main ───────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\n=== AZ (${AZ_SIMS} sims) vs Minimax-${MM_DEPTH}  [${N_GAMES} games] ===\n`);

  // warm-up
  process.stdout.write('Loading model... ');
  await getSession();
  console.log('OK\n');

  let azWins = 0, mmWins = 0, draws = 0;
  const t0 = Date.now();

  for (let g = 0; g < N_GAMES; g++) {
    const azIsP1 = g % 2 === 0;
    const result = await playGame(azIsP1);
    if (result === 'az')   { azWins++; process.stdout.write('A'); }
    else if (result === 'mm') { mmWins++; process.stdout.write('M'); }
    else                   { draws++;  process.stdout.write('D'); }
    process.stdout.write(` `);
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
  console.log(`\n\n--- Results (${elapsed}s) ---`);
  console.log(`AZ wins : ${azWins}/${N_GAMES} (${(azWins/N_GAMES*100).toFixed(0)}%)`);
  console.log(`MM wins : ${mmWins}/${N_GAMES} (${(mmWins/N_GAMES*100).toFixed(0)}%)`);
  console.log(`Draws   : ${draws}/${N_GAMES}`);
  console.log(`Win rate: ${((azWins + draws*0.5)/N_GAMES*100).toFixed(1)}%`);
}

main().catch(e => { console.error('Error:', e); process.exit(1); });
