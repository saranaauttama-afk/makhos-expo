import * as ort from 'onnxruntime-node';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { handEvaluate } from '../src/coreClaude/eval';
import { getFeatures } from '../src/coreClaude/azFeatures';

type CliConfig = {
  model: string;
  games: number;
  depth: number;
  sims: number;
};

function parseArgs(): CliConfig {
  const args = process.argv.slice(2);
  const cfg: CliConfig = {
    model: '',
    games: 12,
    depth: 11,
    sims: 1600,
  };
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i];
    const value = args[i + 1];
    if (key === '--model') cfg.model = value;
    if (key === '--games') cfg.games = Number(value);
    if (key === '--depth') cfg.depth = Number(value);
    if (key === '--sims') cfg.sims = Number(value);
  }
  if (!cfg.model) throw new Error('Missing --model path');
  return cfg;
}

const config = parseArgs();
const MODEL_PATH = config.model;
const N_GAMES = config.games;
const AZ_SIMS = config.sims;
const MM_DEPTH = config.depth;
const C_PUCT = 1.5;
const INF = 99999;

let session: ort.InferenceSession | null = null;

async function getSession() {
  if (!session) session = await ort.InferenceSession.create(MODEL_PATH);
  return session;
}

async function azInfer(features: Float32Array): Promise<{ policyLogits: Float32Array; value: number }> {
  const s = await getSession();
  const tensor = new ort.Tensor('float32', features, [1, 128]);
  const res = await s.run({ features: tensor });
  return {
    policyLogits: res.policy_logits.data as Float32Array,
    value: (res.value.data as Float32Array)[0],
  };
}

interface Node {
  pos: Position;
  move: Move | null;
  parent: Node | null;
  children: Node[];
  N: number;
  W: number;
  Q: number;
  P: number;
  expanded: boolean;
}

function makeNode(pos: Position, move: Move | null, parent: Node | null, prior: number): Node {
  return { pos, move, parent, children: [], N: 0, W: 0, Q: 0, P: prior, expanded: false };
}

function terminalResult(pos: Position, moves: Move[]): number | null {
  if (isDrawByInactivity(pos)) return 0;
  if (!moves.length) return -1;
  return null;
}

function softmaxSubset(logits: Float32Array, indices: number[]): number[] {
  let max = -Infinity;
  for (const index of indices) if (logits[index] > max) max = logits[index];
  const exps = indices.map(index => Math.exp(logits[index] - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function expand(node: Node): Promise<number> {
  node.expanded = true;
  const moves = generateMoves(node.pos);
  const result = terminalResult(node.pos, moves);
  if (result !== null) return result;

  const { policyLogits, value } = await azInfer(getFeatures(node.pos));
  const indices = node.pos.side === 1
    ? moves.map(move => move.from * 32 + move.to)
    : moves.map(move => (31 - move.from) * 32 + (31 - move.to));
  const priors = softmaxSubset(policyLogits, indices);
  for (let i = 0; i < moves.length; i += 1) {
    node.children.push(makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]));
  }
  return value;
}

function selectChild(node: Node): Node {
  const sqrtN = Math.sqrt(node.N);
  let best = node.children[0];
  let bestScore = -Infinity;
  for (const child of node.children) {
    const score = -child.Q + C_PUCT * child.P * sqrtN / (1 + child.N);
    if (score > bestScore) {
      best = child;
      bestScore = score;
    }
  }
  return best;
}

function backprop(leaf: Node, value: number) {
  let current: Node | null = leaf;
  let v = value;
  while (current) {
    current.N += 1;
    current.W += v;
    current.Q = current.W / current.N;
    v = -v;
    current = current.parent;
  }
}

async function azMove(pos: Position): Promise<Move | undefined> {
  const root = makeNode(pos, null, null, 1.0);
  backprop(root, await expand(root));
  if (!root.children.length) return undefined;
  for (let i = 0; i < AZ_SIMS; i += 1) {
    let node = root;
    while (node.expanded && node.children.length) node = selectChild(node);
    backprop(node, node.expanded ? (terminalResult(node.pos, []) ?? 0) : await expand(node));
  }
  let best = root.children[0];
  for (const child of root.children) if (child.N > best.N) best = child;
  return best.move ?? undefined;
}

function minimax(pos: Position, depth: number, alpha: number, beta: number): number {
  if (isDrawByInactivity(pos)) return 0;
  const moves = generateMoves(pos);
  if (!moves.length) return -INF;
  if (depth === 0) return handEvaluate(pos);
  let best = -INF;
  for (const move of moves) {
    const score = -minimax(applyMove(pos, move), depth - 1, -beta, -alpha);
    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }
  return best;
}

function minimaxMove(pos: Position): Move | undefined {
  const moves = generateMoves(pos);
  if (!moves.length) return undefined;
  let best = moves[0];
  let bestScore = -INF;
  for (const move of moves) {
    const score = -minimax(applyMove(pos, move), MM_DEPTH - 1, -INF, INF);
    if (score > bestScore) {
      best = move;
      bestScore = score;
    }
  }
  return best;
}

async function playGame(azIsP1: boolean): Promise<'az' | 'mm' | 'draw'> {
  let pos = initialPosition();
  for (let ply = 0; ply < 250; ply += 1) {
    const moves = generateMoves(pos);
    if (!moves.length) {
      const loser = pos.side === 1 ? 'p1' : 'p2';
      const azSide = azIsP1 ? 'p1' : 'p2';
      return loser === azSide ? 'mm' : 'az';
    }
    if (isDrawByInactivity(pos)) return 'draw';

    const isAzTurn = (pos.side === 1) === azIsP1;
    const move = isAzTurn ? await azMove(pos) : minimaxMove(pos);
    if (!move) return 'draw';
    pos = applyMove(pos, move);
  }
  return 'draw';
}

async function main() {
  await getSession();
  let azWins = 0;
  let mmWins = 0;
  let draws = 0;

  for (let g = 0; g < N_GAMES; g += 1) {
    const azIsP1 = g % 2 === 0;
    const result = await playGame(azIsP1);
    if (result === 'az') azWins += 1;
    else if (result === 'mm') mmWins += 1;
    else draws += 1;
  }

  const winRate = (azWins + draws * 0.5) / N_GAMES;
  console.log(`AZ wins : ${azWins}/${N_GAMES}`);
  console.log(`MM wins : ${mmWins}/${N_GAMES}`);
  console.log(`Draws   : ${draws}/${N_GAMES}`);
  console.log(`Win rate: ${(winRate * 100).toFixed(1)}%`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
