/**
 * battleTestOld.ts — Test OLD model (iter_0059, no flip) vs Minimax-5
 */
import * as ort from 'onnxruntime-node';
import * as path from 'path';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move }                from '../src/coreClaude/movegen';
import { handEvaluate }                                  from '../src/coreClaude/eval';
import { bits }                                          from '../src/coreClaude/bitboards';

const MODEL_PATH = path.join(__dirname, '../../data_old/iter_0059.onnx');
const N_GAMES    = 20;
const AZ_SIMS    = 200;
const MM_DEPTH   = 5;

let session: ort.InferenceSession | null = null;
async function getSession() {
  if (!session) session = await ort.InferenceSession.create(MODEL_PATH);
  return session;
}

// NO flip — old model was trained without flip
function getFeaturesNoFlip(pos: Position): Float32Array {
  const x = new Float32Array(128);
  const myMen   = pos.side === 1 ? pos.p1Men   : pos.p2Men;
  const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen   = pos.side === 1 ? pos.p2Men   : pos.p1Men;
  const opKings = pos.side === 1 ? pos.p2Kings : pos.p1Kings;
  for (const sq of bits(myMen))   x[sq]      = 1;
  for (const sq of bits(myKings)) x[32 + sq] = 1;
  for (const sq of bits(opMen))   x[64 + sq] = 1;
  for (const sq of bits(opKings)) x[96 + sq] = 1;
  return x;
}

async function azInfer(features: Float32Array) {
  const s = await getSession();
  const t = new ort.Tensor('float32', features, [1, 128]);
  const r = await s.run({ features: t });
  return { policyLogits: r['policy_logits'].data as Float32Array, value: (r['value'].data as Float32Array)[0] };
}

const C_PUCT = 1.0;
interface Node { pos: Position; move: Move|null; parent: Node|null; children: Node[]; N:number; W:number; Q:number; P:number; expanded:boolean; }
function makeNode(pos: Position, move: Move|null, parent: Node|null, P: number): Node {
  return { pos, move, parent, children:[], N:0, W:0, Q:0, P, expanded:false };
}
function softmax(logits: Float32Array, indices: number[]): number[] {
  let max = -Infinity; for (const i of indices) if (logits[i] > max) max = logits[i];
  const e = indices.map(i => Math.exp(logits[i] - max)); const s = e.reduce((a,b)=>a+b,0);
  return e.map(x => x/s);
}
function bp(leaf: Node, v: number) { let c: Node|null = leaf; while(c){c.N++;c.W+=v;c.Q=c.W/c.N;v=-v;c=c.parent;} }
async function expand(node: Node): Promise<number> {
  node.expanded = true;
  const moves = generateMoves(node.pos);
  if (isDrawByInactivity(node.pos)) return 0;
  if (!moves.length) return -1;
  const { policyLogits, value } = await azInfer(getFeaturesNoFlip(node.pos));
  // NO flip for old model
  const indices = moves.map(m => m.from * 32 + m.to);
  const priors = softmax(policyLogits, indices);
  for (let i = 0; i < moves.length; i++)
    node.children.push(makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]));
  return value;
}
async function azMove(pos: Position): Promise<Move|undefined> {
  const root = makeNode(pos, null, null, 1.0);
  bp(root, await expand(root));
  if (!root.children.length) return undefined;
  for (let i = 0; i < AZ_SIMS; i++) {
    let node = root;
    while (node.expanded && node.children.length) {
      const sqN = Math.sqrt(node.N); let best = node.children[0], bs = -Infinity;
      for (const c of node.children) { const s = -c.Q + C_PUCT*c.P*sqN/(1+c.N); if(s>bs){bs=s;best=c;} }
      node = best;
    }
    bp(node, node.expanded ? -1 : await expand(node));
  }
  let best = root.children[0]; for (const c of root.children) if(c.N>best.N) best=c;
  return best.move ?? undefined;
}

const INF = 99999;
function minimax(pos: Position, depth: number, alpha: number, beta: number): number {
  if (isDrawByInactivity(pos)) return 0;
  const moves = generateMoves(pos);
  if (!moves.length) return -INF;
  if (depth === 0) return handEvaluate(pos);
  let best = -INF;
  for (const m of moves) {
    const s = -minimax(applyMove(pos, m), depth-1, -beta, -alpha);
    if (s > best) best = s;
    if (s > alpha) alpha = s;
    if (alpha >= beta) break;
  }
  return best;
}
function mmMove(pos: Position): Move|undefined {
  const moves = generateMoves(pos);
  if (!moves.length) return undefined;
  let best = moves[0], bs = -INF;
  for (const m of moves) { const s = -minimax(applyMove(pos,m),MM_DEPTH-1,-INF,INF); if(s>bs){bs=s;best=m;} }
  return best;
}

async function playGame(azIsP1: boolean): Promise<'az'|'mm'|'draw'> {
  let pos = initialPosition();
  for (let ply = 0; ply < 250; ply++) {
    const moves = generateMoves(pos);
    if (!moves.length) { const loser = pos.side===1?'p1':'p2'; return loser===(azIsP1?'p1':'p2')?'mm':'az'; }
    if (isDrawByInactivity(pos)) return 'draw';
    const isAZ = (pos.side===1) === azIsP1;
    const move = isAZ ? await azMove(pos) : mmMove(pos);
    if (!move) return 'draw';
    pos = applyMove(pos, move);
  }
  return 'draw';
}

async function main() {
  console.log(`\n=== OLD model iter_0059 (no flip) vs Minimax-${MM_DEPTH}  [${N_GAMES} games] ===\n`);
  await getSession(); console.log('Model loaded\n');
  let azW=0, mmW=0, draws=0;
  for (let g = 0; g < N_GAMES; g++) {
    const azIsP1 = g%2===0;
    const r = await playGame(azIsP1);
    if(r==='az'){azW++;process.stdout.write('A');}
    else if(r==='mm'){mmW++;process.stdout.write('M');}
    else{draws++;process.stdout.write('D');}
    process.stdout.write(' ');
  }
  console.log(`\n\nAZ wins : ${azW}/${N_GAMES}`);
  console.log(`MM wins : ${mmW}/${N_GAMES}`);
  console.log(`Draws   : ${draws}/${N_GAMES}`);
  console.log(`Win rate: ${((azW+draws*0.5)/N_GAMES*100).toFixed(1)}%`);
}
main().catch(e => { console.error(e); process.exit(1); });
