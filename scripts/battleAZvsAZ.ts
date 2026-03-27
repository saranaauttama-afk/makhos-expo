/**
 * battleAZvsAZ.ts — NEW iter_0049 (with flip) vs OLD iter_0059 (no flip)
 */
import * as ort from 'onnxruntime-node';
import * as path from 'path';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move }                from '../src/coreClaude/movegen';
import { bits }                                          from '../src/coreClaude/bitboards';

const N_GAMES = 100;
const AZ_SIMS = 200;
const C_PUCT  = 1.0;

// ── Feature extractors ────────────────────────────────────────────────────────
function getFeaturesFlip(pos: Position): Float32Array {
  const x = new Float32Array(128);
  const myMen   = pos.side === 1 ? pos.p1Men   : pos.p2Men;
  const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen   = pos.side === 1 ? pos.p2Men   : pos.p1Men;
  const opKings = pos.side === 1 ? pos.p2Kings : pos.p1Kings;
  if (pos.side === 1) {
    for (const sq of bits(myMen))   x[sq]           = 1;
    for (const sq of bits(myKings)) x[32 + sq]      = 1;
    for (const sq of bits(opMen))   x[64 + sq]      = 1;
    for (const sq of bits(opKings)) x[96 + sq]      = 1;
  } else {
    for (const sq of bits(myMen))   x[31 - sq]      = 1;
    for (const sq of bits(myKings)) x[32 + 31 - sq] = 1;
    for (const sq of bits(opMen))   x[64 + 31 - sq] = 1;
    for (const sq of bits(opKings)) x[96 + 31 - sq] = 1;
  }
  return x;
}

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

// ── MCTS ─────────────────────────────────────────────────────────────────────
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

async function makeAZPlayer(modelPath: string, useFlip: boolean) {
  const session = await ort.InferenceSession.create(modelPath);
  const getFeatures = useFlip ? getFeaturesFlip : getFeaturesNoFlip;

  async function azInfer(features: Float32Array) {
    const t = new ort.Tensor('float32', features, [1, 128]);
    const r = await session.run({ features: t });
    return { policyLogits: r['policy_logits'].data as Float32Array, value: (r['value'].data as Float32Array)[0] };
  }

  return async function azMove(pos: Position): Promise<Move|undefined> {
    async function expand(node: Node): Promise<number> {
      node.expanded = true;
      const moves = generateMoves(node.pos);
      if (isDrawByInactivity(node.pos)) return 0;
      if (!moves.length) return -1;
      const { policyLogits, value } = await azInfer(getFeatures(node.pos));
      const indices = useFlip && node.pos.side === -1
        ? moves.map(m => (31 - m.from) * 32 + (31 - m.to))
        : moves.map(m => m.from * 32 + m.to);
      const priors = softmax(policyLogits, indices);
      for (let i = 0; i < moves.length; i++)
        node.children.push(makeNode(applyMove(node.pos, moves[i]), moves[i], node, priors[i]));
      return value;
    }
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
  };
}

async function main() {
  console.log(`\n=== NEW iter_0049 (flip) vs OLD iter_0059 (no flip)  [${N_GAMES} games] ===\n`);
  const newAZ = await makeAZPlayer(path.join(__dirname, '../../data/iter_0049.onnx'), true);
  const oldAZ = await makeAZPlayer(path.join(__dirname, '../../data/iter_0059.onnx'), false);
  console.log('Models loaded\n');

  let newW=0, oldW=0, draws=0;
  for (let g = 0; g < N_GAMES; g++) {
    const newIsP1 = g % 2 === 0;
    let pos = initialPosition();
    let result: 'new'|'old'|'draw' = 'draw';
    for (let ply = 0; ply < 250; ply++) {
      const moves = generateMoves(pos);
      if (!moves.length) {
        const loser = pos.side === 1 ? 'p1' : 'p2';
        result = loser === (newIsP1 ? 'p1' : 'p2') ? 'old' : 'new';
        break;
      }
      if (isDrawByInactivity(pos)) { result = 'draw'; break; }
      const isNew = (pos.side === 1) === newIsP1;
      const move = isNew ? await newAZ(pos) : await oldAZ(pos);
      if (!move) { result = 'draw'; break; }
      pos = applyMove(pos, move);
    }
    if (result==='new') { newW++; process.stdout.write('N'); }
    else if (result==='old') { oldW++; process.stdout.write('O'); }
    else { draws++; process.stdout.write('D'); }
    process.stdout.write(' ');
  }

  console.log(`\n\nNEW (iter_0049) wins : ${newW}/${N_GAMES}`);
  console.log(`OLD (iter_0059) wins : ${oldW}/${N_GAMES}`);
  console.log(`Draws               : ${draws}/${N_GAMES}`);
}
main().catch(e => { console.error(e); process.exit(1); });
