// scripts/trainNN.ts
//
// Train a neural network position evaluator via DISTILLATION:
//   teacher = hand-crafted evaluate() function (continuous centipawn signal)
//   student = small MLP (32→24→1)
//
// This is better than game-outcome training because:
//   - Target is continuous ±2000cp, not sparse {-500, 0, +500}
//   - Much more informative signal per position
//   - Network learns a smooth approximation → better search guidance
//
// Usage:
//   npx tsx --tsconfig tsconfig.test.json scripts/trainNN.ts
//
// After completion, paste the printed block into src/coreClaude/nnWeights.ts.
// IMPORTANT: NN_TRAINED must be false in nnWeights.ts when running this script.

import { applyMove, generateMoves }         from '../src/coreClaude/movegen';
import { bits }                              from '../src/coreClaude/bitboards';
import { initialPosition, isDrawByInactivity, Position }
                                             from '../src/coreClaude/position';
import { evaluate }                          from '../src/coreClaude/eval';
import { iterativeDeepening }                from '../src/coreClaude/search/alphabeta';
import { TT }                                from '../src/coreClaude/search/tt';
import { buildRepetitionCounts, isThreefoldRepetition }
                                             from '../src/coreClaude/search/repetition';
import { hashPosition }                      from '../src/coreClaude/search/zobrist';
import { NN_TRAINED }                        from '../src/coreClaude/nnWeights';

// Safety check: NN must be off so evaluate() returns hand-crafted scores
if (NN_TRAINED) {
  console.error('ERROR: Set NN_TRAINED = false in nnWeights.ts before running this script.');
  process.exit(1);
}

// ── Config ────────────────────────────────────────────────────────────────────
const NUM_GAMES   = 300;  // games to play for position diversity
const THINK_MS    = 150;  // ms/move during data gen (fast — just for variety)
const ORACLE_MS   = 800;  // ms for oracle search (deeper = stronger teacher signal)
const MAX_PLIES   = 120;  // max plies per game
const CLIP        = 900;  // clip teacher scores to ±CLIP cp (ignore blown-out positions)
const LR          = 0.001;
const EPOCHS      = 500;
const BATCH       = 128;

// ── Network dimensions ────────────────────────────────────────────────────────
const IN = 32;
const H  = 24;

// ── Weight initialisation (Xavier uniform) ───────────────────────────────────
function xavierUniform(size: number, fanIn: number): Float32Array {
  const a = Math.sqrt(6 / fanIn);
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) w[i] = (Math.random() * 2 - 1) * a;
  return w;
}

let W1 = xavierUniform(H * IN, IN);
let b1 = new Float32Array(H);            // zero-init biases
let W2 = xavierUniform(H, H);
let b2 = new Float32Array(1);

// Adam buffers
const mW1 = new Float32Array(W1.length); const vW1 = new Float32Array(W1.length);
const mb1 = new Float32Array(H);         const vb1 = new Float32Array(H);
const mW2 = new Float32Array(H);         const vW2 = new Float32Array(H);
const mb2 = new Float32Array(1);         const vb2 = new Float32Array(1);
let adamT = 0;

// ── Adam update ───────────────────────────────────────────────────────────────
const B1_ = 0.9, B2_ = 0.999, EPS = 1e-8;
function adamStep(p: Float32Array, g: Float32Array, m: Float32Array, v: Float32Array) {
  const b1c = 1 - Math.pow(B1_, adamT);
  const b2c = 1 - Math.pow(B2_, adamT);
  for (let i = 0; i < p.length; i++) {
    m[i] = B1_ * m[i] + (1 - B1_) * g[i];
    v[i] = B2_ * v[i] + (1 - B2_) * g[i] * g[i];
    p[i] -= LR * (m[i] / b1c) / (Math.sqrt(v[i] / b2c) + EPS);
  }
}

// ── Train one batch (backprop) ────────────────────────────────────────────────
function trainBatch(xs: Float32Array[], ts: number[]): number {
  const n   = xs.length;
  const gW1 = new Float32Array(W1.length);
  const gb1 = new Float32Array(H);
  const gW2 = new Float32Array(H);
  const gb2 = new Float32Array(1);
  let loss = 0;

  for (let s = 0; s < n; s++) {
    const x = xs[s];
    const t = ts[s];

    // Forward pass
    const h = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      let acc = b1[i];
      const row = i * IN;
      for (let j = 0; j < IN; j++) acc += W1[row + j] * x[j];
      h[i] = acc > 0 ? acc : 0;
    }
    let y = b2[0];
    for (let j = 0; j < H; j++) y += W2[j] * h[j];

    const err = y - t;
    loss += err * err;

    // Backprop
    const dY = (2 * err) / n;
    for (let j = 0; j < H; j++) gW2[j] += dY * h[j];
    gb2[0] += dY;

    for (let j = 0; j < H; j++) {
      const dH = dY * W2[j] * (h[j] > 0 ? 1 : 0);
      gb1[j] += dH;
      const row = j * IN;
      for (let k = 0; k < IN; k++) gW1[row + k] += dH * x[k];
    }
  }

  adamT++;
  adamStep(W1, gW1, mW1, vW1);
  adamStep(b1, gb1, mb1, vb1);
  adamStep(W2, gW2, mW2, vW2);
  adamStep(b2, gb2, mb2, vb2);

  return loss / n;
}

// ── Feature extraction (side-to-move relative) ───────────────────────────────
function getFeatures(pos: Position): Float32Array {
  const f    = new Float32Array(IN);
  const side = pos.side;
  for (const sq of bits(side === 1 ? pos.p1Men   : pos.p2Men))   f[sq] =  1;
  for (const sq of bits(side === 1 ? pos.p1Kings : pos.p2Kings)) f[sq] =  3;
  for (const sq of bits(side === 1 ? pos.p2Men   : pos.p1Men))   f[sq] = -1;
  for (const sq of bits(side === 1 ? pos.p2Kings : pos.p1Kings)) f[sq] = -3;
  return f;
}

// ── Shuffle in place ──────────────────────────────────────────────────────────
function shuffle<T>(arr: T[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ── Generate positions via self-play ─────────────────────────────────────────
type Sample = { x: Float32Array; t: number };

async function collectGame(samples: Sample[]): Promise<void> {
  const hashes: number[] = [];
  let pos = initialPosition();
  const tt = new TT(1 << 16);

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const moves = generateMoves(pos);
    if (!moves.length) break;

    const hash = hashPosition(pos);
    hashes.push(hash);
    if (isThreefoldRepetition(buildRepetitionCounts(hashes)) || isDrawByInactivity(pos)) break;

    // Move: random for first 3 plies to diversify openings, engine after
    let move;
    if (ply < 3) {
      move = moves[Math.floor(Math.random() * moves.length)];
    } else {
      const res = await iterativeDeepening(pos, THINK_MS, tt, undefined, hashes);
      if (!res.best) break;
      move = res.best;
    }

    // Teacher score: use deeper oracle search for stronger signal
    // (alphabeta at ORACLE_MS sees further than leaf eval alone)
    const oracleTT  = new TT(1 << 16);
    const oracle    = await iterativeDeepening(pos, ORACLE_MS, oracleTT, undefined, hashes);
    const score     = oracle.score ?? evaluate(pos);

    // Skip blown-out positions (already decided, not useful for learning)
    if (Math.abs(score) <= CLIP) {
      samples.push({ x: getFeatures(pos), t: score });
    }
    pos = applyMove(pos, move);
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
(async () => {
  // ── Step 1: Generate training positions ──────────────────────────────────
  console.log(`[1/3] Generating positions (${NUM_GAMES} games, teacher = hand-crafted eval)...`);
  const samples: Sample[] = [];

  for (let g = 0; g < NUM_GAMES; g++) {
    await collectGame(samples);
    if ((g + 1) % 20 === 0 || g === NUM_GAMES - 1) {
      process.stdout.write(`\r  game ${g + 1}/${NUM_GAMES}  positions: ${samples.length}`);
    }
  }
  console.log(`\n  Done — ${samples.length} positions (clipped to ±${CLIP}cp)`);

  const xs = samples.map(s => s.x);
  const ts = samples.map(s => s.t);

  // Target stats
  const tMean = ts.reduce((a, b) => a + b, 0) / ts.length;
  const tStd  = Math.sqrt(ts.reduce((a, v) => a + (v - tMean) ** 2, 0) / ts.length);
  console.log(`  Target: mean=${tMean.toFixed(1)}cp  std=${tStd.toFixed(1)}cp`);

  // ── Step 2: Train ─────────────────────────────────────────────────────────
  console.log(`\n[2/3] Training (${EPOCHS} epochs, batch=${BATCH}, lr=${LR})...`);

  for (let ep = 0; ep < EPOCHS; ep++) {
    shuffle(samples);
    const xs2 = samples.map(s => s.x);
    const ts2  = samples.map(s => s.t);

    let epochLoss = 0, batches = 0;
    for (let i = 0; i < xs2.length; i += BATCH) {
      epochLoss += trainBatch(xs2.slice(i, i + BATCH), ts2.slice(i, i + BATCH));
      batches++;
    }

    if ((ep + 1) % 100 === 0 || ep === EPOCHS - 1) {
      const rmse = Math.sqrt(epochLoss / batches);
      process.stdout.write(`\r  epoch ${ep + 1}/${EPOCHS}  RMSE=${rmse.toFixed(1)}cp`);
    }
  }
  console.log();

  // ── Step 3: Evaluate ──────────────────────────────────────────────────────
  let mse = 0;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const h = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      let acc = b1[j];
      const row = j * IN;
      for (let k = 0; k < IN; k++) acc += W1[row + k] * x[k];
      h[j] = acc > 0 ? acc : 0;
    }
    let y = b2[0];
    for (let j = 0; j < H; j++) y += W2[j] * h[j];
    mse += (y - ts[i]) ** 2;
  }
  const rmse = Math.sqrt(mse / xs.length);
  console.log(`\n  Final train RMSE: ${rmse.toFixed(1)} cp  (teacher std: ${tStd.toFixed(1)} cp)`);
  console.log(`  R² ≈ ${(1 - mse / xs.length / (tStd ** 2)).toFixed(3)}  (1.0 = perfect fit)`);

  // ── Step 4: Print weights ──────────────────────────────────────────────────
  function fmt(arr: Float32Array): string {
    return Array.from(arr).map(v => v.toFixed(6)).join(',');
  }

  console.log(`\n[3/3] Paste this into src/coreClaude/nnWeights.ts:\n`);
  console.log(`// Auto-generated by scripts/trainNN.ts (distillation)`);
  console.log(`// ${NUM_GAMES} games, ${samples.length} positions, ${EPOCHS} epochs`);
  console.log(`// Train RMSE: ${rmse.toFixed(1)} cp  |  R²: ${(1 - mse / xs.length / (tStd ** 2)).toFixed(3)}`);
  console.log(``);
  console.log(`export const NN_TRAINED = true;`);
  console.log(``);
  console.log(`export const NN_W1 = new Float32Array([${fmt(W1)}]);`);
  console.log(``);
  console.log(`export const NN_b1 = new Float32Array([${fmt(b1)}]);`);
  console.log(``);
  console.log(`export const NN_W2 = new Float32Array([${fmt(W2)}]);`);
  console.log(``);
  console.log(`export const NN_b2 = new Float32Array([${fmt(b2)}]);`);
})();
