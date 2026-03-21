// scripts/genTrainingData.ts
//
// Generate training positions for Colab GPU training.
// Output: data/positions.jsonl  (one JSON object per line)
//
// Each line: { "x": [128 floats], "t": score_cp }
//
// Input encoding (128 features = 4 channels × 32 squares):
//   x[  0.. 31] = 1 if my man at square i
//   x[ 32.. 63] = 1 if my king at square i
//   x[ 64.. 95] = 1 if enemy man at square i
//   x[ 96..127] = 1 if enemy king at square i
//
// Teacher signal: hand-crafted evaluate() — instant, no oracle overhead
// (Colab trains a larger network that can generalise better than hand-crafted)
//
// Usage:
//   npx tsx --tsconfig tsconfig.test.json scripts/genTrainingData.ts
//
// Expected runtime: ~15–25 min (300 games × 50ms/move)

import * as fs from 'fs';
import * as path from 'path';
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

if (NN_TRAINED) {
  console.error('ERROR: Set NN_TRAINED = false in nnWeights.ts first.');
  process.exit(1);
}

// ── Config ────────────────────────────────────────────────────────────────────
const NUM_GAMES  = 300;
const THINK_MS   = 50;    // fast — just for position diversity
const MAX_PLIES  = 120;
const CLIP       = 800;   // skip blown-out positions
const OUT_FILE   = path.join(process.cwd(), 'data', 'positions.jsonl');

// ── Feature extraction (128-dim one-hot, side-relative) ───────────────────────
function getFeatures(pos: Position): number[] {
  const x    = new Array<number>(128).fill(0);
  const side = pos.side;
  for (const sq of bits(side === 1 ? pos.p1Men   : pos.p2Men))   x[sq]       = 1;
  for (const sq of bits(side === 1 ? pos.p1Kings : pos.p2Kings)) x[32 + sq]  = 1;
  for (const sq of bits(side === 1 ? pos.p2Men   : pos.p1Men))   x[64 + sq]  = 1;
  for (const sq of bits(side === 1 ? pos.p2Kings : pos.p1Kings)) x[96 + sq]  = 1;
  return x;
}

// ── Play one game and collect positions ───────────────────────────────────────
async function collectGame(out: fs.WriteStream): Promise<number> {
  const hashes: number[] = [];
  let pos = initialPosition();
  const tt = new TT(1 << 16);
  let collected = 0;

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const moves = generateMoves(pos);
    if (!moves.length) break;

    const hash = hashPosition(pos);
    hashes.push(hash);
    if (isThreefoldRepetition(buildRepetitionCounts(hashes)) || isDrawByInactivity(pos)) break;

    // Teacher score
    const score = evaluate(pos);

    // Record position if not blown out
    if (Math.abs(score) <= CLIP) {
      out.write(JSON.stringify({ x: getFeatures(pos), t: score }) + '\n');
      collected++;
    }

    // Move: random for first 5 plies (diversity), engine after
    let move;
    if (ply < 5) {
      move = moves[Math.floor(Math.random() * moves.length)];
    } else {
      const res = await iterativeDeepening(pos, THINK_MS, tt, undefined, hashes);
      if (!res.best) break;
      move = res.best;
    }
    pos = applyMove(pos, move);
  }

  return collected;
}

// ── Main ──────────────────────────────────────────────────────────────────────
(async () => {
  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  const out = fs.createWriteStream(OUT_FILE);

  console.log(`Generating ${NUM_GAMES} games → ${OUT_FILE}`);
  console.log(`Think: ${THINK_MS}ms/move  |  Clip: ±${CLIP}cp\n`);

  let total = 0;
  for (let g = 0; g < NUM_GAMES; g++) {
    total += await collectGame(out);
    if ((g + 1) % 20 === 0 || g === NUM_GAMES - 1) {
      process.stdout.write(`\r  game ${g + 1}/${NUM_GAMES}  positions: ${total}`);
    }
  }

  out.end();
  console.log(`\n\nDone — ${total} positions saved to data/positions.jsonl`);
  const mb = (fs.statSync(OUT_FILE).size / 1048576).toFixed(1);
  console.log(`File size: ${mb} MB — ready to upload to Google Drive`);
})();
