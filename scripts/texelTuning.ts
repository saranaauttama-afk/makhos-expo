// scripts/texelTuning.ts
//
// Texel Tuning: auto-optimize eval parameters using self-play game outcomes.
//
// Method:
//   1. Play NUM_GAMES self-play games at low depth → collect (position, outcome) pairs
//   2. Find optimal sigmoid scaling K
//   3. Coordinate descent: adjust each parameter ±STEP, keep if MSE decreases
//   4. Print tuned values to paste into eval.ts
//
// Usage:
//   npx tsx --tsconfig tsconfig.test.json scripts/texelTuning.ts
//
// Typical runtime: ~5–15 min (depends on NUM_GAMES and ITERS).

import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { B1, bitCount, bits, STEPS, toRC } from '../src/coreClaude/bitboards';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { hashPosition } from '../src/coreClaude/search/zobrist';

// ── Config ────────────────────────────────────────────────────────────────────
const NUM_GAMES  = 300;  // self-play games to generate training data
const THINK_MS   = 250;  // ms per move during self-play (low = fast, less accurate)
const MAX_PLIES  = 150;  // draw if game exceeds this
const ITERS      = 120;  // coordinate descent iterations
const STEP       = 2;    // initial parameter adjustment step (cp)
const MIN_STEP   = 1;    // stop when step shrinks below this

// ── Eval parameters ───────────────────────────────────────────────────────────
interface EvalParams {
  rowBonus:    number[]; // [8] advancement bonus by row (0=top,7=bottom)
  colBonus:    number[]; // [8] column safety bonus
  kingCentre:  number;   // king centrality max bonus (cp)
  valMan:      number;   // man piece value (cp)
  valKing:     number;   // king piece value (cp)
  mobilityW:   number;   // mobility score weight (multiplier)
  backRank:    number;   // back-rank guard bonus per piece
  protected:   number;   // protected man bonus per piece
  simplify:    number;   // simplification bonus per traded piece
  kegDistW:    number;   // king endgame: bonus per proximity unit
  kegMaxDist:  number;   // king endgame: max Manhattan distance considered
}

const defaultParams: EvalParams = {
  rowBonus:   [42, 34, 24, 15, 8, 4, 1, 0],
  colBonus:   [0,  1,  3,  6,  6, 3, 1, 0],
  kingCentre: 8,
  valMan:     100,
  valKing:    280,  // base king value (endgame adds 100 via phase factor)
  mobilityW:  5,
  backRank:   5,
  protected:  9,
  simplify:   3,
  kegDistW:   3,
  kegMaxDist: 12,
};

// ── Parameterised evaluate ────────────────────────────────────────────────────
// Self-contained copy of eval logic that reads from an EvalParams object.
// This lets the tuner adjust weights without touching the production eval.ts.

function buildPST(p: EvalParams) {
  const P1 = new Int16Array(32), P2 = new Int16Array(32), K = new Int16Array(32);
  for (let sq = 0; sq < 32; sq++) {
    const { r, c } = toRC(sq);
    P1[sq] = p.rowBonus[r]     + p.colBonus[c];
    P2[sq] = p.rowBonus[7 - r] + p.colBonus[c];
    const dr = r <= 3 ? 3 - r : r - 4;
    const dc = c <= 3 ? 3 - c : c - 4;
    K[sq] = Math.max(0, p.kingCentre - Math.max(dr, dc) * 2);
  }
  return { P1, P2, K };
}

function evalWithParams(pos: Position, ep: EvalParams): number {
  const { P1, P2, K } = buildPST(ep);
  const s = pos.side;

  // material (phase-aware king value: scales from valKing in opening to valKing+100 in endgame)
  const total = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const eg = total <= 8 ? (8 - total) / 8 : 0;
  const kingVal = (ep.valKing + eg * 100) | 0;
  let score =
    ep.valMan * (bitCount(s === 1 ? pos.p1Men   : pos.p2Men)   - bitCount(s === 1 ? pos.p2Men   : pos.p1Men)) +
    kingVal   * (bitCount(s === 1 ? pos.p1Kings : pos.p2Kings) - bitCount(s === 1 ? pos.p2Kings : pos.p1Kings));

  // PSQT
  const myMPST = s === 1 ? P1 : P2, opMPST = s === 1 ? P2 : P1;
  for (const sq of bits(s === 1 ? pos.p1Men   : pos.p2Men))   score += myMPST[sq];
  for (const sq of bits(s === 1 ? pos.p2Men   : pos.p1Men))   score -= opMPST[sq];
  for (const sq of bits(s === 1 ? pos.p1Kings : pos.p2Kings)) score += K[sq];
  for (const sq of bits(s === 1 ? pos.p2Kings : pos.p1Kings)) score -= K[sq];

  // mobility
  const occ = (pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings) >>> 0;
  let my = 0, op = 0;
  const opp = s === 1 ? -1 : 1 as 1 | -1;
  for (const sq of bits(s === 1 ? pos.p1Men : pos.p2Men))
    for (const st of STEPS[sq]) {
      if (s  ===  1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (s  === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (!(occ & B1(st.to))) my++;
    }
  for (const sq of bits(s === 1 ? pos.p1Kings : pos.p2Kings))
    for (const st of STEPS[sq]) if (!(occ & B1(st.to))) my++;
  for (const sq of bits(s === 1 ? pos.p2Men : pos.p1Men))
    for (const st of STEPS[sq]) {
      if (opp ===  1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (opp === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (!(occ & B1(st.to))) op++;
    }
  for (const sq of bits(s === 1 ? pos.p2Kings : pos.p1Kings))
    for (const st of STEPS[sq]) if (!(occ & B1(st.to))) op++;
  score += ep.mobilityW * (my - op);

  // back rank
  for (const sq of bits(s === 1 ? pos.p1Men : pos.p2Men)) {
    const { r } = toRC(sq);
    if ((s === 1 && r === 7) || (s === -1 && r === 0)) score += ep.backRank;
  }
  for (const sq of bits(s === 1 ? pos.p2Men : pos.p1Men)) {
    const { r } = toRC(sq);
    if ((s === 1 && r === 0) || (s === -1 && r === 7)) score -= ep.backRank;
  }

  // protected men
  const myMen = s === 1 ? pos.p1Men : pos.p2Men;
  const opMen = s === 1 ? pos.p2Men : pos.p1Men;
  const myAll = (myMen | (s === 1 ? pos.p1Kings : pos.p2Kings)) >>> 0;
  const opAll = (opMen | (s === 1 ? pos.p2Kings : pos.p1Kings)) >>> 0;
  for (const sq of bits(myMen))
    for (const st of STEPS[sq]) {
      if (s === 1  && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (s === -1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (myAll & B1(st.to)) { score += ep.protected; break; }
    }
  for (const sq of bits(opMen))
    for (const st of STEPS[sq]) {
      if (s === 1  && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (s === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (opAll & B1(st.to)) { score -= ep.protected; break; }
    }

  // simplification
  const myN = bitCount(s === 1 ? pos.p1Men | pos.p1Kings : pos.p2Men | pos.p2Kings);
  const opN = bitCount(s === 1 ? pos.p2Men | pos.p2Kings : pos.p1Men | pos.p1Kings);
  if (myN > opN) score += (16 - total) * ep.simplify;

  // king endgame proximity
  const myKings = s === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen2  = s === 1 ? pos.p2Men   : pos.p1Men;
  if (myKings && opMen2 && myN > opN) {
    if (eg > 0) {
      let keg = 0;
      for (const kSq of bits(myKings)) {
        const { r: kr, c: kc } = toRC(kSq);
        for (const mSq of bits(opMen2)) {
          const { r: mr, c: mc } = toRC(mSq);
          const dist = Math.abs(kr - mr) + Math.abs(kc - mc);
          keg += Math.max(0, ep.kegMaxDist - dist) * ep.kegDistW;
        }
      }
      score += keg * eg;
    }
  }

  return score | 0;
}

// ── Self-play game generator ──────────────────────────────────────────────────
// outcome: 1 = P1 wins, -1 = P2 wins, 0 = draw (from P1's perspective)
interface Sample { pos: Position; outcome: number; }

async function playSelfPlayGame(tt: TT): Promise<Sample[]> {
  let pos = initialPosition();
  const hashes: number[] = [hashPosition(pos)];
  const positions: Position[] = [pos];
  let plies = 0;

  while (plies < MAX_PLIES) {
    const moves = generateMoves(pos);
    if (!moves.length) break;

    const rep = buildRepetitionCounts(hashes);
    if (isThreefoldRepetition(rep, hashPosition(pos))) break;
    if (isDrawByInactivity(pos)) break;

    const result = await iterativeDeepening(
      pos, THINK_MS, tt, () => {}, hashes, { cancelled: false },
    );
    const move = result.best ?? moves[0];
    pos = applyMove(pos, move);
    hashes.push(hashPosition(pos));
    positions.push(pos);
    plies++;
  }

  // Determine outcome (from P1's perspective)
  let outcome = 0;
  const finalMoves = generateMoves(pos);
  if (!finalMoves.length) {
    // Current side has no moves → loses
    outcome = pos.side === 1 ? -1 : 1;
  }
  // else: draw by inactivity / repetition / max plies → outcome stays 0

  // Return only non-terminal positions (skip last 2)
  return positions.slice(0, -2).map(p => ({ pos: p, outcome }));
}

// ── Texel sigmoid & MSE ───────────────────────────────────────────────────────
function sigmoid(x: number, K: number): number {
  return 1 / (1 + Math.pow(10, -K * x / 400));
}

function mse(samples: Sample[], ep: EvalParams, K: number): number {
  let err = 0;
  for (const { pos, outcome } of samples) {
    // Convert outcome (-1/0/1) to [0,1] probability (from current side's view)
    const sideOutcome = pos.side === 1 ? outcome : -outcome;
    const t = (sideOutcome + 1) / 2; // -1→0, 0→0.5, 1→1
    const e = evalWithParams(pos, ep);
    const p = sigmoid(e, K);
    err += (t - p) ** 2;
  }
  return err / samples.length;
}

function findK(samples: Sample[], ep: EvalParams): number {
  let lo = 0.1, hi = 5.0;
  for (let i = 0; i < 50; i++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (mse(samples, ep, m1) < mse(samples, ep, m2)) hi = m2; else lo = m1;
  }
  return (lo + hi) / 2;
}

// ── Coordinate descent ────────────────────────────────────────────────────────
function cloneParams(ep: EvalParams): EvalParams {
  return {
    ...ep,
    rowBonus: [...ep.rowBonus],
    colBonus: [...ep.colBonus],
  };
}

function tuneParams(samples: Sample[], baseParams: EvalParams): EvalParams {
  let ep = cloneParams(baseParams);
  const K = findK(samples, ep);
  console.log(`  K = ${K.toFixed(4)}`);

  let step = STEP;
  for (let iter = 0; iter < ITERS && step >= MIN_STEP; iter++) {
    let improved = false;
    const baseMSE = mse(samples, ep, K);

    // Tune rowBonus (skip row 0 and row 7 — rarely occupied in midgame)
    for (let i = 1; i < 7; i++) {
      for (const delta of [step, -step]) {
        const candidate = cloneParams(ep);
        candidate.rowBonus[i] = Math.max(0, ep.rowBonus[i] + delta);
        if (mse(samples, candidate, K) < baseMSE) { ep = candidate; improved = true; break; }
      }
    }
    // Tune colBonus
    for (let i = 0; i < 8; i++) {
      for (const delta of [step, -step]) {
        const candidate = cloneParams(ep);
        candidate.colBonus[i] = Math.max(0, ep.colBonus[i] + delta);
        if (mse(samples, candidate, K) < baseMSE) { ep = candidate; improved = true; break; }
      }
    }
    // Tune scalar params
    const scalars: (keyof EvalParams)[] = ['mobilityW', 'backRank', 'protected', 'simplify', 'kegDistW'];
    for (const key of scalars) {
      for (const delta of [step, -step]) {
        const candidate = cloneParams(ep);
        (candidate as Record<string, number>)[key as string] =
          Math.max(0, (ep as Record<string, number>)[key as string] + delta);
        if (mse(samples, candidate, K) < baseMSE) { ep = candidate; improved = true; break; }
      }
    }

    if (!improved) step = Math.floor(step / 2);
    if (iter % 10 === 0) {
      process.stdout.write(`\r  iter ${iter}/${ITERS}  step=${step}  MSE=${mse(samples, ep, K).toFixed(6)}   `);
    }
  }
  console.log();
  return ep;
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`Texel Tuning  NUM_GAMES=${NUM_GAMES}  THINK_MS=${THINK_MS}ms  ITERS=${ITERS}`);
  const tt = new TT();
  const samples: Sample[] = [];

  console.log('\n[1/3] Generating self-play games...');
  const gameStart = Date.now();
  for (let g = 0; g < NUM_GAMES; g++) {
    const gameSamples = await playSelfPlayGame(tt);
    samples.push(...gameSamples);
    const pct = ((g + 1) / NUM_GAMES * 100).toFixed(0);
    process.stdout.write(`\r  game ${g + 1}/${NUM_GAMES} (${pct}%)  positions: ${samples.length}   `);
  }
  console.log(`\n  Done — ${samples.length} positions in ${((Date.now() - gameStart) / 1000).toFixed(1)}s`);

  console.log('\n[2/3] Tuning parameters...');
  const tuned = tuneParams(samples, defaultParams);

  const baseMSE  = mse(samples, defaultParams, findK(samples, defaultParams));
  const tunedMSE = mse(samples, tuned,         findK(samples, tuned));
  const improvement = ((baseMSE - tunedMSE) / baseMSE * 100).toFixed(2);
  console.log(`  Base  MSE: ${baseMSE.toFixed(6)}`);
  console.log(`  Tuned MSE: ${tunedMSE.toFixed(6)}  (${improvement}% improvement)`);

  console.log('\n[3/3] Results — paste into src/coreClaude/eval.ts:\n');
  console.log(`  const rowBonus = [${tuned.rowBonus.join(', ')}];`);
  console.log(`  const colBonus = [${tuned.colBonus.join(', ')}];`);
  console.log(`  kingCentre max = ${tuned.kingCentre}`);
  console.log(`  VAL_MAN    = ${tuned.valMan}`);
  console.log(`  VAL_KING   = ${tuned.valKing}`);
  console.log(`  mobilityW  = ${tuned.mobilityW}   (currently 3  in mobilityScore)`);
  console.log(`  backRank   = ${tuned.backRank}   (cp per piece)`);
  console.log(`  protected  = ${tuned.protected}   (cp per piece)`);
  console.log(`  simplify   = ${tuned.simplify}   (cp per traded piece)`);
  console.log(`  kegDistW   = ${tuned.kegDistW}   (king endgame proximity weight)`);
}

main().catch(console.error);
