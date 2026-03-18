// scripts/evalAB.ts — A/B test: New eval vs Old eval (no book, same engine)
//
// Run with:
//   npx tsx --tsconfig tsconfig.test.json scripts/evalAB.ts

import { iterativeDeepening, setEvalFn, resetEvalFn } from '../src/coreCodex/search/alphabeta';
import { TT } from '../src/coreCodex/search/tt';
import { generateMoves, applyMove, Move } from '../src/coreCodex/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreCodex/position';
import { hashPosition } from '../src/coreCodex/search/zobrist';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreCodex/search/repetition';
import { B1, BB, bitCount, bits, STEPS, toRC } from '../src/coreCodex/bitboards';

const THINK_MS  = 600;
const NUM_GAMES = 20;
const MAX_PLIES = 300;

// ── Old eval (pre-Texel baseline) ─────────────────────────────────────────────

const OLD_P1_PST = new Int16Array(32);
const OLD_P2_PST = new Int16Array(32);
const OLD_K_PST  = new Int16Array(32);

(function() {
  const rowBonus = [42, 34, 24, 15, 8, 4, 1, 0];
  const colBonus = [0, 1, 3, 6, 6, 3, 1, 0];
  for (let sq = 0; sq < 32; sq++) {
    const { r, c } = toRC(sq);
    OLD_P1_PST[sq] = rowBonus[r]     + colBonus[c];
    OLD_P2_PST[sq] = rowBonus[7 - r] + colBonus[c];
    const dr = r <= 3 ? 3 - r : r - 4;
    const dc = c <= 3 ? 3 - c : c - 4;
    OLD_K_PST[sq] = Math.max(0, 8 - Math.max(dr, dc) * 2);
  }
})();

function evalOld(p: Position): number {
  const total = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  const eg    = total <= 8 ? (8 - total) / 8 : 0;
  const s     = p.side;

  // material (old: VAL_KING=280, no phase scaling)
  let score =
    100 * (bitCount(s===1?p.p1Men:p.p2Men)   - bitCount(s===1?p.p2Men:p.p1Men)) +
    280 * (bitCount(s===1?p.p1Kings:p.p2Kings) - bitCount(s===1?p.p2Kings:p.p1Kings));

  // psqt
  const myM = s===1?OLD_P1_PST:OLD_P2_PST, opM = s===1?OLD_P2_PST:OLD_P1_PST;
  for (const sq of bits(s===1?p.p1Men:p.p2Men))    score += myM[sq];
  for (const sq of bits(s===1?p.p2Men:p.p1Men))    score -= opM[sq];
  for (const sq of bits(s===1?p.p1Kings:p.p2Kings)) score += OLD_K_PST[sq];
  for (const sq of bits(s===1?p.p2Kings:p.p1Kings)) score -= OLD_K_PST[sq];

  // mobility ×3
  const occ = (p.p1Men|p.p1Kings|p.p2Men|p.p2Kings)>>>0;
  const opp = s===1?-1:1 as 1|-1;
  let my=0, op=0;
  for (const sq of bits(s===1?p.p1Men:p.p2Men))
    for (const st of STEPS[sq]) {
      if (s===1 &&(st.dir==='DL'||st.dir==='DR')) continue;
      if (s===-1&&(st.dir==='UL'||st.dir==='UR')) continue;
      if (!(occ&B1(st.to))) my++;
    }
  for (const sq of bits(s===1?p.p1Kings:p.p2Kings))
    for (const st of STEPS[sq]) if (!(occ&B1(st.to))) my++;
  for (const sq of bits(s===1?p.p2Men:p.p1Men))
    for (const st of STEPS[sq]) {
      if (opp===1 &&(st.dir==='DL'||st.dir==='DR')) continue;
      if (opp===-1&&(st.dir==='UL'||st.dir==='UR')) continue;
      if (!(occ&B1(st.to))) op++;
    }
  for (const sq of bits(s===1?p.p2Kings:p.p1Kings))
    for (const st of STEPS[sq]) if (!(occ&B1(st.to))) op++;
  score += 3*(my-op);

  // back rank ×(1-eg)
  for (const sq of bits(s===1?p.p1Men:p.p2Men)) {
    const {r}=toRC(sq);
    if ((s===1&&r===7)||(s===-1&&r===0)) score += 5*(1-eg);
  }
  for (const sq of bits(s===1?p.p2Men:p.p1Men)) {
    const {r}=toRC(sq);
    if ((s===1&&r===0)||(s===-1&&r===7)) score -= 5*(1-eg);
  }

  // protected men ×9
  const myMen=s===1?p.p1Men:p.p2Men, opMen=s===1?p.p2Men:p.p1Men;
  const myAll=(myMen|(s===1?p.p1Kings:p.p2Kings))>>>0;
  const opAll=(opMen|(s===1?p.p2Kings:p.p1Kings))>>>0;
  for (const sq of bits(myMen))
    for (const st of STEPS[sq]) {
      if (s===1 &&(st.dir==='UL'||st.dir==='UR')) continue;
      if (s===-1&&(st.dir==='DL'||st.dir==='DR')) continue;
      if (myAll&B1(st.to)) { score+=9; break; }
    }
  for (const sq of bits(opMen))
    for (const st of STEPS[sq]) {
      if (s===1 &&(st.dir==='DL'||st.dir==='DR')) continue;
      if (s===-1&&(st.dir==='UL'||st.dir==='UR')) continue;
      if (opAll&B1(st.to)) { score-=9; break; }
    }

  // simplification ×2
  const myN=bitCount(s===1?p.p1Men|p.p1Kings:p.p2Men|p.p2Kings);
  const opN=bitCount(s===1?p.p2Men|p.p2Kings:p.p1Men|p.p1Kings);
  if (myN>opN) score+=(16-total)*2;

  // king endgame proximity ×3
  const myKings=s===1?p.p1Kings:p.p2Kings;
  const opMen2 =s===1?p.p2Men:p.p1Men;
  if (myKings&&opMen2&&myN>opN&&eg>0) {
    let keg=0;
    for (const kSq of bits(myKings)) {
      const {r:kr,c:kc}=toRC(kSq);
      for (const mSq of bits(opMen2)) {
        const {r:mr,c:mc}=toRC(mSq);
        keg+=Math.max(0,12-Math.abs(kr-mr)-Math.abs(kc-mc))*3;
      }
    }
    score+=keg*eg;
  }

  return score|0;
}

// ── Game runner ────────────────────────────────────────────────────────────────

type Result = 'new' | 'old' | 'draw';

async function playGame(newSide: 1|-1, gameNum: number): Promise<Result> {
  let pos = initialPosition();
  const history: number[] = [hashPosition(pos)];
  const ttNew = new TT(), ttOld = new TT();

  process.stdout.write(`  Game ${String(gameNum).padStart(2)}: New=${newSide===1?'P1':'P2'} `);

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const moves = generateMoves(pos);
    if (!moves.length) {
      const winner: Result = pos.side === newSide ? 'old' : 'new';
      console.log(`→ ${winner} wins at ply ${ply+1} (no moves)`);
      return winner;
    }
    if (isDrawByInactivity(pos)) { console.log(`→ draw (inactivity)`); return 'draw'; }
    if (isThreefoldRepetition(buildRepetitionCounts(history), hashPosition(pos))) {
      console.log(`→ draw (threefold)`); return 'draw';
    }

    const isNew = pos.side === newSide;
    const tt    = isNew ? ttNew : ttOld;

    // Override eval for "old" side by temporarily patching
    let move: Move | undefined;
    if (isNew) {
      resetEvalFn();
      const res = await iterativeDeepening(pos, THINK_MS, tt, undefined, history);
      move = res.best;
    } else {
      setEvalFn(evalOld);
      const res = await iterativeDeepening(pos, THINK_MS, tt, undefined, history);
      resetEvalFn();
      move = res.best;
    }

    if (!move) move = moves[0];
    pos = applyMove(pos, move);
    history.push(hashPosition(pos));
  }
  console.log(`→ draw (max plies)`);
  return 'draw';
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nA/B Test: New eval (Texel-tuned) vs Old eval (pre-Texel)`);
  console.log(`Think: ${THINK_MS}ms/move  |  Games: ${NUM_GAMES}\n`);

  let newW=0, oldW=0, draws=0;
  for (let g=0; g<NUM_GAMES; g++) {
    const newSide: 1|-1 = g%2===0 ? 1 : -1;
    const r = await playGame(newSide, g+1);
    if (r==='new') newW++; else if (r==='old') oldW++; else draws++;
  }

  const total=newW+oldW+draws;
  const pct=(n:number)=>((n/total)*100).toFixed(0)+'%';
  console.log(`\n${'═'.repeat(52)}`);
  console.log('RESULTS');
  console.log(`${'─'.repeat(52)}`);
  console.log(`New eval (Texel-tuned) : ${String(newW).padStart(3)} wins  (${pct(newW)})`);
  console.log(`Old eval (pre-Texel)   : ${String(oldW).padStart(3)} wins  (${pct(oldW)})`);
  console.log(`Draws                  : ${String(draws).padStart(3)}        (${pct(draws)})`);
  console.log(`${'─'.repeat(52)}`);
  const verdict = newW>oldW ? '✓ New eval is stronger'
    : oldW>newW ? '✗ Old eval is stronger — consider reverting'
    : '~ No clear difference';
  console.log(`Verdict: ${verdict}`);
  console.log(`${'═'.repeat(52)}\n`);
}

main().catch(console.error);
