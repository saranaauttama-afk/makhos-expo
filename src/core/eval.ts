// src/core/eval.ts
import { BB, bitCount, B1, STEPS, toRC, bits } from './bitboards';
import { Position } from './position';

const START_TOTAL = 16; // 8 ต่อฝั่ง

// น้ำหนักพื้นฐาน (จะผสมตาม phase เกม)
const W_BASE = {
  man: 100,
  king: 220,
  mobilityMen: 2,
  mobilityKing: 3,
  center: 2,
  promoteProgress: 6,
  backRankGuard: 3,
  kingProximity: 2,
  trappedKing: -12,
  simplification: 3, // ชอบเทรดเมื่อได้เปรียบ (ช่วยปิดเกม)
};

// phase: 1 = ต้นเกม, 0 = ปลายเกม
function gamePhase(p: Position) {
  const total = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  const t = Math.max(0, Math.min(1, total / START_TOTAL));
  return t;
}
function occupied(p: Position): BB {
  return (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
}

// ระยะโปรโมต: P1 เดินขึ้น → ระยะ = r, P2 เดินลง → ระยะ = 7-r
function promotionDistanceSum(p: Position, side: 1 | -1): number {
  const men = side === 1 ? p.p1Men : p.p2Men;
  let sum = 0;
  for (const i of bits(men)) {
    const { r } = toRC(i);
    sum += side === 1 ? r : (7 - r);
  }
  return sum;
}

// คุมเซ็นเตอร์หยาบ ๆ
function centerScore(p: Position, side: 1 | -1): number {
  let sc = 0;
  const men = side === 1 ? p.p1Men : p.p2Men;
  const kings = side === 1 ? p.p1Kings : p.p2Kings;
  for (const s of [...bits(men), ...bits(kings)]) {
    const { r, c } = toRC(s);
    if (r >= 2 && r <= 5 && c >= 2 && c <= 5) sc++;
  }
  return sc;
}

// การ์ดแถวหลังบ้าน
function backRankGuards(p: Position, side: 1 | -1): number {
  const men = side === 1 ? p.p1Men : p.p2Men;
  let guards = 0;
  for (const i of bits(men)) {
    const { r } = toRC(i);
    if ((side === 1 && r === 7) || (side === -1 && r === 0)) guards++;
  }
  return guards;
}

// mobility (แบบเบา ๆ)
function mobility(p: Position, side: 1 | -1): { men: number; king: number } {
  const occ = occupied(p);
  let menMoves = 0, kingMoves = 0;

  const myMen = side === 1 ? p.p1Men : p.p2Men;
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;

  for (const i of bits(myMen)) {
    for (const st of STEPS[i]) {
      if (side === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (((~occ) >>> 0) & B1(st.to)) menMoves++;
    }
  }
  for (const i of bits(myKings)) {
    for (const st of STEPS[i]) {
      if (((~occ) >>> 0) & B1(st.to)) kingMoves++;
    }
  }
  return { men: menMoves, king: kingMoves };
}

// คิงติดกับ (ไม่มีทางออกใกล้ ๆ)
function trappedKings(p: Position, side: 1 | -1): number {
  const occ = occupied(p);
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  let trapped = 0;
  for (const i of bits(myKings)) {
    let exits = 0;
    for (const st of STEPS[i]) if (((~occ) >>> 0) & B1(st.to)) exits++;
    if (exits === 0) trapped++;
  }
  return trapped;
}

// ปลายเกม: คิงของเราควร “กดระยะ” เข้าหาศัตรู
function kingProximityGain(p: Position, side: 1 | -1): number {
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  const opAll   = side === 1 ? (p.p2Men | p.p2Kings) : (p.p1Men | p.p1Kings);
  if (opAll === 0) return 0;

  let sum = 0, cnt = 0;
  for (const k of bits(myKings)) {
    const { r: rk, c: ck } = toRC(k);
    let best = 99;
    for (const e of bits(opAll)) {
      const { r: re, c: ce } = toRC(e);
      const d = Math.max(Math.abs(rk - re), Math.abs(ck - ce));
      if (d < best) best = d;
    }
    if (best < 99) { sum += best; cnt++; }
  }
  if (!cnt) return 0;
  const avg = sum / cnt;
  return Math.max(0, 6 - avg);
}

export function evaluate(p: Position): number {
  const gp = gamePhase(p), eg = 1 - gp;
  const W = {
    man: W_BASE.man,
    king: Math.round(W_BASE.king + 40 * eg),
    mobilityMen: W_BASE.mobilityMen,
    mobilityKing: Math.round(W_BASE.mobilityKing + 1 * eg),
    center: W_BASE.center,
    promoteProgress: Math.round(W_BASE.promoteProgress + 4 * eg),
    backRankGuard: W_BASE.backRankGuard,
    kingProximity: Math.round(W_BASE.kingProximity + 3 * eg),
    trappedKing: W_BASE.trappedKing,
    simplification: W_BASE.simplification,
  };

  const myMen   = p.side === 1 ? p.p1Men : p.p2Men;
  const myKings = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen   = p.side === 1 ? p.p2Men : p.p1Men;
  const opKings = p.side === 1 ? p.p2Kings : p.p1Kings;

  const myMenN   = bitCount(myMen);
  const myKingsN = bitCount(myKings);
  const opMenN   = bitCount(opMen);
  const opKingsN = bitCount(opKings);

  let score = 0;

  // material
  score += W.man  * (myMenN   - opMenN);
  score += W.king * (myKingsN - opKingsN);

  // mobility
  const mMob = mobility(p, p.side);
  const oMob = mobility({ ...p, side: (p.side === 1 ? -1 : 1) } as Position, (p.side === 1 ? -1 : 1));
  score += W.mobilityMen  * (mMob.men  - oMob.men);
  score += W.mobilityKing * (mMob.king - oMob.king);

  // center
  score += W.center * (centerScore(p, p.side) - centerScore(p, (p.side === 1 ? -1 : 1)));

  // promotion / back rank
  const myProm  = promotionDistanceSum(p, p.side);
  const opProm  = promotionDistanceSum(p, (p.side === 1 ? -1 : 1));
  score += W.promoteProgress * (opProm - myProm) / 10;
  score += W.backRankGuard * (backRankGuards(p, p.side) - backRankGuards(p, (p.side === 1 ? -1 : 1)));

  // endgame pressure
  score += W.kingProximity * (kingProximityGain(p, p.side) - kingProximityGain(p, (p.side === 1 ? -1 : 1)));
  score += W.trappedKing   * (trappedKings(p, p.side) - trappedKings(p, (p.side === 1 ? -1 : 1)));

  // simplification bias (ชอบเทรดเมื่อ “นำ”)
  const totalNow = myMenN + myKingsN + opMenN + opKingsN;
  const leadSimple = (myMenN - opMenN) + 2 * (myKingsN - opKingsN); // นับ king = 2
  const leadSign = leadSimple > 0 ? 1 : leadSimple < 0 ? -1 : 0;
  score += W.simplification * leadSign * (START_TOTAL - totalNow);

  return score | 0;
}
