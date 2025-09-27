// src/core/eval.ts
import { BB, bitCount, B1, STEPS, toRC, bits } from './bitboards';
import { generateMoves } from './movegen';
import { Position } from './position';

const START_TOTAL = 16; // 8 ต่อฝั่ง

// น้ำหนักพื้นฐาน (จะผสมตาม phase เกม)
const W_BASE = {
  man: 100,
  king: 210,            // เดิม 220 → ลดค่าคิงฐานลงเล็กน้อย
  mobilityMen: 2,
  mobilityKing: 3,
  center: 2,
  promoteProgress: 6,
  backRankGuard: 3,
  kingProximity: 2,
  trappedKing: -12,
  simplification: 6,    // เดิม 3 → เพิ่มแรงผลัก "ยอมแลกเมื่อได้เปรียบ"
  captureSwing: 90,      // Immediate capture swing weight (discourage hanging pieces)
  captureTargets: 45,   // Count of pieces immediately capturable
};

// phase: 1 = ต้นเกม, 0 = ปลายเกม
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


interface CaptureInfo { maxChain: number; targets: number; }

function captureInfo(p: Position, side: 1 | -1): CaptureInfo {
  const view = (p.side === side ? p : { ...p, side } as Position);
  const moves = generateMoves(view);
  if (!moves.length || moves[0].captured.length === 0) {
    return { maxChain: 0, targets: 0 };
  }
  let maxChain = 0;
  let mask = 0;
  for (const m of moves) {
    const len = m.captured.length;
    if (len > maxChain) maxChain = len;
    for (const sq of m.captured) mask = (mask | B1(sq)) >>> 0;
  }
  return { maxChain, targets: bitCount(mask) };
}

export function evaluate(p: Position): number {
  const gpTotal = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  const gp = Math.max(0, Math.min(1, gpTotal / START_TOTAL));
  const eg = 1 - gp; // 1 = ปลายเกม

  const myMen   = p.side === 1 ? p.p1Men : p.p2Men;
  const myKings = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen   = p.side === 1 ? p.p2Men : p.p1Men;
  const opKings = p.side === 1 ? p.p2Kings : p.p1Kings;

  const myMenN   = bitCount(myMen);
  const myKingsN = bitCount(myKings);
  const opMenN   = bitCount(opMen);
  const opKingsN = bitCount(opKings);

  const myAll = myMenN + myKingsN;
  const opAll = opMenN + opKingsN;

  // สถานะ “นำ” แบบหยาบ (คิงนับ 2)
  const leadSimple = (myMenN - opMenN) + 2 * (myKingsN - opKingsN);
  const leading = leadSimple > 0;

  // ==== Dynamic Weights (ไม่แตะ movegen/search) ====

  // 1) ลดค่าคิง "เมื่อปลายเกมและเรานำ" → กล้าแลกเพื่อปิด
  let W_KING = W_BASE.king;
  if (eg >= 0.5 && leading) W_KING -= 60;         // ปลายเกมครึ่งหลัง
  if (eg >= 0.8 && leading && opAll <= 2) W_KING -= 90; // จะจบแล้ว ยิ่งลด

  // 2) เร่งโปรโมตในปลายเกม → ดันให้เดินที่ส่งเสริมเม็ด
  const W_PROM = W_BASE.promoteProgress + Math.round(6 * eg); // 6 → 12 เมื่อ eg=1

  // 3) Simplification bias: เมื่อ "นำ" และอีกฝ่ายเหลือน้อย → อยากลดชิ้นรวม
  let SIMP = W_BASE.simplification;
  if (leading) SIMP += Math.round(8 * eg); // ปลายเกมเพิ่มแรง
  if (leading && opAll <= 2) SIMP += 10;   // จะปิดแล้ว เพิ่มพิเศษ

  // ==== คะแนนจริง ====
  let score = 0;

  // Material
  score += W_BASE.man  * (myMenN   - opMenN);
  score += W_KING      * (myKingsN - opKingsN);

  // Mobility (เบา ๆ)
  const mMob = mobility(p, p.side);
  const oMob = mobility({ ...p, side: (p.side === 1 ? -1 : 1) } as Position, (p.side === 1 ? -1 : 1));
  score += W_BASE.mobilityMen  * (mMob.men  - oMob.men);
  score += W_BASE.mobilityKing * (mMob.king - oMob.king);

  // Center / Back rank / King proximity / Trapped king (เดิม)
  score += W_BASE.center * (centerScore(p, p.side) - centerScore(p, (p.side === 1 ? -1 : 1)));
  const myProm  = promotionDistanceSum(p, p.side);
  const opProm  = promotionDistanceSum(p, (p.side === 1 ? -1 : 1));
  score += W_PROM * (opProm - myProm) / 10;
  score += W_BASE.backRankGuard * (backRankGuards(p, p.side) - backRankGuards(p, (p.side === 1 ? -1 : 1)));
  score += W_BASE.kingProximity * (kingProximityGain(p, p.side) - kingProximityGain(p, (p.side === 1 ? -1 : 1)));
  score += W_BASE.trappedKing   * (trappedKings(p, p.side) - trappedKings(p, (p.side === 1 ? -1 : 1)));

  const myCap = captureInfo(p, p.side);
  const opCap = captureInfo(p, (p.side === 1 ? -1 : 1));
  if (myCap.maxChain || opCap.maxChain) {
    let capWeight = W_BASE.captureSwing;
    if (eg >= 0.7) capWeight += 20;
    score += capWeight * (myCap.maxChain - opCap.maxChain);
  }
  if (myCap.targets || opCap.targets) {
    const threatWeight = W_BASE.captureTargets + Math.round(4 * eg);
    score += threatWeight * (myCap.targets - opCap.targets);
  }
  // Simplification (อยากให้ชิ้นรวมน้อยลงเมื่อเรานำ) — ไม่แตะตัวเลขมโหฬาร
  score += SIMP * (START_TOTAL - (myAll + opAll));

  // Endgame boosters เล็กน้อย (ทั่วไป ไม่ผูกกับรูปเฉพาะ)
  if (leading && opAll === 1) score += 140;   // เขาเหลือ 1 ตัว → ผลักให้ยอมแลกเพื่อจบ
  if (leading && opAll <= 2) score += 70;

  return score | 0;
}


