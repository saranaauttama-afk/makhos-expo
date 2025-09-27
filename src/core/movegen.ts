// src/core/movegen.ts
// กติกา: เม็ด (man) เดินเฉยเฉพาะ "ไปข้างหน้า" 1 ช่อง, กินเฉพาะ "ไปข้างหน้า" แบบข้าม 1 ตัว
//        ฮอส (king) เดินเฉยได้สุดกระดานตามแนวทแยง, "กิน" ต้องลงที่ช่องถัดไป "ทันที" หลังตัวที่ถูกกิน (ไม่ไถไกล)
//        ทั้งเม็ดและฮอส "กินต่อหลายตา" ได้, บังคับกิน, (ออปชัน) เลือกเส้นที่กินได้มากสุด

import { Position } from './position';
import { BB, B1, bits, STEPS, toRC } from './bitboards';

// ---- ตั้งค่าเสริม ----
const ENFORCE_MAX_CAPTURE = true;

// ฝั่ง/ทิศ (ให้ตรงกับที่ STEPS ใช้)
type Dir = 'UL' | 'UR' | 'DL' | 'DR';

export type Move = {
  from: number;
  to: number;
  captured: number[];
  promote: boolean; // true เฉพาะเม็ดที่จบตาแล้วกลายเป็นฮอส
};

// ---------- helpers พื้นฐาน ----------
function occupied(p: Position): BB {
  return (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
}
function willPromote(side: 1 | -1, to: number): boolean {
  const { r } = toRC(to);
  return side === 1 ? (r === 0) : (r === 7);
}

// ช่องถัดไปใน “ทิศเดียวกัน” โดยอิงจาก STEPS (แทน nextInDir)
function nextStepInDir(from: number, dir: Dir): number {
  for (const st of STEPS[from]) {
    if ((st as any).dir === dir) return (st as any).to as number;
  }
  return -1;
}

// ---------- การสร้าง “เดินเฉย” ----------
function genManQuiet(p: Position, from: number, out: Move[]) {
  const occ = occupied(p);
  for (const st of STEPS[from]) {
    const dir = (st as any).dir as Dir;
    if (p.side === 1 && (dir === 'DL' || dir === 'DR')) continue;
    if (p.side === -1 && (dir === 'UL' || dir === 'UR')) continue;

    const toSq = (st as any).to as number;
    const toBit = B1(toSq);
    if (((~occ) >>> 0) & toBit) {
      out.push({ from, to: toSq, captured: [], promote: willPromote(p.side, toSq) });
    }
  }
}

function genKingQuiet(p: Position, from: number, out: Move[]) {
  const occ0 = occupied(p);
  for (const st of STEPS[from]) {
    const dir = (st as any).dir as Dir;
    let cur = (st as any).to as number;
    while (cur >= 0) {
      const bit = B1(cur);
      if (occ0 & bit) break;      // เจอชิ้น → จบทางนี้
      out.push({ from, to: cur, captured: [], promote: false });
      cur = nextStepInDir(cur, dir);
    }
  }
}

// ---------- การสร้าง “เดินกิน” (DFS รวมกินต่อหลายตา) ----------
// เม็ด: กินเฉพาะไปข้างหน้า ข้ามศัตรู 1 ตัว แลนด์ที่ช่องถัดไป (ห้ามถอยหลัง)
// เม็ด: กินเฉพาะไปข้างหน้า ข้ามศัตรู 1 ตัว แลนด์ที่ช่องถัดไป (ห้ามถอยหลัง)
// *** เวอร์ชันแก้บั๊ก: ไม่ใช้ foundAny แบบแชร์, ตัดสินใจต่อทิศเท่านั้น ***
function genManCapturesFrom(
  p: Position,
  from: number,
  myMen: BB, myKings: BB,
  opMen: BB, opKings: BB,
  visited: number[],
  acc: Move[]
) {
  const occ0 = (myMen | myKings | opMen | opKings) >>> 0;

  for (const st of STEPS[from]) {
    const dir = (st as any).dir as 'UL' | 'UR' | 'DL' | 'DR';
    // เม็ด "กิน" เฉพาะไปข้างหน้า
    if (p.side === 1 && (dir === 'DL' || dir === 'DR')) continue;
    if (p.side === -1 && (dir === 'UL' || dir === 'UR')) continue;

    const over = (st as any).to as number;
    if (over < 0) continue;
    const overBit = B1(over);
    const enemyMask = (opMen | opKings) >>> 0;

    if (((enemyMask & overBit) === 0) || visited.includes(over)) continue;

    // ต้องลงที่ "ช่องถัดไปทันที" ในทิศเดิม
    const landing = nextStepInDir(over, dir);
    if (landing < 0) continue;
    const landingBit = B1(landing);
    if (occ0 & landingBit) continue;

    // จำลองหลังการกิน
    let nMyMen = myMen, nMyKings = myKings;
    let nOpMen = opMen, nOpKings = opKings;
    nOpMen   &= ~overBit;
    nOpKings &= ~overBit;
    nMyMen    = (nMyMen & ~B1(from)) | landingBit;

    const promotedNow = willPromote(p.side, landing);

    // **ตัดสินใจต่อทิศ**: ถ้ากินต่อได้ (และยังไม่โปรโมต) ให้ต่อ; ไม่งั้นจบที่นี่
    let pushed = false;
    if (!promotedNow) {
      const cont: Move[] = [];
      genManCapturesFrom(p, landing, nMyMen, nMyKings, nOpMen, nOpKings, [...visited, over], cont);
      if (cont.length) {
        for (const m of cont) {
          acc.push({ from, to: m.to, captured: [over, ...m.captured], promote: false });
        }
        pushed = true;
      }
    }

    if (!pushed) {
      acc.push({ from, to: landing, captured: [over], promote: promotedNow });
    }
  }
}


// ฮอส: หา “ศัตรูตัวแรกบนทิศ” แล้วแลนด์ที่ "ช่องถัดไปทันที" เท่านั้น (ไม่ไถไกล)
function genKingCapturesFrom(
  p: Position,
  from: number,
  myMen: BB, myKings: BB,
  opMen: BB, opKings: BB,
  visited: number[],
  acc: Move[]
) {
  const occ0 = (myMen | myKings | opMen | opKings) >>> 0;

  for (const st0 of STEPS[from]) {
    const dir = (st0 as any).dir as Dir;

    // เดินตามทิศจนเจอชิ้นแรก
    let cur = (st0 as any).to as number;
    let blockedByOwn = false;
    while (cur >= 0) {
      const curBit = B1(cur);
      if (occ0 & curBit) {
        if ((myMen | myKings) & curBit) blockedByOwn = true;
        break;
      }
      cur = nextStepInDir(cur, dir);
    }
    if (blockedByOwn || cur < 0) continue;

    const enemySq = cur;
    if (!((opMen | opKings) & B1(enemySq)) || visited.includes(enemySq)) continue;

    // แลนด์ได้ "ช่องถัดไป" เท่านั้น
    const landing = nextStepInDir(enemySq, dir);
    if (landing < 0) continue;
    const landingBit = B1(landing);
    if (occ0 & landingBit) continue;

    // จำลองหลังการกิน
    const fromBit = B1(from);
    let nMyMen = myMen, nMyKings = myKings;
    let nOpMen = opMen, nOpKings = opKings;
    nOpMen   &= ~B1(enemySq);
    nOpKings &= ~B1(enemySq);
    nMyKings  = (nMyKings & ~fromBit) | landingBit;

    // กินต่อ
    const cont: Move[] = [];
    genKingCapturesFrom(p, landing, nMyMen, nMyKings, nOpMen, nOpKings, [...visited, enemySq], cont);
    if (cont.length) {
      for (const m of cont) {
        acc.push({ from, to: m.to, captured: [enemySq, ...m.captured], promote: false });
      }
    } else {
      acc.push({ from, to: landing, captured: [enemySq], promote: false });
    }
  }
}

// รวม “ตากินทั้งหมด” ของฝั่งที่กำลังเดิน
function genAllCaptures(p: Position): Move[] {
  const moves: Move[] = [];
  const myMen   = p.side === 1 ? p.p1Men   : p.p2Men;
  const myKings = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen   = p.side === 1 ? p.p2Men   : p.p1Men;
  const opKings = p.side === 1 ? p.p2Kings : p.p1Kings;

  for (const from of bits(myMen))   genManCapturesFrom(p, from, myMen, myKings, opMen, opKings, [], moves);
  for (const from of bits(myKings)) genKingCapturesFrom(p, from, myMen, myKings, opMen, opKings, [], moves);

  if (moves.length === 0) return moves;

  if (!ENFORCE_MAX_CAPTURE) return moves;

  // บังคับเลือก “เส้นที่กินได้มากสุด”
  let best = 0;
  for (const m of moves) if (m.captured.length > best) best = m.captured.length;
  return moves.filter(m => m.captured.length === best);
}

// รวม “เดินเฉยทั้งหมด” (กรณีไม่มีตากิน)
function genAllQuiets(p: Position): Move[] {
  const moves: Move[] = [];
  const myMen   = p.side === 1 ? p.p1Men   : p.p2Men;
  const myKings = p.side === 1 ? p.p1Kings : p.p2Kings;

  for (const from of bits(myMen))   genManQuiet(p, from, moves);
  for (const from of bits(myKings)) genKingQuiet(p, from, moves);

  return moves;
}

// ---------- API หลัก ----------
export function generateMoves(p: Position): Move[] {
  const caps = genAllCaptures(p);
  if (caps.length) return caps;  // ถ้ามีตากิน → ต้องกิน
  return genAllQuiets(p);
}

export function applyMove(p: Position, m: Move): Position {
  const meMenBit   = p.side === 1 ? p.p1Men   : p.p2Men;
  const meKingsBit = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMenBit   = p.side === 1 ? p.p2Men   : p.p1Men;
  const opKingsBit = p.side === 1 ? p.p2Kings : p.p1Kings;

  const fromBit = B1(m.from);
  const toBit   = B1(m.to);

  let nMyMen   = meMenBit;
  let nMyKings = meKingsBit;
  let nOpMen   = opMenBit;
  let nOpKings = opKingsBit;

  const isFromMan  = (meMenBit & fromBit) !== 0;
  const isFromKing = (meKingsBit & fromBit) !== 0;

  // ลบศัตรูที่ถูกกินทั้งหมด
  for (const c of m.captured) {
    const cb = B1(c);
    nOpMen   &= ~cb;
    nOpKings &= ~cb;
  }

  if (isFromMan) {
    nMyMen = (nMyMen & ~fromBit) | toBit;

    if (m.promote || willPromote(p.side, m.to)) {
      nMyMen   &= ~toBit;
      nMyKings |= toBit;
    }
  } else if (isFromKing) {
    nMyKings = (nMyKings & ~fromBit) | toBit;
  } else {
    return p; // safety guard
  }

  const np: Position = { ...p };
  if (p.side === 1) {
    np.p1Men   = nMyMen;
    np.p1Kings = nMyKings;
    np.p2Men   = nOpMen;
    np.p2Kings = nOpKings;
  } else {
    np.p2Men   = nMyMen;
    np.p2Kings = nMyKings;
    np.p1Men   = nOpMen;
    np.p1Kings = nOpKings;
  }
  np.side = (p.side === 1 ? -1 : 1) as 1 | -1;

  return np;
}
