// src/core/position.ts
import { BB, B1, bitCount } from './bitboards';

export type Side = 1 | -1; // 1 = P1 (ด้านล่าง เดินขึ้น), -1 = P2 (ด้านบน เดินลง)

export interface Position {
  side: Side;
  p1Men: BB; p1Kings: BB;
  p2Men: BB; p2Kings: BB;
  halfmoveClock: number; // plies since last capture
}

export const EMPTY: Position = { side: 1, p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0 };

export function clone(p: Position): Position { return { ...p }; }

/** เริ่มเกมแบบ 8 ตัว/ฝั่ง: P2 = สองแถวบน (index 0..7), P1 = สองแถวล่าง (index 24..31) */
export function initialPosition(): Position {
  let p1Men = 0, p2Men = 0;

  // ด้านบน (P2): แถว r=0 และ r=1 -> index 0..7
  for (const i of [0,1,2,3, 4,5,6,7]) p2Men |= B1(i);

  // ด้านล่าง (P1): แถว r=6 และ r=7 -> index 24..31
  for (const i of [24,25,26,27, 28,29,30,31]) p1Men |= B1(i);

  return { side: 1, p1Men, p1Kings: 0, p2Men, p2Kings: 0, halfmoveClock: 0 };
}

export function occupied(p: Position): BB {
  return (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
}
export function isDrawByInactivity(p: Position): boolean {
  // กฎไม่มีการกิน: ไม่จับ 32 ตา → เสมอ
  if (p.halfmoveClock >= 32) return true;
  // กฎฮอสล้วน: เหลือแต่ king ทั้งกระดาน + ไม่จับ 16 ตา → เสมอ
  const allKings = p.p1Men === 0 && p.p2Men === 0;
  if (allKings && p.halfmoveClock >= 16) return true;
  return false;
}

export function sideMen(p: Position): BB { return p.side === 1 ? p.p1Men : p.p2Men; }
export function sideKings(p: Position): BB { return p.side === 1 ? p.p1Kings : p.p2Kings; }
export function oppMen(p: Position): BB { return p.side === 1 ? p.p2Men : p.p1Men; }
export function oppKings(p: Position): BB { return p.side === 1 ? p.p2Kings : p.p1Kings; }
export function isTerminal(p: Position): boolean {
  const myCount = bitCount(sideMen(p) | sideKings(p));
  const opCount = bitCount(oppMen(p) | oppKings(p));
  return myCount === 0 || opCount === 0;
}

