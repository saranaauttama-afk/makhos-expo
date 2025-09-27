// src/core/movegen.ts
// Thai Checkers movegen:
// - Men: step 1 forward diag; capture forward; forced capture; multi-capture chains
// - Kings (Hos): fly any distance onto captures but must land immediately behind the captured piece;
//   forced capture; multi-capture chains.

import { BB, B1, bits, STEPS } from './bitboards';
import { Position, occupied, sideMen, sideKings } from './position';

export interface Move {
  from: number;
  to: number;
  captured: number[]; // indices of captured squares (dark-square indices)
  promote: boolean;   // men only; kings never promote
}

// P1 เดินขึ้น (ไปด้านบน) → โปรโมตเมื่อถึงแถวบนสุด (dark 4 ช่องแรก: 0..3)
// P2 เดินลง (ไปด้านล่าง) → โปรโมตเมื่อถึงแถวล่างสุด (dark 4 ช่องท้าย: 28..31)
const LAST_RANK_P1 = new Set([0, 1, 2, 3]);
const LAST_RANK_P2 = new Set([28, 29, 30, 31]);

type Dir = 'UL' | 'UR' | 'DL' | 'DR';

// ---- helpers ---------------------------------------------------------------

function willPromote(side: 1 | -1, to: number): boolean {
  return side === 1 ? LAST_RANK_P1.has(to) : LAST_RANK_P2.has(to);
}

function nextInDir(from: number, dir: Dir): number {
  const st = STEPS[from].find(s => s.dir === dir);
  return st ? st.to : -1;
}

function *ray(from: number, dir: Dir): Iterable<number> {
  let cur = from;
  // walk outward until out of board
  while (true) {
    const nxt = nextInDir(cur, dir);
    if (nxt < 0) return;
    yield nxt;
    cur = nxt;
  }
}

// ---- public API ------------------------------------------------------------

export function applyMove(p: Position, m: Move): Position {
  const q: Position = { ...p };
  const myMen = p.side === 1 ? 'p1Men' : 'p2Men';
  const myKings = p.side === 1 ? 'p1Kings' : 'p2Kings';
  const opMen = p.side === 1 ? 'p2Men' : 'p1Men';
  const opKings = p.side === 1 ? 'p2Kings' : 'p1Kings';

  const fromBit = B1(m.from);
  const toBit = B1(m.to);

  const movingKing = ((q as any)[myKings] & fromBit) !== 0;

  if (movingKing) (q as any)[myKings] = ((q as any)[myKings] & ~fromBit) >>> 0;
  else            (q as any)[myMen]   = ((q as any)[myMen]   & ~fromBit) >>> 0;

  if (movingKing) (q as any)[myKings] = ((q as any)[myKings] | toBit) >>> 0;
  else            (q as any)[myMen]   = ((q as any)[myMen]   | toBit) >>> 0;

  for (const c of m.captured) {
    const cb = B1(c);
    if ((q as any)[opMen] & cb)   (q as any)[opMen]   = ((q as any)[opMen]   & ~cb) >>> 0;
    else                          (q as any)[opKings] = ((q as any)[opKings] & ~cb) >>> 0;
  }

  if (m.promote && !movingKing) {
    (q as any)[myMen]   = ((q as any)[myMen]   & ~toBit) >>> 0;
    (q as any)[myKings] = ((q as any)[myKings] | toBit)  >>> 0;
  }

  q.side = (p.side === 1 ? -1 : 1);
  q.halfmoveClock = m.captured.length > 0 ? 0 : p.halfmoveClock + 1;
  return q;
}

export function generateMoves(p: Position): Move[] {
  const occ = occupied(p);
  const emptyMask = (~occ) >>> 0;
  const myMenBB = sideMen(p);
  const myKingsBB = sideKings(p);

  const captures: Move[] = [];

  // 1) forced captures — Men
  for (const from of bits(myMenBB)) {
    genMenCapturesFrom(p, from, captures);
  }
  // 1) forced captures — Kings (flying)
  for (const from of bits(myKingsBB)) {
    genKingCapturesFrom(p, from, captures);
  }
  if (captures.length) return captures;

  // 2) quiet moves — Men (forward one step)
  const quiet: Move[] = [];
  for (const from of bits(myMenBB)) {
    for (const st of STEPS[from]) {
      if (p.side === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue; // P1 up only
      if (p.side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue; // P2 down only
      const toBit = B1(st.to);
      if (emptyMask & toBit) quiet.push({ from, to: st.to, captured: [], promote: willPromote(p.side, st.to) });
    }
  }

  // 2) quiet moves — Kings (fly any distance until blocked)
  for (const from of bits(myKingsBB)) {
    for (const dir of ['UL','UR','DL','DR'] as Dir[]) {
      for (const sq of ray(from, dir)) {
        const toBit = B1(sq);
        if (occ & toBit) break;        // blocked
        quiet.push({ from, to: sq, captured: [], promote: false });
      }
    }
  }

  return quiet;
}

// ---- Men captures (adjacent jump, forward only) ----------------------------

function genMenCapturesFrom(p: Position, from: number, out: Move[]) {
  const myMen0   = p.side === 1 ? p.p1Men   : p.p2Men;
  const myKings0 = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen0   = p.side === 1 ? p.p2Men   : p.p1Men;
  const opKings0 = p.side === 1 ? p.p2Kings : p.p1Kings;

  const path: number[] = [];
  const caps: number[] = [];

  function dfs(cur: number, myMen: BB, myKings: BB, opMen: BB, opKings: BB) {
    let extended = false;

    // adjacent jumps: look at next step, then step after (landing)
    for (const st of STEPS[cur]) {
      // men capture forward only
      if (p.side === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue; // P1 up only
      if (p.side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue; // P2 down only

      const over = st.to;
      if (over < 0) continue;

      const overBit = B1(over);
      const occNow  = (myMen | myKings | opMen | opKings) >>> 0;
      const isEnemy = ((opMen | opKings) & overBit) !== 0;
      if (!isEnemy) continue;

      const landing = nextInDir(over, st.dir);
      if (landing < 0) continue;
      const landingBit = B1(landing);
      const isEmptyLanding = ((~occNow) >>> 0) & landingBit;
      if (!isEmptyLanding) continue;

      // apply capture
      const fromBit = B1(cur);
      const capturedWasKing = (opKings & overBit) !== 0;

      let myMenN = myMen, myKingsN = myKings, opMenN = opMen, opKingsN = opKings;
      // move piece
      if ((myKingsN & fromBit) !== 0) {
        myKingsN = ((myKingsN & ~fromBit) | landingBit) >>> 0;
      } else {
        myMenN = ((myMenN & ~fromBit) | landingBit) >>> 0;
      }
      // remove captured
      if (capturedWasKing) opKingsN = (opKingsN & ~overBit) >>> 0;
      else                 opMenN   = (opMenN   & ~overBit) >>> 0;

      path.push(landing);
      caps.push(over);
      dfs(landing, myMenN, myKingsN, opMenN, opKingsN);
      path.pop();
      caps.pop();
      extended = true;
    }

    if (!extended && caps.length > 0) {
      const lastTo = path.length ? path[path.length - 1] : cur;
      const promote = willPromote(p.side, lastTo);
      out.push({ from, to: lastTo, captured: [...caps], promote });
    }
  }

  dfs(from, myMen0, myKings0, opMen0, opKings0);
}

// ---- King captures (flying) ------------------------------------------------

function genKingCapturesFrom(p: Position, from: number, out: Move[]) {
  const myMen0   = p.side === 1 ? p.p1Men   : p.p2Men;
  const myKings0 = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen0   = p.side === 1 ? p.p2Men   : p.p1Men;
  const opKings0 = p.side === 1 ? p.p2Kings : p.p1Kings;

  const path: number[] = [];
  const caps: number[] = [];

  function dfs(cur: number, myMen: BB, myKings: BB, opMen: BB, opKings: BB) {
    let extended = false;

    for (const dir of ['UL','UR','DL','DR'] as Dir[]) {
      // march outward until first enemy (no friendly allowed in-between)
      let seenEnemy = false;
      let enemyIdx = -1;

      // step squares one by one along the ray
      for (const sq of ray(cur, dir)) {
        const bit = B1(sq);
        const occNow = (myMen | myKings | opMen | opKings) >>> 0;

        const isMine   = ((myMen | myKings) & bit) !== 0;
        const isEnemy  = ((opMen | opKings) & bit) !== 0;
        const isEmpty  = ((~occNow) >>> 0 & bit) !== 0;

        if (isMine) break; // blocked by own piece

        if (!seenEnemy) {
          if (isEmpty) {
            // just an empty along the way before enemy; continue scanning
            continue;
          } else if (isEnemy) {
            seenEnemy = true;
            enemyIdx = sq;
            // continue to look for landing squares beyond the enemy
            continue;
          } else {
            // should not happen
            break;
          }
        } else {
          // already saw exactly one enemy; landing must be the first empty square behind it
          if (!isEmpty) break; // blocked after enemy
          const landing = sq;
          const fromBit = B1(cur);
          const landingBit = B1(landing);
          const enemyBit = B1(enemyIdx);
          const capturedWasKing = (opKings & enemyBit) !== 0;

          // apply capture
          let myMenN = myMen, myKingsN = myKings, opMenN = opMen, opKingsN = opKings;
          // always king moves here
          myKingsN = ((myKingsN & ~fromBit) | landingBit) >>> 0;

          if (capturedWasKing) opKingsN = (opKingsN & ~enemyBit) >>> 0;
          else                 opMenN   = (opMenN   & ~enemyBit) >>> 0;

          path.push(landing);
          caps.push(enemyIdx);
          dfs(landing, myMenN, myKingsN, opMenN, opKingsN);
          path.pop();
          caps.pop();

          extended = true;
          break; // only the immediate landing square is legal
        }
      }
    }

    if (!extended && caps.length > 0) {
      const lastTo = path.length ? path[path.length - 1] : cur;
      // king never promotes
      out.push({ from, to: lastTo, captured: [...caps], promote: false });
    }
  }

  // ensure the moving piece is actually a king at `from`
  const fromBit = B1(from);
  const isKing = (myKings0 & fromBit) !== 0;
  if (!isKing) return;

  dfs(from, myMen0, myKings0, opMen0, opKings0);
}


