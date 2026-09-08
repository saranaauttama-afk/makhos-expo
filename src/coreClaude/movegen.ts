// src/coreClaude/movegen.ts
// Optimized Thai Checkers move generator (Phase 3).
// Keeps old API, adds generateMovesInto / generateCapturesInto.

import { BB, B1, bits, NEXT, RAYS } from './bitboards';
import { Position, occupied, sideKings, sideMen } from './position';

export interface Move {
  from: number;
  to: number;
  captured: number[];
  promote: boolean;
  path?: number[];
}

const UL = 0;
const UR = 1;
const DL = 2;
const DR = 3;

const P1_DIR_A = UL;
const P1_DIR_B = UR;
const P2_DIR_A = DL;
const P2_DIR_B = DR;

const LAST_RANK_P1 = 0b00000000000000000000000000001111 >>> 0; // 0..3
const LAST_RANK_P2 = 0b11110000000000000000000000000000 >>> 0; // 28..31

const EMPTY_CAPTURED: number[] = Object.freeze([]) as unknown as number[];
const MAX_CHAIN = 16;

function willPromote(side: 1 | -1, to: number): boolean {
  const bit = B1(to);
  return side === 1 ? (LAST_RANK_P1 & bit) !== 0 : (LAST_RANK_P2 & bit) !== 0;
}

function pushQuiet(out: Move[], from: number, to: number, promote: boolean): void {
  out.push({ from, to, captured: EMPTY_CAPTURED, promote });
}

function pushCapture(
  out: Move[],
  from: number,
  to: number,
  promote: boolean,
  caps: Int8Array,
  path: Int8Array,
  depth: number,
): void {
  const captured = new Array<number>(depth);
  const movePath = new Array<number>(depth);

  for (let i = 0; i < depth; i++) {
    captured[i] = caps[i];
    movePath[i] = path[i];
  }

  out.push({
    from,
    to,
    captured,
    promote,
    path: movePath,
  });
}

export function applyMove(p: Position, m: Move): Position {
  const fromBit = B1(m.from);
  const toBit = B1(m.to);
  const isCapture = m.captured.length > 0;

  let p1Men = p.p1Men;
  let p1Kings = p.p1Kings;
  let p2Men = p.p2Men;
  let p2Kings = p.p2Kings;

  if (p.side === 1) {
    const movingKing = (p1Kings & fromBit) !== 0;

    if (movingKing) {
      p1Kings = ((p1Kings & ~fromBit) | toBit) >>> 0;
    } else {
      p1Men = ((p1Men & ~fromBit) | toBit) >>> 0;
    }

    for (let i = 0; i < m.captured.length; i++) {
      const cb = B1(m.captured[i]);
      if (p2Men & cb) p2Men = (p2Men & ~cb) >>> 0;
      else p2Kings = (p2Kings & ~cb) >>> 0;
    }

    if (m.promote && !movingKing) {
      p1Men = (p1Men & ~toBit) >>> 0;
      p1Kings = (p1Kings | toBit) >>> 0;
    }
  } else {
    const movingKing = (p2Kings & fromBit) !== 0;

    if (movingKing) {
      p2Kings = ((p2Kings & ~fromBit) | toBit) >>> 0;
    } else {
      p2Men = ((p2Men & ~fromBit) | toBit) >>> 0;
    }

    for (let i = 0; i < m.captured.length; i++) {
      const cb = B1(m.captured[i]);
      if (p1Men & cb) p1Men = (p1Men & ~cb) >>> 0;
      else p1Kings = (p1Kings & ~cb) >>> 0;
    }

    if (m.promote && !movingKing) {
      p2Men = (p2Men & ~toBit) >>> 0;
      p2Kings = (p2Kings | toBit) >>> 0;
    }
  }

  return {
    side: p.side === 1 ? -1 : 1,
    p1Men,
    p1Kings,
    p2Men,
    p2Kings,
    halfmoveClock: isCapture ? 0 : p.halfmoveClock + 1,
  };
}

export function generateMoves(p: Position): Move[] {
  return generateMovesInto(p, []);
}

export function generateMovesInto(p: Position, out: Move[]): Move[] {
  generateCapturesInto(p, out);
  if (out.length > 0) return out;

  const occ = occupied(p);
  const empty = (~occ) >>> 0;
  const men = sideMen(p);
  const kings = sideKings(p);

  if (p.side === 1) {
    addMenQuietMoves(p.side, men, empty, out, P1_DIR_A, P1_DIR_B);
  } else {
    addMenQuietMoves(p.side, men, empty, out, P2_DIR_A, P2_DIR_B);
  }

  addKingQuietMoves(kings, occ, out);
  return out;
}

export function generateCapturesInto(p: Position, out: Move[]): Move[] {
  out.length = 0;

  const men = sideMen(p);
  const kings = sideKings(p);

  for (const from of bits(men)) {
    genMenCapturesFromFast(p, from, out);
  }

  for (const from of bits(kings)) {
    genKingCapturesFromFast(p, from, out);
  }

  // Makhos has compulsory capture but no global majority-capture priority.
  // Each DFS result is already a complete sequence (it is emitted only when
  // that moving piece has no further capture), so preserve every such result.
  return out;
}

function addMenQuietMoves(
  side: 1 | -1,
  men: BB,
  empty: BB,
  out: Move[],
  dirA: number,
  dirB: number,
): void {
  for (const from of bits(men)) {
    const toA = NEXT[from][dirA];
    if (toA >= 0 && (empty & B1(toA))) {
      pushQuiet(out, from, toA, willPromote(side, toA));
    }

    const toB = NEXT[from][dirB];
    if (toB >= 0 && (empty & B1(toB))) {
      pushQuiet(out, from, toB, willPromote(side, toB));
    }
  }
}

function addKingQuietMoves(kings: BB, occ: BB, out: Move[]): void {
  for (const from of bits(kings)) {
    for (let d = 0; d < 4; d++) {
      const ray = RAYS[from][d];

      for (let i = 0; i < ray.length; i++) {
        const to = ray[i];
        if (occ & B1(to)) break;
        pushQuiet(out, from, to, false);
      }
    }
  }
}

export function hasCapturesAvailable(p: Position): boolean {
  const occ = occupied(p);
  const myMen = sideMen(p);
  const myKings = sideKings(p);
  const myAll = (myMen | myKings) >>> 0;
  const opAll = p.side === 1
    ? (p.p2Men | p.p2Kings) >>> 0
    : (p.p1Men | p.p1Kings) >>> 0;

  const empty = (~occ) >>> 0;
  const dirA = p.side === 1 ? P1_DIR_A : P2_DIR_A;
  const dirB = p.side === 1 ? P1_DIR_B : P2_DIR_B;

  for (const from of bits(myMen)) {
    if (hasMenCaptureFrom(from, dirA, opAll, empty)) return true;
    if (hasMenCaptureFrom(from, dirB, opAll, empty)) return true;
  }

  for (const from of bits(myKings)) {
    for (let d = 0; d < 4; d++) {
      const ray = RAYS[from][d];
      let seenEnemy = false;

      for (let i = 0; i < ray.length; i++) {
        const sq = ray[i];
        const bit = B1(sq);

        if (myAll & bit) break;

        if (opAll & bit) {
          if (seenEnemy) break;
          seenEnemy = true;
          continue;
        }

        if (seenEnemy) return true;
      }
    }
  }

  return false;
}

function hasMenCaptureFrom(from: number, dir: number, opAll: BB, empty: BB): boolean {
  const over = NEXT[from][dir];
  if (over < 0) return false;
  if (!(opAll & B1(over))) return false;

  const landing = NEXT[over][dir];
  if (landing < 0) return false;

  return (empty & B1(landing)) !== 0;
}

function genMenCapturesFromFast(p: Position, from: number, out: Move[]): void {
  const caps = new Int8Array(MAX_CHAIN);
  const path = new Int8Array(MAX_CHAIN);

  const myMen0 = p.side === 1 ? p.p1Men : p.p2Men;
  const myKings0 = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen0 = p.side === 1 ? p.p2Men : p.p1Men;
  const opKings0 = p.side === 1 ? p.p2Kings : p.p1Kings;

  const dirA = p.side === 1 ? P1_DIR_A : P2_DIR_A;
  const dirB = p.side === 1 ? P1_DIR_B : P2_DIR_B;

  function dfs(
    cur: number,
    depth: number,
    myMen: BB,
    myKings: BB,
    opMen: BB,
    opKings: BB,
  ): void {
    let extended = false;

    extended = tryMenCaptureDir(cur, depth, dirA, myMen, myKings, opMen, opKings) || extended;
    extended = tryMenCaptureDir(cur, depth, dirB, myMen, myKings, opMen, opKings) || extended;

    if (!extended && depth > 0) {
      const to = path[depth - 1];
      pushCapture(out, from, to, willPromote(p.side, to), caps, path, depth);
    }
  }

  function tryMenCaptureDir(
    cur: number,
    depth: number,
    dir: number,
    myMen: BB,
    myKings: BB,
    opMen: BB,
    opKings: BB,
  ): boolean {
    if (depth >= MAX_CHAIN) return false;

    const over = NEXT[cur][dir];
    if (over < 0) return false;

    const overBit = B1(over);
    if (!((opMen | opKings) & overBit)) return false;

    const landing = NEXT[over][dir];
    if (landing < 0) return false;

    const landingBit = B1(landing);
    const occNow = (myMen | myKings | opMen | opKings) >>> 0;

    if (occNow & landingBit) return false;

    const fromBit = B1(cur);
    let myMenN = myMen;
    let myKingsN = myKings;
    let opMenN = opMen;
    let opKingsN = opKings;

    if (myKingsN & fromBit) {
      myKingsN = ((myKingsN & ~fromBit) | landingBit) >>> 0;
    } else {
      myMenN = ((myMenN & ~fromBit) | landingBit) >>> 0;
    }

    if (opKingsN & overBit) opKingsN = (opKingsN & ~overBit) >>> 0;
    else opMenN = (opMenN & ~overBit) >>> 0;

    caps[depth] = over;
    path[depth] = landing;

    dfs(landing, depth + 1, myMenN, myKingsN, opMenN, opKingsN);
    return true;
  }

  dfs(from, 0, myMen0, myKings0, opMen0, opKings0);
}

function genKingCapturesFromFast(p: Position, from: number, out: Move[]): void {
  const caps = new Int8Array(MAX_CHAIN);
  const path = new Int8Array(MAX_CHAIN);

  const myMen0 = p.side === 1 ? p.p1Men : p.p2Men;
  const myKings0 = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen0 = p.side === 1 ? p.p2Men : p.p1Men;
  const opKings0 = p.side === 1 ? p.p2Kings : p.p1Kings;

  if (!(myKings0 & B1(from))) return;

  function dfs(
    cur: number,
    depth: number,
    myMen: BB,
    myKings: BB,
    opMen: BB,
    opKings: BB,
  ): void {
    let extended = false;

    for (let d = 0; d < 4; d++) {
      if (tryKingCaptureDir(cur, depth, d, myMen, myKings, opMen, opKings)) {
        extended = true;
      }
    }

    if (!extended && depth > 0) {
      const to = path[depth - 1];
      pushCapture(out, from, to, false, caps, path, depth);
    }
  }

  function tryKingCaptureDir(
    cur: number,
    depth: number,
    dir: number,
    myMen: BB,
    myKings: BB,
    opMen: BB,
    opKings: BB,
  ): boolean {
    if (depth >= MAX_CHAIN) return false;

    const ray = RAYS[cur][dir];
    let enemy = -1;

    for (let i = 0; i < ray.length; i++) {
      const sq = ray[i];
      const bit = B1(sq);

      if ((myMen | myKings) & bit) return false;

      if (enemy < 0) {
        if ((opMen | opKings) & bit) {
          enemy = sq;
        }
        continue;
      }

      // Thai-hos behavior from old code:
      // once an enemy is seen, landing must be the first empty square behind it.
      if ((myMen | myKings | opMen | opKings) & bit) return false;

      const landing = sq;
      const fromBit = B1(cur);
      const landingBit = B1(landing);
      const enemyBit = B1(enemy);

      let myMenN = myMen;
      let myKingsN = myKings;
      let opMenN = opMen;
      let opKingsN = opKings;

      myKingsN = ((myKingsN & ~fromBit) | landingBit) >>> 0;

      if (opKingsN & enemyBit) opKingsN = (opKingsN & ~enemyBit) >>> 0;
      else opMenN = (opMenN & ~enemyBit) >>> 0;

      caps[depth] = enemy;
      path[depth] = landing;

      dfs(landing, depth + 1, myMenN, myKingsN, opMenN, opKingsN);
      return true;
    }

    return false;
  }

  dfs(from, 0, myMen0, myKings0, opMen0, opKings0);
}
