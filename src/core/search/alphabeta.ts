// src/core/search/alphabeta.ts
// PVS + Aspiration + LMR + TT + Quiescence + Extensions (Endgame/Threat)
import { Move, generateMoves, applyMove } from '../movegen';
import { Position, isDrawByInactivity } from '../position';
import { evaluate } from '../eval';
import { TT, Bound } from './tt';
import { hashPosition } from './zobrist';
import { bitCount } from '../bitboards';

export interface SearchInfo { depth: number; score: number; nodes: number; pv: Move[]; }
export interface SearchResult { best?: Move; score: number; nodes: number; depth: number; }
type OnInfo = (info: SearchInfo) => void;

const INF = 1e9 | 0;
const MAX_PLY = 64;

// heuristics
const killers0 = new Int32Array(MAX_PLY).fill(-1);
const killers1 = new Int32Array(MAX_PLY).fill(-1);
const history = new Int32Array(32 * 32);

function keyMove(m: Move) { return (m.from << 5) | m.to; }
function sameMove(m: Move, key: number) { return key === ((m.from << 5) | m.to); }

export function iterativeDeepening(root: Position, timeMs: number, tt = new TT(), onInfo?: OnInfo): SearchResult {
  const deadline = Date.now() + timeMs;
  killers0.fill(-1); killers1.fill(-1); history.fill(0);

  let best: Move | undefined; let bestScore = 0; let nodes = 0; let reached = 0;
  let lastScore = 0; let haveLast = false;

  for (let depth = 1; depth <= 22; depth++) {
    let alpha = haveLast ? lastScore - 80 : -INF;
    let beta  = haveLast ? lastScore + 80 : +INF;

    let result;
    while (true) {
      const st = { nodes: 0 };
      result = searchRoot(root, depth, alpha, beta, tt, deadline, st, 0);
      nodes += st.nodes;

      if (Date.now() > deadline) break;
      if (result.score <= alpha) { alpha = Math.max(-INF, alpha - 160); continue; } // fail-low
      if (result.score >= beta)  { beta  = Math.min(+INF, beta  + 160); continue; } // fail-high
      break;
    }

    if (Date.now() > deadline) break;
    if (result.move) { best = result.move; bestScore = result.score; reached = depth; }
    lastScore = result.score; haveLast = true;

    onInfo?.({ depth, score: bestScore, nodes, pv: getPV(root, tt, 12) });
  }

  return { best, score: bestScore, nodes, depth: reached };
}


function clampDepth(parentDepth: number, childDepth: number): number {
  if (parentDepth <= 0) return 0;
  const maxChild = parentDepth - 1;
  if (childDepth > maxChild) return maxChild;
  if (childDepth < 0) return 0;
  return childDepth;
}

function searchRoot(pos: Position, depth: number, alpha: number, beta: number, tt: TT, deadline: number, acc: {nodes: number}, ply: number) {
  if (isDrawByInactivity(pos)) return { move: undefined as Move | undefined, score: 0 };
  const moves = generateMoves(pos);
  if (moves.length === 0) return { move: undefined as Move | undefined, score: -999999 + ply };

  const key = hashPosition(pos);
  const tthit = tt.get(key);
  const ttMove = tthit?.move ?? -1;

  const ordered = orderMoves(moves, ttMove, ply);

  let bestScore = -INF; let bestMove: Move | undefined;
  const a0 = alpha, b0 = beta;

  for (let i = 0; i < ordered.length; i++) {
    if (Date.now() > deadline) break;
    const m = ordered[i];
    const child = applyMove(pos, m);

    // === Extensions ===
    let d = depth - 1;

    // Endgame extension: เหลือหมากรวม ≤ 5 → ต่อความลึก
    const totalPieces = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    if (totalPieces <= 5) d++;

    // Threat extension: ถ้าตาถัดไปฝั่งตรงข้ามมีตากินบังคับ → ต่อความลึก
    if (d >= 0) {
      const oppMoves = generateMoves(child);
      if (oppMoves.some(mm => mm.captured.length > 0)) d++;
    }

    // PVS (+ LMR สำหรับ quiet ช้า ๆ)
    let depthToUse = d;
    const lateQuiet = (i >= 3 && d >= 2 && m.captured.length === 0);
    if (lateQuiet) depthToUse = d - 1;
    depthToUse = clampDepth(depth, depthToUse);

    let sc: number;
    if (i === 0) {
      sc = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1);
    } else {
      sc = -alphabeta(child, depthToUse, -(alpha + 1), -alpha, tt, deadline, acc, ply + 1);
      if (sc > alpha && sc < beta) {
        sc = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1);
      }
    }

    acc.nodes++;
    if (sc > bestScore) { bestScore = sc; bestMove = m; }
    if (sc > alpha) alpha = sc;
    if (alpha >= beta) {
      if (m.captured.length === 0) updateHeuristics(m, depth, ply);
      break;
    }
  }

  if (bestMove) {
    const entry = { key, depth, score: bestScore, move: keyMove(bestMove), bound: Bound.EXACT as Bound };
    if (bestScore <= a0) entry.bound = Bound.UPPER;
    else if (bestScore >= b0) entry.bound = Bound.LOWER;
    tt.put(entry);
  }
  return { move: bestMove, score: bestScore };
}

function alphabeta(pos: Position, depth: number, alpha: number, beta: number, tt: TT, deadline: number, acc: {nodes: number}, ply: number): number {
  if (isDrawByInactivity(pos)) return 0;
  if (ply >= MAX_PLY - 1) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);
  if (depth <= 0) return quiesce(pos, alpha, beta, deadline, acc, ply);

  const key = hashPosition(pos);
  const hit = tt.get(key);
  if (hit && hit.depth >= depth) {
    if (hit.bound === Bound.EXACT) return hit.score;
    if (hit.bound === Bound.LOWER && hit.score > alpha) alpha = hit.score;
    else if (hit.bound === Bound.UPPER && hit.score < beta) beta = hit.score;
    if (alpha >= beta) return hit.score;
  }

  const moves = generateMoves(pos);
  if (moves.length === 0) return -999999 + ply;

  const ordered = orderMoves(moves, hit?.move ?? -1, ply);

  let best = -INF;
  const a0 = alpha, b0 = beta;
  let bestKey = -1;

  for (let i = 0; i < ordered.length; i++) {
    if (Date.now() > deadline) break;
    const m = ordered[i];
    const child = applyMove(pos, m);

    // === Extensions ===
    let d = depth - 1;
    const totalPieces = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    if (totalPieces <= 5) d++;
    if (d >= 0) {
      const oppMoves = generateMoves(child);
      if (oppMoves.some(mm => mm.captured.length > 0)) d++;
    }

    // LMR สำหรับ quiet ที่มาช้า
    let depthToUse = d;
    const lateQuiet = (i >= 3 && d >= 2 && m.captured.length === 0);
    if (lateQuiet) depthToUse = d - 1;
    depthToUse = clampDepth(depth, depthToUse);

    let sc: number;
    if (i === 0) {
      sc = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1);
    } else {
      sc = -alphabeta(child, depthToUse, -(alpha + 1), -alpha, tt, deadline, acc, ply + 1);
      if (sc > alpha && depthToUse < depth - 1) {
        sc = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1);
      } else if (sc > alpha && sc < beta) {
        sc = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1);
      }
    }

    acc.nodes++;
    if (sc > best) { best = sc; bestKey = keyMove(m); }
    if (sc > alpha) alpha = sc;
    if (alpha >= beta) {
      if (m.captured.length === 0) updateHeuristics(m, depth, ply);
      break;
    }
  }

  const entry = { key, depth, score: best, move: bestKey, bound: Bound.EXACT as Bound };
  if (best <= a0) entry.bound = Bound.UPPER;
  else if (best >= b0) entry.bound = Bound.LOWER;
  tt.put(entry);

  return best;
}

function quiesce(pos: Position, alpha: number, beta: number, deadline: number, acc: {nodes: number}, ply: number): number {
  if (isDrawByInactivity(pos)) return 0;
  if (ply >= MAX_PLY - 1) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);

  let stand = evaluate(pos);
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

  // ขยายเฉพาะ “กิน”
  const caps = generateMoves(pos).filter(m => m.captured.length > 0);
  caps.sort((a,b) => b.captured.length - a.captured.length);

  for (const m of caps) {
    const child = applyMove(pos, m);
    const sc = -quiesce(child, -beta, -alpha, deadline, acc, ply + 1);
    acc.nodes++;
    if (sc >= beta) return sc;
    if (sc > alpha) alpha = sc;
  }
  return alpha;
}

function updateHeuristics(m: Move, depth: number, ply: number) {
  const mk = keyMove(m);
  if (killers0[ply] !== mk) { killers1[ply] = killers0[ply]; killers0[ply] = mk; }
  history[mk] += depth * depth;
}

function orderMoves(moves: Move[], ttKey: number, ply: number): Move[] {
  return moves
    .map(m => {
      const mk = keyMove(m);
      let s = 0;
      if (ttKey >= 0 && sameMove(m, ttKey)) s += 1_000_000;
      if (m.captured.length) s += 10000 * m.captured.length;
      if (mk === killers0[ply]) s += 5000;
      if (mk === killers1[ply]) s += 4000;
      s += history[mk] | 0;
      // ช่วยชอบโปรโมต
      if (!m.captured.length && (m as any).promote) s += 1500;
      return { m, s };
    })
    .sort((a,b) => b.s - a.s)
    .map(x => x.m);
}

// principal variation จาก TT
function getPV(pos: Position, tt: TT, maxLen = 12): Move[] {
  const pv: Move[] = [];
  let cur = pos;
  for (let i = 0; i < maxLen; i++) {
    const hit = tt.get(hashPosition(cur));
    if (!hit || hit.move == null || hit.move < 0) break;
    const from = (hit.move >> 5) & 31;
    const to = hit.move & 31;
    const cand = generateMoves(cur).find(m => m.from === from && m.to === to);
    if (!cand) break;
    pv.push(cand);
    cur = applyMove(cur, cand);
  }
  return pv;
}
