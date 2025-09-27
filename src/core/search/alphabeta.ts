// src/core/search/alphabeta.ts
// PVS + Aspiration + LMR + TT + Quiescence
// + SAFE Extensions (per-line budget, dynamic for endgame)
// + Root finisher scan (2–3 ply, forced lines)
// + Mobility-drop bonus at root
// + Single-reply extension & disable-LMR in forced nodes
// + Tiny deterministic tiebreak in root ordering

import { Move, generateMoves, applyMove } from '../movegen';
import { Position } from '../position';
import { evaluate } from '../eval';
import { TT, Bound } from './tt';
import { hashPosition } from './zobrist';
import { bitCount } from '../bitboards';

export interface SearchInfo { depth: number; score: number; nodes: number; pv: Move[]; }
export interface SearchResult { best?: Move; score: number; nodes: number; depth: number; }
type OnInfo = (info: SearchInfo) => void;

const INF = 1e9 | 0;
const MAX_PLY = 96;

// ---------------------------------------------------------------------------
// Heuristics (killer / history)
const killers0 = new Int32Array(MAX_PLY).fill(-1);
const killers1 = new Int32Array(MAX_PLY).fill(-1);
const history  = new Int32Array(32 * 32);

function keyMove(m: Move) { return (m.from << 5) | m.to; }
function sameMove(m: Move, key: number) { return key === ((m.from << 5) | m.to); }

// ---------------------------------------------------------------------------
// Helpers (endgame check, finisher scan, mobility at root)
function kingsOnlyAndCount(p: Position) {
  const men = bitCount(p.p1Men | p.p2Men);
  const k1  = bitCount(p.p1Kings);
  const k2  = bitCount(p.p2Kings);
  return { kingsOnly: men === 0, ktotal: k1 + k2, k1, k2 };
}
function calcRootBudget(p: Position) {
  const { kingsOnly, ktotal } = kingsOnlyAndCount(p);
  return (kingsOnly && ktotal <= 3) ? 2 : 1; // endgame (kings-only ≤3) → งบ 2, อื่น ๆ 1
}

function oppPiecesBits(p: Position): number {
  return p.side === 1 ? (p.p2Men | p.p2Kings) : (p.p1Men | p.p1Kings);
}
function isImmediateWin(pos: Position): boolean {
  const oppLeft = bitCount(oppPiecesBits(pos));
  if (oppLeft === 0) return true;
  const oppMoves = generateMoves(pos);
  return oppMoves.length === 0;
}
function forcedMovesOnly(pos: Position): Move[] {
  const ms = generateMoves(pos);
  const caps = ms.filter(m => m.captured.length > 0);
  return caps.length ? caps : ms;
}
// ชนะภายใน 2 จังหวะ (เรา→เขา→เรา จบ) — ใช้เส้นบังคับก่อน
function isRootForcedWinInTwo(root: Position, m: Move): boolean {
  const afterMine = applyMove(root, m);
  if (isImmediateWin(afterMine)) return true;

  const opp1 = forcedMovesOnly(afterMine);
  for (const oc of opp1) {
    const afterOpp = applyMove(afterMine, oc);
    const my2 = forcedMovesOnly(afterOpp);
    let ok = false;
    for (const r of my2) {
      const fin = applyMove(afterOpp, r);
      if (isImmediateWin(fin)) { ok = true; break; }
    }
    if (!ok) return false;
  }
  return true;
}
// ชนะภายใน 3 จังหวะบังคับ (เรา→เขา→เรา→เขา จบ)
function isRootForcedWinInThree(root: Position, m: Move): boolean {
  const afterMine = applyMove(root, m);
  const opp1 = forcedMovesOnly(afterMine);
  for (const oc of opp1) {
    const afterOpp = applyMove(afterMine, oc);
    const my2 = forcedMovesOnly(afterOpp);

    let existReply = false;
    for (const r of my2) {
      const afterMy = applyMove(afterOpp, r);
      if (isImmediateWin(afterMy)) { existReply = true; break; }

      const opp2 = forcedMovesOnly(afterMy);
      let allLose = true;
      for (const oc2 of opp2) {
        const afterOpp2 = applyMove(afterMy, oc2);
        if (!isImmediateWin(afterOpp2)) { allLose = false; break; }
      }
      if (allLose) { existReply = true; break; }
    }
    if (!existReply) return false;
  }
  return true;
}

// Mobility-drop: เดินแล้วจำนวนทางของคู่แข่งลดลงมาก → ดี (โดยเฉพาะ kings-only)
function mobilityDropScore(root: Position, m: Move): number {
  const after = applyMove(root, m);
  const oppMoves = generateMoves(after).length;
  const { kingsOnly, ktotal } = kingsOnlyAndCount(after);
  const base = Math.max(0, 12 - oppMoves); // 12 เป็นค่าชี้วัดคร่าว ๆ
  if (kingsOnly && ktotal <= 3) return base * 6;
  if (kingsOnly) return base * 4;
  return base * 2;
}

// ---------------------------------------------------------------------------
// SAFE extension with per-line budget (+ single-reply extension)
function computeExtensionFlexible(args: {
  depth: number; budget: number;
  endgameSmall: boolean; oppHasCapture: boolean;
  parentHasSingle: boolean; childHasSingle: boolean;
}) {
  let d = args.depth - 1; // ลดลง 1 อย่างน้อยเสมอ
  let b = args.budget;

  // ต่อจากเหตุบังคับฝั่งเรา (parent มีทางเดียว)
  if (d > 0 && b > 0 && args.parentHasSingle) { d++; b--; }

  // ต่อจากเหตุปลายเกม/ตากินบังคับ/ลูกมีทางเดียว
  if (d > 0 && b > 0 && (args.endgameSmall || args.oppHasCapture || args.childHasSingle)) { d++; b--; }

  if (d > args.depth) d = args.depth;
  if (d < 0) d = 0;
  return { depth: d, budget: b };
}

// ---------------------------------------------------------------------------
// Main
export function iterativeDeepening(root: Position, timeMs: number, tt = new TT(), onInfo?: OnInfo): SearchResult {
  const deadline = Date.now() + timeMs;
  killers0.fill(-1); killers1.fill(-1); history.fill(0);

  let best: Move | undefined; let bestScore = 0; let nodes = 0; let reached = 0;

  let lastScore = 0; let haveLast = false;
  const ROOT_BUDGET = calcRootBudget(root);

  for (let depth = 1; depth <= 22; depth++) {
    let alpha = haveLast ? lastScore - 80 : -INF;
    let beta  = haveLast ? lastScore + 80 : +INF;

    let result;
    while (true) {
      const st = { nodes: 0 };
      result = searchRoot(root, depth, alpha, beta, tt, deadline, st, 0, ROOT_BUDGET);
      nodes += st.nodes;

      if (Date.now() > deadline) break;
      if (result.score <= alpha) { alpha = Math.max(-INF, alpha - 160); continue; }
      if (result.score >= beta)  { beta  = Math.min(+INF, beta  + 160); continue; }
      break;
    }

    if (Date.now() > deadline) break;
    if (result.move) { best = result.move; bestScore = result.score; reached = depth; }
    lastScore = result.score; haveLast = true;

    onInfo?.({ depth, score: bestScore, nodes, pv: getPV(root, tt, 12) });
  }
  return { best, score: bestScore, nodes, depth: reached };
}

function searchRoot(
  pos: Position, depth: number, alpha: number, beta: number,
  tt: TT, deadline: number, acc: {nodes: number}, ply: number, budget: number
) {
  const moves = generateMoves(pos);
  if (moves.length === 0) return { move: undefined as Move | undefined, score: -999999 + ply };

  // 0) Root finisher — ถ้าเจอ forced win ภายใน 2/3 จังหวะ เลือกทันที
  for (const m of moves) {
    if (isRootForcedWinInTwo(pos, m) || isRootForcedWinInThree(pos, m)) {
      return { move: m, score: 900_000 };
    }
  }

  const key = hashPosition(pos);
  const tthit = tt.get(key);
  const ttMove = tthit?.move ?? -1;

  // 1) จัดลำดับพื้นฐาน
  const base = orderMoves(moves, ttMove, ply);

  // 2) root ordering: finisher priority + mobility-drop + tiny tiebreak
// 2) root ordering: finisher priority + mobility-drop + (NEW) anti-suicide penalty + tiny tiebreak
const ordered = base
  .map(m => {
    let pr = 0;

    // --- ใช้ตัวแปรช่วยกันคำนวณซ้ำ
    const win2 = isRootForcedWinInTwo(pos, m);
    const win3 = !win2 && isRootForcedWinInThree(pos, m);

    // finisher priority
    if (win2)      pr += 1_000_000;
    else if (win3) pr +=   900_000;

    // mobility drop (บีบทางเดินคู่แข่ง)
    pr += mobilityDropScore(pos, m);

    // --- [เพิ่มตรงนี้] ลดอันดับเบา ๆ ถ้าเดินแล้วคู่แข่ง "มีตากินทันที" และไม่ใช่ตัวจบ
    const after = applyMove(pos, m);
    const oppHasCapNow = generateMoves(after).some(mm => mm.captured.length > 0);
    if (!win2 && !win3 && oppHasCapNow) {
      pr -= 200; // ปรับได้ 100–300 ตามชอบ
    }

    // tiny deterministic tiebreak
    const h = (hashPosition(pos) ^ ((m.from << 5) | m.to)) >>> 0;
    pr += (h & 7); // 0..7

    return { m, pr };
  })
  .sort((a,b) => b.pr - a.pr)
  .map(x => x.m);


  let bestScore = -INF; let bestMove: Move | undefined;
  const a0 = alpha, b0 = beta;

  for (let i = 0; i < ordered.length; i++) {
    if (Date.now() > deadline) break;
    const m = ordered[i];
    const child = applyMove(pos, m);

    // เตรียมข้อมูลสำหรับ extension/LMR
    const childMoves = generateMoves(child);
    const total = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    const oppHasCap = childMoves.some(mm => mm.captured.length > 0);
    const childHasSingle = (childMoves.length === 1);
    const parentHasSingle = (ordered.length === 1);
    const endgameSmall = (total <= 5);

    const ext = computeExtensionFlexible({
      depth, budget,
      endgameSmall, oppHasCapture: oppHasCap,
      parentHasSingle, childHasSingle
    });
    let d = ext.depth;
    let nextBudget = ext.budget;

    // LMR: ลดเฉพาะ quiet ที่มาช้า และไม่ใช่โหนดบังคับ
    const isQuiet = (m.captured.length === 0);
    const disableLMR = (ordered.length <= 2) || childHasSingle;
    const lateQuiet = (i >= 3 && d >= 2 && isQuiet && !disableLMR);
    if (lateQuiet) d = Math.max(0, d - 1);

    let sc: number;
    if (i === 0) {
      sc = -alphabeta(child, d, -beta, -alpha, tt, deadline, acc, ply + 1, nextBudget);
    } else {
      sc = -alphabeta(child, d, -(alpha + 1), -alpha, tt, deadline, acc, ply + 1, nextBudget);
      if (sc > alpha && sc < beta) {
        sc = -alphabeta(child, d, -beta, -alpha, tt, deadline, acc, ply + 1, nextBudget);
      }
    }

    // บูสต์เล็กน้อยเฉพาะ root
    if (isRootForcedWinInTwo(pos, m) || isRootForcedWinInThree(pos, m)) sc += 500;
    sc += Math.min(100, mobilityDropScore(pos, m));

    acc.nodes++;
    if (sc > bestScore) { bestScore = sc; bestMove = m; }
    if (sc > alpha) alpha = sc;
    if (alpha >= beta) {
      if (isQuiet) updateHeuristics(m, depth, ply);
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

function alphabeta(
  pos: Position, depth: number, alpha: number, beta: number,
  tt: TT, deadline: number, acc: {nodes: number}, ply: number, budget: number
): number {
  if (ply >= MAX_PLY) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);
  if (depth <= 0) return quiesce(pos, alpha, beta, deadline, acc, ply);

  // TT probe
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

    const childMoves = generateMoves(child);
    const total = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    const oppHasCap = childMoves.some(mm => mm.captured.length > 0);
    const childHasSingle = (childMoves.length === 1);
    const parentHasSingle = (ordered.length === 1);
    const endgameSmall = (total <= 5);

    const ext = computeExtensionFlexible({
      depth, budget,
      endgameSmall, oppHasCapture: oppHasCap,
      parentHasSingle, childHasSingle
    });
    let d = ext.depth;
    let nextBudget = ext.budget;

    // LMR: เฉพาะ quiet ช้า และไม่ใช่โหนดบังคับ
    const isQuiet = (m.captured.length === 0);
    const disableLMR = (ordered.length <= 2) || childHasSingle;
    const lateQuiet = (i >= 3 && d >= 2 && isQuiet && !disableLMR);
    if (lateQuiet) d = Math.max(0, d - 1);

    let sc: number;
    if (i === 0) {
      sc = -alphabeta(child, d, -beta, -alpha, tt, deadline, acc, ply + 1, nextBudget);
    } else {
      sc = -alphabeta(child, d, -(alpha + 1), -alpha, tt, deadline, acc, ply + 1, nextBudget);
      if (sc > alpha && d < depth - 1) {
        sc = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1, nextBudget);
      } else if (sc > alpha && sc < beta) {
        sc = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1, nextBudget);
      }
    }

    acc.nodes++;
    if (sc > best) { best = sc; bestKey = keyMove(m); }
    if (sc > alpha) alpha = sc;
    if (alpha >= beta) {
      if (isQuiet) updateHeuristics(m, depth, ply);
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
  if (ply >= MAX_PLY) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);

  let stand = evaluate(pos);
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

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
      if (!m.captured.length && (m as any).promote) s += 1500;
      return { m, s };
    })
    .sort((a,b) => b.s - a.s)
    .map(x => x.m);
}

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
