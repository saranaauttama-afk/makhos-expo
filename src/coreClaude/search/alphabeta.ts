// alphabeta.ts — clean Negamax + Alpha-Beta + Iterative Deepening
// Fixes vs original:
//   1. Time check every 512 nodes (not every node)
//   2. LMR with proper log-scaled table
//   3. Aspiration window ±150, doubles on fail
//   4. Delta pruning in quiescence
//   5. History aging between searches

import { Move, applyMove, generateMoves, hasCapturesAvailable } from '../movegen';
import { B1, bitCount, toRC } from '../bitboards';
import { evaluate } from '../eval';
import { isDrawByInactivity, Position } from '../position';
import { Bound, TT } from './tt';
import { probeSmallEndgame } from './endgameTablebase';
import {
  buildRepetitionCounts, getRepetitionCount, isThreefoldRepetition,
  popRepetition, pushRepetition, RepetitionCounts,
} from './repetition';
import { hashPosition } from './zobrist';

export interface SearchInfo  { depth: number; score: number; nodes: number; pv: Move[]; }
export interface SearchResult { best?: Move; score: number; nodes: number; depth: number; }
export interface CancelToken  { cancelled: boolean; }
type OnInfo = (info: SearchInfo) => void;

// Eval override — allows A/B testing without changing all call sites
let _eval: (p: Position) => number = evaluate;
export function setEvalFn(fn: (p: Position) => number): void { _eval = fn; }
export function resetEvalFn(): void { _eval = evaluate; }

const INF        = 1_000_000;
const MAX_PLY    = 64;
const TC_MASK    = 511; // check time every 512 nodes

// Shared stop flag — set when deadline fires; cleared before each iterativeDeepening.
// All levels check this at the top of negamax so the search unwinds immediately
// once time is up, rather than waiting for each level's own TC_MASK checkpoint.
const stop = { flag: false };

const killers0    = new Int32Array(MAX_PLY).fill(-1);
const killers1    = new Int32Array(MAX_PLY).fill(-1);
const history     = new Int32Array(1024); // (from<<5|to) max = 31*32+31 = 1023
// Countermove table: for each move key, stores the quiet move that most recently
// caused a beta cutoff in response to it.  Improves move ordering ~10% nodes.
const counterMove = new Int32Array(1024).fill(-1);

// LMR reduction table [moveIdx][depth]
const LMR: Uint8Array[] = Array.from({ length: 32 }, (_, i) =>
  new Uint8Array(32).map((_, d) =>
    i < 3 || d < 2 ? 0 : Math.min(3, Math.floor(0.5 + Math.log(i+1) * Math.log(d+1) / 2.2))
  )
);

function key(m: Move) { return (m.from << 5) | m.to; }

// prevKey = key of the move that led to this position (-1 at root)
function orderMoves(pos: Position, moves: Move[], ttMove: number, ply: number, prevKey = -1): Move[] {
  const cm = prevKey >= 0 ? counterMove[prevKey] : -1; // look up countermove
  return moves.map(m => {
    const k = key(m);
    let s = 0;
    if (k === ttMove)        s += 2_000_000;
    if (m.captured.length)   s += 100_000 + m.captured.length * 10_000;
    if (k === killers0[ply]) s += 8_000;
    if (k === killers1[ply]) s += 7_000;
    if (k === cm)            s += 6_000; // countermove: good response to opponent's last move
    s += history[k] | 0;
    // static order: forward advance bonus for men
    if (!m.captured.length) {
      const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
      if (!(myKings & B1(m.from))) {
        const { r: rf } = toRC(m.from), { r: rt } = toRC(m.to);
        s += (pos.side === 1 ? rf - rt : rt - rf) * 20;
      }
    }
    return { m, s };
  }).sort((a, b) => b.s - a.s).map(x => x.m);
}

function updateKillers(m: Move, ply: number) {
  const k = key(m);
  if (killers0[ply] !== k) { killers1[ply] = killers0[ply]; killers0[ply] = k; }
}

function getPV(pos: Position, tt: TT, max = 10): Move[] {
  const pv: Move[] = []; let cur = pos;
  for (let i = 0; i < max; i++) {
    const hit = tt.get(hashPosition(cur)); if (!hit || hit.move == null) break;
    const mv = generateMoves(cur).find(m => key(m) === hit.move!); if (!mv) break;
    pv.push(mv); cur = applyMove(cur, mv);
  }
  return pv;
}

// ── Quiescence ───────────────────────────────────────────────────────────────
function quiesce(
  pos: Position, alpha: number, beta: number,
  deadline: number, acc: {n:number}, ply: number
): number {
  if (stop.flag) return _eval(pos);
  if (isDrawByInactivity(pos)) return 0;
  if (ply >= MAX_PLY) return _eval(pos);

  const stand = _eval(pos);
  if (stand >= beta) return beta;
  if (stand + 300 < alpha) return alpha; // delta pruning
  if (stand > alpha) alpha = stand;

  const caps = generateMoves(pos).filter(m => m.captured.length > 0);
  caps.sort((a, b) => b.captured.length - a.captured.length);

  for (const m of caps) {
    if (stop.flag) break;
    if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) { stop.flag = true; break; }
    acc.n++;
    const score = -quiesce(applyMove(pos, m), -beta, -alpha, deadline, acc, ply+1);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ── Negamax ──────────────────────────────────────────────────────────────────
function negamax(
  pos: Position, depth: number, alpha: number, beta: number, tt: TT,
  deadline: number, acc: {n:number}, ply: number, rep: RepetitionCounts,
  nullOk = true, prevMoveKey = -1, iidOk = true,
): number {
  const h = hashPosition(pos);
  if (isDrawByInactivity(pos) || isThreefoldRepetition(rep, h)) return 0;

  // NOTE: endgame tablebase probe intentionally removed from hot path —
  // probeSmallEndgameFromCounts triggers full retrograde DFS for every node
  // in king endgames, causing catastrophic slowness (226s+ per move).
  // The root-level probe in iterativeDeepening handles endgame positions.

  if (ply >= MAX_PLY) return _eval(pos);
  if (stop.flag) return _eval(pos);
  if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) { stop.flag = true; return _eval(pos); }
  if (depth <= 0) return quiesce(pos, alpha, beta, deadline, acc, ply);

  // TT probe
  const hit = tt.get(h);
  let ttMove = hit?.move ?? -1; // let — may be updated by IID below
  if (hit && hit.depth >= depth && getRepetitionCount(rep, h) <= 1) {
    if (hit.bound === Bound.EXACT) return hit.score;
    if (hit.bound === Bound.LOWER) alpha = Math.max(alpha, hit.score);
    else                           beta  = Math.min(beta,  hit.score);
    if (alpha >= beta) return hit.score;
  }

  const moves = generateMoves(pos);
  if (!moves.length) return -INF + ply;

  // In Thai Checkers, forced-capture rule means either ALL moves are captures
  // or ALL moves are quiet.  isQuiet === true means no captures are available.
  const isQuiet = moves[0].captured.length === 0;

  // ── Pruning (not at root, not in repeated positions) ─────────────────────
  if (ply > 0 && getRepetitionCount(rep, h) <= 1) {

    if (isQuiet) {
      const se = _eval(pos);

      // Reverse Futility Pruning (Static Null Move):
      // If static eval is way above beta even after subtracting a depth-scaled
      // margin, our position is so good the opponent will avoid this line.
      if (depth <= 4 && se - 120 * depth >= beta) return se;

      // Razoring:
      // If static eval is way below alpha even after adding a generous margin,
      // drop to quiescence — the position is likely a dead loss for us.
      if (depth <= 2) {
        const margin = depth === 1 ? 350 : 550;
        if (se + margin < alpha) {
          const q = quiesce(pos, alpha - 1, alpha, deadline, acc, ply);
          if (q < alpha) return q;
        }
      }

      // Null Move Pruning:
      // Skip our turn and let the opponent move twice. If the position is still
      // >= beta, we can prune — our position is too good to refute.
      // Only in quiet nodes (can't pass on forced captures), not near mate.
      if (nullOk && depth >= 3 && beta < INF - MAX_PLY && se >= beta) {
        const R = depth >= 6 ? 3 : 2;
        const nullPos: Position = { ...pos, side: (-pos.side) as 1 | -1 };
        // Don't push nullPos to rep — it's a synthetic position, not a real game state
        const s = -negamax(nullPos, depth - R - 1, -beta, -(beta - 1), tt, deadline, acc, ply + 1, rep, false);
        if (s >= beta) return beta;
      }
    }

    // Probcut:
    // At deep nodes with forced captures, try the top-3 captures at a much
    // shallower depth with a wide beta.  If any scores >= pcBeta, we can
    // safely apply a full beta cutoff without searching deeper.
    if (!isQuiet && depth >= 5) {
      const pcBeta  = Math.min(INF - ply, beta + 200);
      const pcDepth = depth - 4;
      // Linear top-3 scan — no allocation, no sort
      let tried = 0;
      let best0 = -1, best1 = -1, best2 = -1; // indices into moves[]
      for (let mi = 0; mi < moves.length; mi++) {
        const cl = moves[mi].captured.length;
        if (best0 < 0 || cl > moves[best0].captured.length) { best2 = best1; best1 = best0; best0 = mi; }
        else if (best1 < 0 || cl > moves[best1].captured.length) { best2 = best1; best1 = mi; }
        else if (best2 < 0 || cl > moves[best2].captured.length) { best2 = mi; }
      }
      const pcIndices = [best0, best1, best2].filter(x => x >= 0);
      for (const mi of pcIndices) {
        const m = moves[mi];
        if (tried >= 3) break;
        if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) break;
        const child = applyMove(pos, m);
        const ch    = hashPosition(child);
        pushRepetition(rep, ch);
        acc.n++;
        tried++;
        const s = -negamax(child, pcDepth, -pcBeta, -(pcBeta - 1), tt, deadline, acc, ply + 1, rep, true, key(m));
        popRepetition(rep, ch);
        if (s >= pcBeta) return beta; // Probcut — fail high
      }
    }
  }

  // Internal Iterative Deepening (IID):
  // No TT move at a deep node means bad move ordering — the first move tried
  // will likely fail to cut off.  Run a quick depth-2 search to populate the
  // TT so we get a good move to try first.  Cost is small; benefit is large.
  // IID: at PV nodes (full window) only — null-window nodes have many siblings
  // so IID cost is not worth it.  Cap at depth 4 to keep the sub-search cheap.
  // iidOk=false on the sub-search prevents cascading IID.
  const isPV = beta > alpha + 1;
  if (ttMove === -1 && depth >= 5 && ply > 0 && isPV && iidOk && Date.now() <= deadline) {
    negamax(pos, Math.min(depth - 2, 4), alpha, beta, tt, deadline, acc, ply, rep, false, prevMoveKey, false);
    const iidHit = tt.get(h);
    if (iidHit?.move != null) ttMove = iidHit.move;
  }

  const ordered = orderMoves(pos, moves, ttMove, ply, prevMoveKey);
  let best = -INF, bestKey = -1;
  const a0 = alpha, b0 = beta;

  for (let i = 0; i < ordered.length; i++) {
    if (stop.flag) break;
    if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) { stop.flag = true; break; }
    const m     = ordered[i];
    const child = applyMove(pos, m);
    const ch    = hashPosition(child);
    const isQ   = m.captured.length === 0;
    const single = ordered.length === 1;
    const total  = bitCount(child.p1Men|child.p1Kings|child.p2Men|child.p2Kings);

    // Extensions
    let d = depth - 1;
    if (single) d = Math.min(depth, d+1);       // our only move — extend
    if (total <= 5) d = Math.min(depth, d+1);   // endgame ext

    // LMR — skip if opponent will have forced captures (sacrifice/tactic position).
    // hasCapturesAvailable is O(pieces) vs full generateMoves, avoiding double movegen.
    const opHasCaptures = isQ && hasCapturesAvailable(child);
    if (i >= 3 && d >= 2 && isQ && !single && !opHasCaptures) {
      d = Math.max(1, d - (LMR[Math.min(31,i)][Math.min(31,d)] | 0));
    }

    // Late Move Pruning (LMP): at very shallow depth, stop searching quiet
    // moves beyond a threshold — they're very unlikely to raise alpha.
    if (isQ && !single && depth <= 2 && i >= (depth === 1 ? 6 : 10) && alpha > -INF + MAX_PLY) break;

    pushRepetition(rep, ch);
    const mk = key(m); // move key — passed as prevMoveKey to child nodes
    let score: number;
    if (i === 0) {
      score = -negamax(child, d, -beta, -alpha, tt, deadline, acc, ply+1, rep, true, mk);
    } else {
      score = -negamax(child, d, -(alpha+1), -alpha, tt, deadline, acc, ply+1, rep, true, mk);
      if (score > alpha && score < beta) {
        score = -negamax(child, depth-1, -beta, -alpha, tt, deadline, acc, ply+1, rep, true, mk);
      }
    }
    popRepetition(rep, ch);
    acc.n++;

    if (score > best) { best = score; bestKey = mk; }
    if (score > alpha) { alpha = score; }
    if (alpha >= beta) {
      if (isQ) {
        updateKillers(m, ply);
        history[mk] = Math.min(30000, history[mk] + depth * depth);
        // Countermove: remember that this quiet move countered prevMoveKey well
        if (prevMoveKey >= 0) counterMove[prevMoveKey] = mk;
      }
      break;
    }
  }

  const bound: Bound = best <= a0 ? Bound.UPPER : best >= b0 ? Bound.LOWER : Bound.EXACT;
  if (getRepetitionCount(rep, h) <= 1)
    tt.put({ key: h, depth, score: best, move: bestKey >= 0 ? bestKey : undefined, bound });
  return best;
}

// ── Iterative Deepening ──────────────────────────────────────────────────────
export async function iterativeDeepening(
  root: Position, timeMs: number, tt = new TT(), onInfo?: OnInfo,
  historyHashes: number[] = [], cancel?: CancelToken,
): Promise<SearchResult> {
  const deadline  = Date.now() + timeMs;
  const rootHash  = hashPosition(root);
  const normHist  = historyHashes.length ? historyHashes : [rootHash];
  const rep       = buildRepetitionCounts(normHist);

  killers0.fill(-1); killers1.fill(-1);
  counterMove.fill(-1);                                       // reset per-search
  for (let i = 0; i < history.length; i++) history[i] >>= 1; // age history
  stop.flag = false;                                          // clear stop flag

  if (isThreefoldRepetition(rep, rootHash))
    return { best: undefined, score: 0, nodes: 0, depth: 0 };

  // Cap probe so it never eats into the search budget.
  // Default maxMs=3000 could exceed timeMs entirely, leaving no time for search.
  const probeMs = Math.min(500, Math.floor(timeMs * 0.3));
  const eg = probeSmallEndgame(root, normHist, probeMs);
  // Only shortcut when we have an actual move — draw positions at the depth
  // limit store bestMoveKey = NO_MOVE_KEY so eg.best would be undefined.
  // Fall through to regular search so the engine still picks a legal move.
  if (eg?.best) {
    onInfo?.({ depth: eg.dtm, score: eg.score, nodes: 0, pv: [eg.best] });
    return { best: eg.best, score: eg.score, nodes: 0, depth: eg.dtm };
  }

  let best: Move | undefined, bestScore = 0, nodes = 0, reached = 0;
  let lastScore = 0, haveLast = false;
  const acc = { n: 0 };
  // Adaptive time: track how many consecutive depths produced the same best move
  const startTime = Date.now();
  let stableDepths = 0, lastBestKey = -1;

  for (let depth = 1; depth <= 24; depth++) {
    // Yield to the JS event loop between depth iterations so React Native can
    // process layout/input events and the UI doesn't freeze.
    await new Promise<void>(r => setTimeout(r, 0));
    if (cancel?.cancelled || Date.now() > deadline) break;

    let winSize = haveLast ? 150 : INF;
    let alpha   = haveLast ? lastScore - winSize : -INF;
    let beta    = haveLast ? lastScore + winSize : INF;
    let result: { move?: Move; score: number } = { score: 0 };

    while (true) {
      // Run root search (first move full window, rest PVS)
      const moves = generateMoves(root);
      if (!moves.length) break;
      const ordered = orderMoves(root, moves, tt.get(rootHash)?.move ?? -1, 0);

      let rootBest = -INF, rootMove: Move | undefined;
      acc.n = 0;

      for (let i = 0; i < ordered.length; i++) {
        if (cancel?.cancelled) break;
        if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) break;
        const m      = ordered[i];
        const child  = applyMove(root, m);
        const ch     = hashPosition(child);
        const mk     = key(m);
        const single = ordered.length === 1;
        const total  = bitCount(child.p1Men|child.p1Kings|child.p2Men|child.p2Kings);
        const isQ    = m.captured.length === 0;

        let d = depth - 1;
        if (single) d = Math.min(depth, d+1);
        if (total <= 5) d = Math.min(depth, d+1);
        // LMR at root — skip on tactical positions (opponent has forced captures)
        const opCapRoot = isQ && hasCapturesAvailable(child);
        if (i >= 3 && d >= 2 && isQ && !single && !opCapRoot) {
          d = Math.max(1, d - (LMR[Math.min(31,i)][Math.min(31,d)] | 0));
        }

        pushRepetition(rep, ch);
        let score: number;
        if (i === 0) {
          score = -negamax(child, d, -beta, -alpha, tt, deadline, acc, 1, rep, true, mk);
        } else {
          score = -negamax(child, d, -(alpha+1), -alpha, tt, deadline, acc, 1, rep, true, mk);
          if (score > alpha && score < beta)
            score = -negamax(child, depth-1, -beta, -alpha, tt, deadline, acc, 1, rep, true, mk);
        }
        popRepetition(rep, ch);
        acc.n++;

        if (score > rootBest) { rootBest = score; rootMove = m; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
      }

      nodes += acc.n;
      result = { move: rootMove, score: rootBest };

      if (cancel?.cancelled || Date.now() > deadline) break;
      if (rootBest <= (haveLast ? lastScore - winSize : -INF)) {
        winSize = Math.min(INF, winSize * 2); alpha = Math.max(-INF, (haveLast ? lastScore : 0) - winSize); continue;
      }
      if (rootBest >= (haveLast ? lastScore + winSize : INF)) {
        winSize = Math.min(INF, winSize * 2); beta = Math.min(INF, (haveLast ? lastScore : 0) + winSize); continue;
      }
      break;
    }

    if (cancel?.cancelled || Date.now() > deadline) break;
    if (result.move) { best = result.move; bestScore = result.score; reached = depth; }
    lastScore = result.score; haveLast = true;
    onInfo?.({ depth, score: bestScore, nodes, pv: getPV(root, tt) });

    // Adaptive time: if the best move hasn't changed for 3+ depths and we've
    // used ≥50% of the time budget, the search has converged — stop early.
    const newBestKey = result.move ? key(result.move) : -1;
    if (newBestKey === lastBestKey) { stableDepths++; } else { stableDepths = 0; lastBestKey = newBestKey; }
    if (stableDepths >= 3 && Date.now() > startTime + timeMs * 0.5) break;
  }

  // Safety fallback: if the search somehow produced no best move (e.g. time
  // expired before depth-1 completed, or threefold detected mid-search) but
  // legal moves exist, return the first one so the game never stalls.
  if (!best) {
    const fallback = generateMoves(root);
    if (fallback.length) best = fallback[0];
  }

  return { best, score: bestScore, nodes, depth: reached };
}
