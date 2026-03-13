// src/coreCodex/search/alphabeta.ts
// Iterative deepening negamax with alpha-beta pruning, PVS, TT, LMR, and quiescence.
import { Move, applyMove, generateMoves } from '../movegen';
import { B1, bitCount, toRC } from '../bitboards';
import { evaluate } from '../eval';
import { isDrawByInactivity, Position } from '../position';
import { Bound, TT } from './tt';
import { probeSmallEndgame, probeSmallEndgameFromCounts } from './endgameTablebase';
import { lookupOpeningBook } from './openingBook';
import {
  buildRepetitionCounts,
  getRepetitionCount,
  isThreefoldRepetition,
  popRepetition,
  pushRepetition,
  RepetitionCounts,
} from './repetition';
import { hashPosition } from './zobrist';

export interface SearchInfo { depth: number; score: number; nodes: number; pv: Move[]; }
export interface SearchResult { best?: Move; score: number; nodes: number; depth: number; }
type OnInfo = (info: SearchInfo) => void;

const INF = 1e9 | 0;
const MAX_PLY = 96;

const killers0 = new Int32Array(MAX_PLY).fill(-1);
const killers1 = new Int32Array(MAX_PLY).fill(-1);
const history = new Int32Array(32 * 32);

interface RootEntry {
  move: Move;
  child: Position;
  finisherBoost: number;
  mobilityScore: number;
  priority: number;
}

interface ExtensionArgs {
  depth: number;
  budget: number;
  endgameSmall: boolean;
  oppHasCapture: boolean;
  parentHasSingle: boolean;
  childHasSingle: boolean;
}

function keyMove(m: Move) {
  return (m.from << 5) | m.to;
}

function sameMove(m: Move, key: number) {
  return key === keyMove(m);
}

function sideToMovePiecesBits(p: Position): number {
  return p.side === 1 ? (p.p1Men | p.p1Kings) : (p.p2Men | p.p2Kings);
}

function kingsOnlyAndCount(p: Position) {
  const men = bitCount(p.p1Men | p.p2Men);
  const k1 = bitCount(p.p1Kings);
  const k2 = bitCount(p.p2Kings);
  return { kingsOnly: men === 0, ktotal: k1 + k2, k1, k2 };
}

function calcRootBudget(p: Position) {
  const { kingsOnly, ktotal } = kingsOnlyAndCount(p);
  return kingsOnly && ktotal <= 3 ? 2 : 1;
}

function shouldRunFinisherScan(p: Position, moves: Move[]) {
  const totalPieces = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  return totalPieces <= 6 || moves.length <= 8;
}

function isImmediateWin(pos: Position): boolean {
  if (bitCount(sideToMovePiecesBits(pos)) === 0) return true;
  return generateMoves(pos).length === 0;
}

function forcedMovesOnly(pos: Position): Move[] {
  const moves = generateMoves(pos);
  const captures = moves.filter((m) => m.captured.length > 0);
  return captures.length ? captures : moves;
}

function isRootForcedWinInTwo(root: Position, move: Move): boolean {
  const afterMine = applyMove(root, move);
  if (isImmediateWin(afterMine)) return true;

  const oppReplies = forcedMovesOnly(afterMine);
  for (const reply of oppReplies) {
    const afterOpp = applyMove(afterMine, reply);
    const myFinishes = forcedMovesOnly(afterOpp);
    let foundFinish = false;
    for (const finish of myFinishes) {
      const finalPos = applyMove(afterOpp, finish);
      if (isImmediateWin(finalPos)) {
        foundFinish = true;
        break;
      }
    }
    if (!foundFinish) return false;
  }
  return true;
}

function isRootForcedWinInThree(root: Position, move: Move): boolean {
  const afterMine = applyMove(root, move);
  const oppReplies = forcedMovesOnly(afterMine);

  for (const reply of oppReplies) {
    const afterOpp = applyMove(afterMine, reply);
    const myReplies = forcedMovesOnly(afterOpp);
    let lineWins = false;

    for (const myReply of myReplies) {
      const afterMy = applyMove(afterOpp, myReply);
      if (isImmediateWin(afterMy)) {
        lineWins = true;
        break;
      }

      const oppSecondReplies = forcedMovesOnly(afterMy);
      let everyReplyLoses = true;
      for (const oppSecond of oppSecondReplies) {
        const finalPos = applyMove(afterMy, oppSecond);
        if (!isImmediateWin(finalPos)) {
          everyReplyLoses = false;
          break;
        }
      }

      if (everyReplyLoses) {
        lineWins = true;
        break;
      }
    }

    if (!lineWins) return false;
  }

  return true;
}

function mobilityDropScore(after: Position): number {
  const oppMoves = generateMoves(after).length;
  const { kingsOnly, ktotal } = kingsOnlyAndCount(after);
  const base = Math.max(0, 12 - oppMoves);
  if (kingsOnly && ktotal <= 3) return base * 6;
  if (kingsOnly) return base * 4;
  return base * 2;
}

function computeExtensionFlexible(args: ExtensionArgs) {
  let depth = args.depth - 1;
  let budget = args.budget;

  if (depth > 0 && budget > 0 && args.parentHasSingle) {
    depth++;
    budget--;
  }

  if (depth > 0 && budget > 0 && (args.endgameSmall || args.oppHasCapture || args.childHasSingle)) {
    depth++;
    budget--;
  }

  if (depth > args.depth) depth = args.depth;
  if (depth < 0) depth = 0;
  return { depth, budget };
}

function staticMoveOrderBonus(pos: Position, move: Move): number {
  const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen = pos.side === 1 ? pos.p2Men : pos.p1Men;
  const opKings = pos.side === 1 ? pos.p2Kings : pos.p1Kings;
  const movingKing = (myKings & B1(move.from)) !== 0;
  const from = toRC(move.from);
  const to = toRC(move.to);

  let score = 0;

  if (move.captured.length > 0) {
    for (const square of move.captured) {
      const bit = B1(square);
      score += (opKings & bit) !== 0 ? 260 : 120;
    }
  }

  if (!movingKing) {
    const advance = pos.side === 1 ? from.r - to.r : to.r - from.r;
    score += advance * 30;
  } else {
    const centerFrom = Math.abs(from.r - 3.5) + Math.abs(from.c - 3.5);
    const centerTo = Math.abs(to.r - 3.5) + Math.abs(to.c - 3.5);
    score += Math.round((centerFrom - centerTo) * 10);
  }

  if (!move.captured.length && move.promote) score += 1_200;
  return score;
}

export function iterativeDeepening(
  root: Position,
  timeMs: number,
  tt = new TT(),
  onInfo?: OnInfo,
  historyHashes: number[] = [],
): SearchResult {
  const deadline = Date.now() + timeMs;
  const rootBudget = calcRootBudget(root);
  const rootHash = hashPosition(root);
  const normalizedHistory = historyHashes.length ? historyHashes : [rootHash];
  const repetitionCounts = buildRepetitionCounts(normalizedHistory);
  killers0.fill(-1);
  killers1.fill(-1);
  history.fill(0);

  if (isThreefoldRepetition(repetitionCounts, rootHash)) {
    return { best: undefined, score: 0, nodes: 0, depth: 0 };
  }

  const endgameHit = probeSmallEndgame(root, normalizedHistory);
  if (endgameHit) {
    onInfo?.({ depth: endgameHit.dtm, score: endgameHit.score, nodes: 0, pv: endgameHit.best ? [endgameHit.best] : [] });
    return { best: endgameHit.best, score: endgameHit.score, nodes: 0, depth: endgameHit.dtm };
  }

  const bookHit = lookupOpeningBook(root);
  if (bookHit) {
    const bookScore = -evaluate(applyMove(root, bookHit.move));
    onInfo?.({ depth: 0, score: bookScore, nodes: 0, pv: [bookHit.move] });
    return { best: bookHit.move, score: bookScore, nodes: 0, depth: 0 };
  }

  let best: Move | undefined;
  let bestScore = 0;
  let nodes = 0;
  let reached = 0;
  let lastScore = 0;
  let haveLast = false;

  for (let depth = 1; depth <= 22; depth++) {
    let alpha = haveLast ? lastScore - 80 : -INF;
    let beta = haveLast ? lastScore + 80 : INF;

    let result;
    while (true) {
      const st = { nodes: 0 };
      result = searchRoot(root, depth, alpha, beta, tt, deadline, st, 0, rootBudget, repetitionCounts);
      nodes += st.nodes;

      if (Date.now() > deadline) break;
      if (result.score <= alpha) {
        alpha = Math.max(-INF, alpha - 160);
        continue;
      }
      if (result.score >= beta) {
        beta = Math.min(INF, beta + 160);
        continue;
      }
      break;
    }

    if (Date.now() > deadline) break;
    if (result.move) {
      best = result.move;
      bestScore = result.score;
      reached = depth;
    }
    lastScore = result.score;
    haveLast = true;

    onInfo?.({ depth, score: bestScore, nodes, pv: getPV(root, tt, 12) });
  }

  return { best, score: bestScore, nodes, depth: reached };
}

function searchRoot(
  pos: Position,
  depth: number,
  alpha: number,
  beta: number,
  tt: TT,
  deadline: number,
  acc: { nodes: number },
  ply: number,
  budget: number,
  repetitionCounts: RepetitionCounts,
) {
  const currentHash = hashPosition(pos);
  if (isDrawByInactivity(pos) || isThreefoldRepetition(repetitionCounts, currentHash)) {
    return { move: undefined as Move | undefined, score: 0 };
  }

  const endgameHit = probeSmallEndgameFromCounts(pos, repetitionCounts);
  if (endgameHit) {
    return { move: endgameHit.best, score: endgameHit.score };
  }

  const moves = generateMoves(pos);
  if (moves.length === 0) return { move: undefined as Move | undefined, score: -999999 + ply };

  const key = hashPosition(pos);
  const ttHit = tt.get(key);
  const ttMove = ttHit?.move ?? -1;
  const base = orderMoves(pos, moves, ttMove, ply);
  const finisherScan = shouldRunFinisherScan(pos, moves);

  const ordered: RootEntry[] = base
    .map((move) => {
      const child = applyMove(pos, move);
      const childHash = hashPosition(child);
      const mobilityScore = mobilityDropScore(child);
      const oppHasCaptureNow = generateMoves(child).some((m) => m.captured.length > 0);
      const winInTwo = finisherScan && isRootForcedWinInTwo(pos, move);
      const winInThree = finisherScan && !winInTwo && isRootForcedWinInThree(pos, move);

      let priority = mobilityScore;
      let finisherBoost = 0;
      if (winInTwo) finisherBoost = 1_000_000;
      else if (winInThree) finisherBoost = 900_000;
      priority += finisherBoost;

      if (!finisherBoost && oppHasCaptureNow) priority -= 200;
      if (getRepetitionCount(repetitionCounts, childHash) >= 2) priority -= 800;
      priority += (key ^ keyMove(move)) & 7;

      return { move, child, finisherBoost, mobilityScore, priority };
    })
    .sort((a, b) => b.priority - a.priority);

  if (ordered[0]?.finisherBoost >= 900_000) {
    return { move: ordered[0].move, score: 900_000 };
  }

  let bestScore = -INF;
  let bestMove: Move | undefined;
  const a0 = alpha;
  const b0 = beta;

  for (let i = 0; i < ordered.length; i++) {
    if (Date.now() > deadline) break;

    const entry = ordered[i];
    const move = entry.move;
    const child = entry.child;
    const childMoves = generateMoves(child);
    const totalPieces = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    const oppHasCapture = childMoves.some((m) => m.captured.length > 0);
    const childHasSingle = childMoves.length === 1;
    const parentHasSingle = ordered.length === 1;

    const ext = computeExtensionFlexible({
      depth,
      budget,
      endgameSmall: totalPieces <= 5,
      oppHasCapture,
      parentHasSingle,
      childHasSingle,
    });

    let depthToUse = ext.depth;
    const isQuiet = move.captured.length === 0;
    const disableLMR = ordered.length <= 2 || childHasSingle;
    const lateQuiet = i >= 3 && depthToUse >= 2 && isQuiet && !disableLMR;
    if (lateQuiet) depthToUse = Math.max(0, depthToUse - 1);

    let score: number;
    pushRepetition(repetitionCounts, hashPosition(child));
    if (i === 0) {
      score = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1, ext.budget, repetitionCounts);
    } else {
      score = -alphabeta(
        child,
        depthToUse,
        -(alpha + 1),
        -alpha,
        tt,
        deadline,
        acc,
        ply + 1,
        ext.budget,
        repetitionCounts,
      );
      if (score > alpha && score < beta) {
        score = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1, ext.budget, repetitionCounts);
      }
    }
    popRepetition(repetitionCounts, hashPosition(child));

    if (entry.finisherBoost) score += 500;
    score += Math.min(100, entry.mobilityScore);

    acc.nodes++;
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) {
      if (isQuiet) updateHeuristics(move, depth, ply);
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
  pos: Position,
  depth: number,
  alpha: number,
  beta: number,
  tt: TT,
  deadline: number,
  acc: { nodes: number },
  ply: number,
  budget: number,
  repetitionCounts: RepetitionCounts,
): number {
  const key = hashPosition(pos);
  if (isDrawByInactivity(pos) || isThreefoldRepetition(repetitionCounts, key)) return 0;
  const endgameHit = probeSmallEndgameFromCounts(pos, repetitionCounts);
  if (endgameHit) return endgameHit.score;
  if (ply >= MAX_PLY) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);
  if (depth <= 0) return quiesce(pos, alpha, beta, deadline, acc, ply);

  const hit = tt.get(key);
  if (hit && hit.depth >= depth && getRepetitionCount(repetitionCounts, key) <= 1) {
    if (hit.bound === Bound.EXACT) return hit.score;
    if (hit.bound === Bound.LOWER && hit.score > alpha) alpha = hit.score;
    else if (hit.bound === Bound.UPPER && hit.score < beta) beta = hit.score;
    if (alpha >= beta) return hit.score;
  }

  const moves = generateMoves(pos);
  if (moves.length === 0) return -999999 + ply;

  const ordered = orderMoves(pos, moves, hit?.move ?? -1, ply);
  let best = -INF;
  const a0 = alpha;
  const b0 = beta;
  let bestKey = -1;

  for (let i = 0; i < ordered.length; i++) {
    if (Date.now() > deadline) break;

    const move = ordered[i];
    const child = applyMove(pos, move);
    const childHash = hashPosition(child);
    const childMoves = generateMoves(child);
    const totalPieces = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
    const oppHasCapture = childMoves.some((m) => m.captured.length > 0);
    const childHasSingle = childMoves.length === 1;
    const parentHasSingle = ordered.length === 1;

    const ext = computeExtensionFlexible({
      depth,
      budget,
      endgameSmall: totalPieces <= 5,
      oppHasCapture,
      parentHasSingle,
      childHasSingle,
    });

    let depthToUse = ext.depth;
    const isQuiet = move.captured.length === 0;
    const disableLMR = ordered.length <= 2 || childHasSingle;
    const lateQuiet = i >= 3 && depthToUse >= 2 && isQuiet && !disableLMR;
    if (lateQuiet) depthToUse = Math.max(0, depthToUse - 1);

    let score: number;
    pushRepetition(repetitionCounts, childHash);
    if (i === 0) {
      score = -alphabeta(child, depthToUse, -beta, -alpha, tt, deadline, acc, ply + 1, ext.budget, repetitionCounts);
    } else {
      score = -alphabeta(
        child,
        depthToUse,
        -(alpha + 1),
        -alpha,
        tt,
        deadline,
        acc,
        ply + 1,
        ext.budget,
        repetitionCounts,
      );
      if (score > alpha && depthToUse < depth - 1) {
        score = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1, ext.budget, repetitionCounts);
      } else if (score > alpha && score < beta) {
        score = -alphabeta(child, depth - 1, -beta, -alpha, tt, deadline, acc, ply + 1, ext.budget, repetitionCounts);
      }
    }
    popRepetition(repetitionCounts, childHash);

    acc.nodes++;
    if (score > best) {
      best = score;
      bestKey = keyMove(move);
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) {
      if (isQuiet) updateHeuristics(move, depth, ply);
      break;
    }
  }

  const entry = { key, depth, score: best, move: bestKey, bound: Bound.EXACT as Bound };
  if (best <= a0) entry.bound = Bound.UPPER;
  else if (best >= b0) entry.bound = Bound.LOWER;
  if (getRepetitionCount(repetitionCounts, key) <= 1) tt.put(entry);
  return best;
}

function quiesce(
  pos: Position,
  alpha: number,
  beta: number,
  deadline: number,
  acc: { nodes: number },
  ply: number,
): number {
  if (isDrawByInactivity(pos)) return 0;
  if (ply >= MAX_PLY) return evaluate(pos);
  if (Date.now() > deadline) return evaluate(pos);

  let stand = evaluate(pos);
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

  const captures = generateMoves(pos).filter((m) => m.captured.length > 0);
  captures.sort((a, b) => b.captured.length - a.captured.length);

  for (const move of captures) {
    const child = applyMove(pos, move);
    const score = -quiesce(child, -beta, -alpha, deadline, acc, ply + 1);
    acc.nodes++;
    if (score >= beta) return score;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

function updateHeuristics(move: Move, depth: number, ply: number) {
  const moveKey = keyMove(move);
  if (killers0[ply] !== moveKey) {
    killers1[ply] = killers0[ply];
    killers0[ply] = moveKey;
  }
  history[moveKey] += depth * depth;
}

function orderMoves(pos: Position, moves: Move[], ttKey: number, ply: number): Move[] {
  return moves
    .map((move) => {
      const moveKey = keyMove(move);
      let score = 0;
      if (ttKey >= 0 && sameMove(move, ttKey)) score += 1_000_000;
      if (move.captured.length) score += 10_000 * move.captured.length;
      if (moveKey === killers0[ply]) score += 5_000;
      if (moveKey === killers1[ply]) score += 4_000;
      score += history[moveKey] | 0;
      score += staticMoveOrderBonus(pos, move);
      return { move, score };
    })
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.move);
}

function getPV(pos: Position, tt: TT, maxLen = 12): Move[] {
  const pv: Move[] = [];
  let current = pos;
  for (let i = 0; i < maxLen; i++) {
    const hit = tt.get(hashPosition(current));
    if (!hit || hit.move == null || hit.move < 0) break;
    const from = (hit.move >> 5) & 31;
    const to = hit.move & 31;
    const move = generateMoves(current).find((candidate) => candidate.from === from && candidate.to === to);
    if (!move) break;
    pv.push(move);
    current = applyMove(current, move);
  }
  return pv;
}
