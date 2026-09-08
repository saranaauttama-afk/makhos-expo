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
import {
  hashPosition, hashSearchState, verifyHashSearchState,
} from './zobrist';

export interface SearchInfo  { depth: number; score: number; nodes: number; pv: Move[]; }
export interface RootSearchCandidate { move: Move; score: number; }
export interface SearchResult {
  best?: Move;
  score: number;
  nodes: number;
  qnodes: number;
  depth: number;
  elapsedMs: number;
  timedOut: boolean;
  pv: Move[];
  limitReached?: 'nodes';
  overrideReason?: string;
  rootCandidates?: RootSearchCandidate[];
}
export interface CancelToken  { cancelled: boolean; }
export interface DeterministicSearchOptions {
  /** Exact completed iterative-deepening depth; wall clock is ignored. */
  depth?: number;
  /** Maximum combined main-search and qsearch nodes; wall clock is ignored. */
  nodes?: number;
}
export interface SearchFeatureFlags {
  reverseFutility: boolean;
  razoring: boolean;
  nullMove: boolean;
  probCut: boolean;
  iid: boolean;
  lmr: boolean;
  lmp: boolean;
  extensions: boolean;
}
export const DEFAULT_SEARCH_FEATURES: Readonly<SearchFeatureFlags> = Object.freeze({
  reverseFutility: true,
  razoring: true,
  nullMove: true,
  probCut: true,
  iid: true,
  lmr: true,
  lmp: true,
  extensions: true,
});
export type RootMoveScores = ReadonlyMap<number, number>;
type OnInfo = (info: SearchInfo) => void;
type RootCandidate = RootSearchCandidate;
type RootMoveSource =
  | 'normalSearch'
  | 'openingDiversification'
  | 'trapOverride'
  | 'lowMobilityRecaptureOverride'
  | 'promotionOverride'
  | 'rootTacticalSafetyOverride'
  | 'antiHangSafetyOverride'
  | 'fallbackLegalMove';

interface RootOverrideCounter {
  attempts: number;
  accepted: number;
}

export interface RootOverrideStats {
  searches: number;
  openingDiversification: RootOverrideCounter;
  trapOverride: RootOverrideCounter;
  lowMobilityRecaptureOverride: RootOverrideCounter;
  promotionOverride: RootOverrideCounter;
  rootTacticalSafetyOverride: RootOverrideCounter;
  antiHangSafetyOverride: RootOverrideCounter;
  finalMoveSource: Record<RootMoveSource, number>;
}

// Eval override — allows A/B testing without changing all call sites
let _eval: (p: Position) => number = evaluate;
export function setEvalFn(fn: (p: Position) => number): void { _eval = fn; }
export function resetEvalFn(): void { _eval = evaluate; }

const INF        = 1_000_000;
const MAX_PLY    = 64;
const TC_MASK    = 511; // check time every 512 nodes
const ENABLE_LOW_MOBILITY_EXACT_TIEBREAK =
  process.env.MAKHOS_ENABLE_LOW_MOBILITY_EXACT_TIEBREAK === '1';

// Shared stop flag — set when deadline fires; cleared before each iterativeDeepening.
// All levels check this at the top of negamax so the search unwinds immediately
// once time is up, rather than waiting for each level's own TC_MASK checkpoint.
const stop = { flag: false };
let activeNodeLimit: number | undefined;
let activeNodeCount = 0;
let activeFeatures: SearchFeatureFlags = { ...DEFAULT_SEARCH_FEATURES };

export function scoreToTT(score: number, ply: number): number {
  if (score >= INF - MAX_PLY) return score + ply;
  if (score <= -INF + MAX_PLY) return score - ply;
  return score;
}

export function scoreFromTT(score: number, ply: number): number {
  if (score >= INF - MAX_PLY) return score - ply;
  if (score <= -INF + MAX_PLY) return score + ply;
  return score;
}

function enterNode(acc: {n:number; q:number}, kind: 'main' | 'q'): boolean {
  if (activeNodeLimit !== undefined && activeNodeCount >= activeNodeLimit) {
    stop.flag = true;
    return false;
  }
  activeNodeCount++;
  if (kind === 'main') acc.n++;
  else acc.q++;
  return true;
}

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

export function moveKey(m: Move) { return (m.from << 5) | m.to; }

function makeRootOverrideCounter(): RootOverrideCounter {
  return { attempts: 0, accepted: 0 };
}

function makeRootOverrideStats(): RootOverrideStats {
  return {
    searches: 0,
    openingDiversification: makeRootOverrideCounter(),
    trapOverride: makeRootOverrideCounter(),
    lowMobilityRecaptureOverride: makeRootOverrideCounter(),
    promotionOverride: makeRootOverrideCounter(),
    rootTacticalSafetyOverride: makeRootOverrideCounter(),
    antiHangSafetyOverride: makeRootOverrideCounter(),
    finalMoveSource: {
      normalSearch: 0,
      openingDiversification: 0,
      trapOverride: 0,
      lowMobilityRecaptureOverride: 0,
      promotionOverride: 0,
      rootTacticalSafetyOverride: 0,
      antiHangSafetyOverride: 0,
      fallbackLegalMove: 0,
    },
  };
}

let rootOverrideStats = makeRootOverrideStats();

export function resetRootOverrideStats(): void {
  rootOverrideStats = makeRootOverrideStats();
}

export function getRootOverrideStats(): RootOverrideStats {
  return {
    searches: rootOverrideStats.searches,
    openingDiversification: { ...rootOverrideStats.openingDiversification },
    trapOverride: { ...rootOverrideStats.trapOverride },
    lowMobilityRecaptureOverride: { ...rootOverrideStats.lowMobilityRecaptureOverride },
    promotionOverride: { ...rootOverrideStats.promotionOverride },
    rootTacticalSafetyOverride: { ...rootOverrideStats.rootTacticalSafetyOverride },
    antiHangSafetyOverride: { ...rootOverrideStats.antiHangSafetyOverride },
    finalMoveSource: { ...rootOverrideStats.finalMoveSource },
  };
}

function rootHint(rootMoveScores: RootMoveScores | undefined, k: number): number {
  return rootMoveScores?.get(k) ?? 0;
}

function pickDiversifiedRoot(candidates: RootCandidate[], bestScore: number): RootCandidate | undefined {
  if (candidates.length < 2) return undefined;
  if (Math.abs(bestScore) > INF - MAX_PLY) return undefined;

  const sorted = [...candidates]
    .filter(c => c.move.captured.length === 0 && c.score >= bestScore - 35)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);
  if (sorted.length < 2) return undefined;

  let total = 0;
  const weights = sorted.map((c, i) => {
    const scoreWeight = Math.exp((c.score - sorted[0].score) / 28);
    const rankWeight = 1 / (1 + i * 0.7);
    const weight = scoreWeight * rankWeight;
    total += weight;
    return weight;
  });

  let roll = Math.random() * total;
  for (let i = 0; i < sorted.length; i++) {
    roll -= weights[i];
    if (roll <= 0) return sorted[i];
  }
  return sorted[0];
}

function extendTacticalDepth(depth: number, d: number, move: Move, opHasCaptures: boolean, ply: number): number {
  if (ply > 18 || depth < 2) return d;
  let ext = 0;
  if (move.captured.length > 0 || opHasCaptures) ext++;
  if (move.captured.length >= 2 && depth <= 6) ext++;
  // Common tactical blind spot on low depths: quiet move that allows immediate
  // capture, or a shallow single-capture that gets recaptured right away.
  if (depth <= 6 && opHasCaptures) {
    if (move.captured.length === 0) ext++;
    if (move.captured.length === 1) ext++;
  }
  if (ext <= 0) return d;
  const cap = move.captured.length >= 2 ? depth + 2 : depth + 1;
  return Math.min(cap, d + ext);
}

function candidateWindow(candidates: RootCandidate[], bestScore: number, margin = 80): RootCandidate[] {
  return [...candidates]
    .filter(c => c.score >= bestScore - margin)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);
}

function immediateCaptureRiskAfter(root: Position, move: Move): number {
  const child = applyMove(root, move);
  const oppMoves = generateMoves(child);
  if (!oppMoves.length) return -5_000; // immediate win: opponent has no legal move
  if (oppMoves[0].captured.length === 0) return 0; // opponent cannot capture right away

  let maxCap = 0;
  let maxCapLines = 0;
  let hangingMovedPieceMax = 0;
  for (const reply of oppMoves) {
    const cap = reply.captured.length;
    if (cap > maxCap) {
      maxCap = cap;
      maxCapLines = 1;
    } else if (cap === maxCap) {
      maxCapLines++;
    }
    if (reply.captured.includes(move.to)) {
      if (cap > hangingMovedPieceMax) hangingMovedPieceMax = cap;
    }
  }

  // Penalize immediate tactical shots heavily, especially multi-captures.
  let risk = maxCap * 140;
  if (maxCap >= 2) risk += 180;
  if (maxCap >= 3) risk += 220;
  // Strongly punish moves that hang the moved piece to immediate capture.
  if (hangingMovedPieceMax > 0) {
    risk += 260 + hangingMovedPieceMax * 210;
    if (hangingMovedPieceMax >= 2) risk += 240;
  }
  if (move.captured.length === 0) risk += 50; // quiet self-pin blunders are common
  if (maxCapLines >= 2) risk += 40;
  return risk;
}

function safeOverrideMargin(currentRisk: number, saferRisk: number): number {
  const riskDrop = currentRisk - saferRisk;
  if (currentRisk >= 900) return Math.min(520, 220 + riskDrop);
  if (currentRisk >= 650) return Math.min(420, 180 + riskDrop);
  if (currentRisk >= 420) return Math.min(300, 140 + riskDrop);
  return 115;
}

function materialForSide(pos: Position, side: 1 | -1): number {
  const myMen = side === 1 ? pos.p1Men : pos.p2Men;
  const myKings = side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen = side === 1 ? pos.p2Men : pos.p1Men;
  const opKings = side === 1 ? pos.p2Kings : pos.p1Kings;
  return bitCount(myMen) * 100 + bitCount(myKings) * 300 - bitCount(opMen) * 100 - bitCount(opKings) * 300;
}

function piecesForSide(pos: Position, side: 1 | -1): number {
  return bitCount(side === 1 ? (pos.p1Men | pos.p1Kings) : (pos.p2Men | pos.p2Kings));
}

function scoreRootMaterialDelta(root: Position, pos: Position): { terminalWin: boolean; netGain: number; pieceGain: number } {
  const rootSide = root.side;
  const opponentSide = (rootSide === 1 ? -1 : 1) as 1 | -1;
  const beforeNet = materialForSide(root, rootSide);
  const beforeMyPieces = piecesForSide(root, rootSide);
  const beforeOppPieces = piecesForSide(root, opponentSide);
  const terminalWin = generateMoves(pos).length === 0;
  const netGain = terminalWin ? INF : materialForSide(pos, rootSide) - beforeNet;
  const myPiecesNow = piecesForSide(pos, rootSide);
  const oppPiecesNow = piecesForSide(pos, opponentSide);
  const pieceGain = terminalWin ? INF : (beforeOppPieces - oppPiecesNow) - (beforeMyPieces - myPiecesNow);
  return { terminalWin, netGain, pieceGain };
}

function scoreAfterImmediateCounter(root: Position, afterOurMove: Position): { terminalWin: boolean; netGain: number; pieceGain: number } {
  const opponentReplies = generateMoves(afterOurMove);
  if (!opponentReplies.length) return { terminalWin: true, netGain: INF, pieceGain: INF };
  if (opponentReplies[0].captured.length === 0) return scoreRootMaterialDelta(root, afterOurMove);

  let worst: { terminalWin: boolean; netGain: number; pieceGain: number } | undefined;
  for (const reply of opponentReplies) {
    const afterCounter = applyMove(afterOurMove, reply);
    const current = scoreRootMaterialDelta(root, afterCounter);
    if (!worst || current.netGain < worst.netGain || (current.netGain === worst.netGain && current.pieceGain < worst.pieceGain)) {
      worst = current;
    }
  }

  return worst ?? scoreRootMaterialDelta(root, afterOurMove);
}

function trapReplyScore(root: Position, afterReply: Position): { terminalWin: boolean; netGain: number; pieceGain: number } | undefined {
  const ourReplies = generateMoves(afterReply);
  if (!ourReplies.length) return undefined;
  if (ourReplies[0].captured.length === 0) return undefined;

  let best: { terminalWin: boolean; netGain: number; pieceGain: number } | undefined;
  for (const recapture of ourReplies) {
    const afterRecapture = applyMove(afterReply, recapture);
    const current = scoreAfterImmediateCounter(root, afterRecapture);

    if (!best || current.netGain > best.netGain || (current.netGain === best.netGain && current.pieceGain > best.pieceGain)) {
      best = current;
    }
  }

  return best;
}

function isSoundForcedTrap(root: Position, move: Move): boolean {
  if (move.captured.length > 0) return false;

  const total = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  const child = applyMove(root, move);
  const opponentReplies = generateMoves(child);
  if (!opponentReplies.length) return true;
  if (opponentReplies[0].captured.length === 0) return false;

  const minGain = total > 10 ? 160 : 120;
  let hasTerminalWin = false;
  let worstNetGainAfterTrap = INF;
  let worstPieceGain = INF;

  for (const reply of opponentReplies) {
    const afterReply = applyMove(child, reply);
    const replyScore = trapReplyScore(root, afterReply);
    if (!replyScore) return false;

    hasTerminalWin = hasTerminalWin || replyScore.terminalWin;
    worstNetGainAfterTrap = Math.min(worstNetGainAfterTrap, replyScore.netGain);
    worstPieceGain = Math.min(worstPieceGain, replyScore.pieceGain);
  }

  return hasTerminalWin || worstNetGainAfterTrap >= minGain || (worstPieceGain >= 1 && worstNetGainAfterTrap >= 0);
}

function pickSoundForcedTrap(
  root: Position,
  candidates: RootCandidate[] | undefined,
  bestScore: number,
): RootCandidate | undefined {
  const source = candidates?.length
    ? candidates
    : generateMoves(root).map(move => ({ move, score: bestScore - 180 }));
  const traps = source
    .filter(candidate => isSoundForcedTrap(root, candidate.move))
    .sort((a, b) => b.score - a.score);
  if (!traps.length) return undefined;

  const bestTrap = traps[0];
  const total = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  const lowMobility = generateMoves(root).length <= 3;
  const maxConcession = lowMobility && total <= 8
    ? 900
    : bestTrap.move.promote
      ? 80
      : 260;
  if (bestTrap.score < bestScore - maxConcession) return undefined;
  return bestTrap;
}

function hasForcedRecaptureReply(root: Position, move: Move): boolean {
  if (move.captured.length > 0) return false;

  const child = applyMove(root, move);
  const opponentReplies = generateMoves(child);
  if (!opponentReplies.length || opponentReplies[0].captured.length === 0) return false;

  for (const reply of opponentReplies) {
    const afterReply = applyMove(child, reply);
    const ourReplies = generateMoves(afterReply);
    if (!ourReplies.length || ourReplies[0].captured.length === 0) return false;

    const bestRecaptureLen = Math.max(...ourReplies.map(recapture => recapture.captured.length));
    if (bestRecaptureLen <= reply.captured.length) return false;
  }

  return true;
}

function pickLowMobilityRecaptureCandidate(
  root: Position,
  candidates: RootCandidate[] | undefined,
  bestScore: number,
): RootCandidate | undefined {
  const legal = generateMoves(root);
  const total = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  if (total > 8 || legal.length > 3 || legal[0]?.captured.length > 0) return undefined;

  // Contextual threshold: use mobility as indicator of squeeze severity
  // - 1-2 legal moves = extreme squeeze → aggressive override (550)
  // - 3 legal moves = mild squeeze → conservative override (350)
  // This prevents over-aggressive override in positions with more options
  const threshold = legal.length <= 2 ? 550 : 350;

  const source = candidates?.length
    ? candidates
    : legal.map(move => ({ move, score: bestScore - 180 }));
  return source
    .filter(candidate => candidate.score >= bestScore - threshold)
    .filter(candidate => hasForcedRecaptureReply(root, candidate.move))
    .sort((a, b) => b.score - a.score)[0];
}

function allowLowMobilityExactTiebreak(root: Position, legalMoves: Move[]): boolean {
  const total = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  return (
    ENABLE_LOW_MOBILITY_EXACT_TIEBREAK &&
    total <= 6 &&
    root.p1Kings === 0 &&
    root.p2Kings === 0 &&
    legalMoves.length <= 3 &&
    legalMoves[0]?.captured.length === 0
  );
}

function pickEndgamePromotionCandidate(
  root: Position,
  candidates: RootCandidate[] | undefined,
  bestScore: number,
): RootCandidate | undefined {
  const total = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  if (total > 6) return undefined;

  const source = candidates?.length
    ? candidates
    : generateMoves(root).map(move => ({ move, score: bestScore - 120 }));
  const promotion = source
    .filter(candidate => candidate.move.promote && candidate.score >= bestScore - 180)
    .sort((a, b) => b.score - a.score)[0];
  return promotion;
}

function pickSaferRootCandidate(
  root: Position,
  candidates: RootCandidate[],
  bestScore: number,
  currentMove?: Move,
): RootCandidate | undefined {
  if (candidates.length < 2) return undefined;

  const analyzed = candidates.map(c => ({
    ...c,
    risk: immediateCaptureRiskAfter(root, c.move),
  }));

  const byScore = [...analyzed].sort((a, b) => b.score - a.score);
  const currentKey = currentMove ? moveKey(currentMove) : -1;
  const currentBest = analyzed.find(c => moveKey(c.move) === currentKey) ?? byScore[0];
  if (!currentBest) return undefined;
  if (isSoundForcedTrap(root, currentBest.move)) return undefined;

  const safest = [...analyzed].sort((a, b) => (a.risk - b.risk) || (b.score - a.score))[0];
  if (!safest) return undefined;

  // Only override when the tactical risk difference is meaningful. For severe
  // immediate hangs, trust the one-ply safety signal more than a shallow eval.
  if (currentBest.risk < 220) return undefined;
  if (safest.risk + 90 > currentBest.risk) return undefined;

  const margin = safeOverrideMargin(currentBest.risk, safest.risk);
  const scoreFloor = Math.max(bestScore, currentBest.score) - margin;
  if (safest.score < scoreFloor) return undefined;
  return safest;
}

function pickAbsoluteAntiHangMove(root: Position, current: Move | undefined): Move | undefined {
  if (!current) return undefined;
  if (isSoundForcedTrap(root, current)) return undefined;
  const currentRisk = immediateCaptureRiskAfter(root, current);
  if (currentRisk < 820) return undefined;

  const legal = generateMoves(root);
  if (legal.length < 2) return undefined;

  // In low-mobility positions (<=3 legal moves), only override if the risk is VERY high.
  // The hanging piece detection in eval should handle moderate cases.
  const isLowMobility = legal.length <= 3;
  if (isLowMobility && currentRisk < 1100) return undefined;

  const safest = legal
    .map(move => ({ move, risk: immediateCaptureRiskAfter(root, move) }))
    .sort((a, b) =>
      (a.risk - b.risk) ||
      (b.move.captured.length - a.move.captured.length) ||
      (Number(b.move.promote) - Number(a.move.promote))
    )[0];
  if (!safest) return undefined;
  if (isSoundForcedTrap(root, current)) return undefined;
  if (safest.risk + 300 > currentRisk) return undefined;
  if (safest.risk > 180) return undefined;
  return safest.move;
}

// prevKey = key of the move that led to this position (-1 at root)
function orderMoves(
  pos: Position,
  moves: Move[],
  ttMove: number,
  ply: number,
  prevKey = -1,
  rootMoveScores?: RootMoveScores,
): Move[] {
  const cm = prevKey >= 0 ? counterMove[prevKey] : -1; // look up countermove
  return moves.map(m => {
    const k = moveKey(m);
    let s = 0;
    if (k === ttMove)        s += 2_000_000;
    s += Math.round(rootHint(rootMoveScores, k) * 120_000);
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
  const k = moveKey(m);
  if (killers0[ply] !== k) { killers1[ply] = killers0[ply]; killers0[ply] = k; }
}

function getPV(pos: Position, tt: TT, repetitions: RepetitionCounts, max = 64): Move[] {
  const pv: Move[] = []; let cur = pos;
  const pvRepetitions = new Map(repetitions);
  for (let i = 0; i < max; i++) {
    const hit = tt.get(hashSearchState(cur, pvRepetitions), verifyHashSearchState(cur, pvRepetitions));
    if (!hit || hit.move == null) break;
    const mv = generateMoves(cur).find(m => moveKey(m) === hit.move!); if (!mv) break;
    pv.push(mv); cur = applyMove(cur, mv);
    pushRepetition(pvRepetitions, hashPosition(cur));
  }
  return pv;
}

function getRootPV(root: Position, best: Move | undefined, tt: TT, repetitions: RepetitionCounts): Move[] {
  if (!best) return [];
  const child = applyMove(root, best);
  const childRepetitions = new Map(repetitions);
  pushRepetition(childRepetitions, hashPosition(child));
  return [best, ...getPV(child, tt, childRepetitions, 63)];
}

// ── Quiescence ───────────────────────────────────────────────────────────────
function quiesce(
  pos: Position, alpha: number, beta: number,
  deadline: number, acc: {n:number; q:number}, ply: number,
  rep: RepetitionCounts, lastCapSquare = -1
): number {
  if (!enterNode(acc, 'q')) return _eval(pos);
  if (stop.flag) return _eval(pos);
  const h = hashPosition(pos);
  if (isDrawByInactivity(pos) || isThreefoldRepetition(rep, h)) return 0;
  if (ply >= MAX_PLY) return _eval(pos);

  const moves = generateMoves(pos);
  if (!moves.length) return -INF + ply;

  // If a capture is forced, stand-pat is not a legal continuation in Thai
  // Checkers. Search the capture chain instead of allowing static eval cutoffs.
  const caps = moves[0].captured.length > 0 ? moves : [];
  if (!caps.length) {
    const stand = _eval(pos);
    if (stand >= beta) return beta;

    // Delta pruning: skip moves that can't possibly improve alpha.
    // EXCEPT: don't prune if we have men close to promotion (rows 1 or 6).
    // This fixes promotion race blind spots.
    if (stand + 300 < alpha) {
      const side = pos.side;
      const myMen = side === 1 ? pos.p1Men : pos.p2Men;
      let hasPromotionThreat = false;
      for (let sq = 0; sq < 32 && !hasPromotionThreat; sq++) {
        if (!(myMen & (1 << sq))) continue;
        const r = Math.floor(sq / 4);
        if ((side === 1 && r === 1) || (side === -1 && r === 6)) {
          hasPromotionThreat = true;
        }
      }
      if (!hasPromotionThreat) return alpha;
    }

    if (stand > alpha) alpha = stand;
    return alpha;
  }

  // Sort captures by length, but prioritize recaptures (same square as last capture)
  caps.sort((a, b) => {
    const aIsRecap = lastCapSquare >= 0 && a.to === lastCapSquare ? 1 : 0;
    const bIsRecap = lastCapSquare >= 0 && b.to === lastCapSquare ? 1 : 0;
    if (aIsRecap !== bIsRecap) return bIsRecap - aIsRecap;
    return b.captured.length - a.captured.length;
  });

  for (const m of caps) {
    if (stop.flag) break;
    if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) { stop.flag = true; break; }
    // Pass the captured square to detect recaptures in the next level
    const nextLastCapSquare = m.captured.length > 0 ? m.from : -1;
    const child = applyMove(pos, m);
    const childHash = hashPosition(child);
    pushRepetition(rep, childHash);
    const score = -quiesce(child, -beta, -alpha, deadline, acc, ply+1, rep, nextLastCapSquare);
    popRepetition(rep, childHash);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ── Negamax ──────────────────────────────────────────────────────────────────
function negamax(
  pos: Position, depth: number, alpha: number, beta: number, tt: TT,
  deadline: number, acc: {n:number; q:number}, ply: number, rep: RepetitionCounts,
  nullOk = true, prevMoveKey = -1, iidOk = true,
): number {
  if (!enterNode(acc, 'main')) return _eval(pos);
  const h = hashPosition(pos);
  const ttKey = hashSearchState(pos, rep);
  const ttVerifyKey = verifyHashSearchState(pos, rep);
  if (isDrawByInactivity(pos) || isThreefoldRepetition(rep, h)) return 0;

  // NOTE: endgame tablebase probe intentionally removed from hot path —
  // probeSmallEndgameFromCounts triggers full retrograde DFS for every node
  // in king endgames, causing catastrophic slowness (226s+ per move).
  // The root-level probe in iterativeDeepening handles endgame positions.

  if (ply >= MAX_PLY) return _eval(pos);
  if (stop.flag) return _eval(pos);
  if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) { stop.flag = true; return _eval(pos); }
  if (depth <= 0) return quiesce(pos, alpha, beta, deadline, acc, ply, rep);

  // TT probe
  const hit = tt.get(ttKey, ttVerifyKey);
  let ttMove = hit?.move ?? -1; // let — may be updated by IID below
  if (hit && hit.depth >= depth && getRepetitionCount(rep, h) <= 1) {
    const ttScore = scoreFromTT(hit.score, ply);
    if (hit.bound === Bound.EXACT) return ttScore;
    if (hit.bound === Bound.LOWER) alpha = Math.max(alpha, ttScore);
    else                           beta  = Math.min(beta,  ttScore);
    if (alpha >= beta) return ttScore;
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
      if (activeFeatures.reverseFutility && depth <= 4 && se - 120 * depth >= beta) return se;

      // Razoring:
      // If static eval is way below alpha even after adding a generous margin,
      // drop to quiescence — the position is likely a dead loss for us.
      if (activeFeatures.razoring && depth <= 2) {
        const margin = depth === 1 ? 350 : 550;
        if (se + margin < alpha) {
          const q = quiesce(pos, alpha - 1, alpha, deadline, acc, ply, rep);
          if (q < alpha) return q;
        }
      }

      // Null Move Pruning:
      // Skip our turn and let the opponent move twice. If the position is still
      // >= beta, we can prune — our position is too good to refute.
      // Only in quiet nodes (can't pass on forced captures), not near mate.
      const inactivityLimit = pos.p1Men === 0 && pos.p2Men === 0 ? 16 : 32;
      const nullMoveDrawSafe = pos.halfmoveClock + 2 < inactivityLimit;
      if (activeFeatures.nullMove && nullMoveDrawSafe && nullOk && depth >= 3 && beta < INF - MAX_PLY && se >= beta) {
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
    if (activeFeatures.probCut && !isQuiet && depth >= 5) {
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
        tried++;
        const s = -negamax(child, pcDepth, -pcBeta, -(pcBeta - 1), tt, deadline, acc, ply + 1, rep, true, moveKey(m));
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
  if (activeFeatures.iid && ttMove === -1 && depth >= 5 && ply > 0 && isPV && iidOk && Date.now() <= deadline) {
    negamax(pos, Math.min(depth - 2, 4), alpha, beta, tt, deadline, acc, ply, rep, false, prevMoveKey, false);
    const iidHit = tt.get(ttKey, ttVerifyKey);
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
    if (activeFeatures.extensions && single) d = Math.min(depth, d+1);       // our only move — extend
    if (activeFeatures.extensions && total <= 5) d = Math.min(depth, d+1);   // endgame ext

    // LMR — skip if opponent will have forced captures (sacrifice/tactic position).
    // hasCapturesAvailable is O(pieces) vs full generateMoves, avoiding double movegen.
    const opHasCaptures = isQ && hasCapturesAvailable(child);
    if (activeFeatures.extensions) d = extendTacticalDepth(depth, d, m, opHasCaptures, ply);
    const fullD = d;
    if (activeFeatures.lmr && i >= 3 && d >= 2 && isQ && !single && !opHasCaptures) {
      d = Math.max(1, d - (LMR[Math.min(31,i)][Math.min(31,d)] | 0));
    }

    // Late Move Pruning (LMP): at very shallow depth, stop searching quiet
    // moves beyond a threshold — they're very unlikely to raise alpha.
    if (activeFeatures.lmp && isQ && !single && depth <= 2 && i >= (depth === 1 ? 6 : 10) && alpha > -INF + MAX_PLY) break;

    pushRepetition(rep, ch);
    const mk = moveKey(m); // move key — passed as prevMoveKey to child nodes
    let score: number;
    if (i === 0) {
      score = -negamax(child, d, -beta, -alpha, tt, deadline, acc, ply+1, rep, true, mk);
    } else {
      score = -negamax(child, d, -(alpha+1), -alpha, tt, deadline, acc, ply+1, rep, true, mk);
      if (score > alpha && score < beta) {
        score = -negamax(child, fullD, -beta, -alpha, tt, deadline, acc, ply+1, rep, true, mk);
      }
    }
    popRepetition(rep, ch);
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
  // A timeout/node-stop can unwind a partially searched node. Never publish
  // that provisional bound to a caller-reused TT.
  if (!stop.flag && getRepetitionCount(rep, h) <= 1)
    tt.put({ key: ttKey, verifyKey: ttVerifyKey, depth, score: scoreToTT(best, ply), move: bestKey >= 0 ? bestKey : undefined, bound });
  return best;
}

// ── Iterative Deepening ──────────────────────────────────────────────────────
export async function iterativeDeepening(
  root: Position, timeMs: number, tt = new TT(), onInfo?: OnInfo,
  historyHashes: number[] = [], cancel?: CancelToken,
  maxDepth = 24,
  rootMoveScores?: RootMoveScores,
  diversifyRoot = false,
  deterministic?: DeterministicSearchOptions,
  featureOverrides: Partial<SearchFeatureFlags> = {},
): Promise<SearchResult> {
  rootOverrideStats.searches++;
  if (deterministic?.depth !== undefined && deterministic.nodes !== undefined)
    throw new Error('fixed-depth and fixed-node limits are mutually exclusive');
  const isDeterministic = deterministic?.depth !== undefined || deterministic?.nodes !== undefined;
  const effectiveMaxDepth = deterministic?.depth ?? maxDepth;
  const deadline  = isDeterministic ? Number.POSITIVE_INFINITY : Date.now() + timeMs;
  const startTime = Date.now();
  const rootHash  = hashPosition(root);
  const normHist  = historyHashes.length ? historyHashes : [rootHash];
  const rep       = buildRepetitionCounts(normHist);

  killers0.fill(-1); killers1.fill(-1);
  counterMove.fill(-1);                                       // reset per-search
  if (isDeterministic) history.fill(0);
  else for (let i = 0; i < history.length; i++) history[i] >>= 1; // age history
  stop.flag = false;                                          // clear stop flag
  activeNodeLimit = deterministic?.nodes === undefined
    ? undefined
    : Math.max(1, Math.floor(deterministic.nodes));
  activeNodeCount = 0;
  activeFeatures = { ...DEFAULT_SEARCH_FEATURES, ...featureOverrides };

  if (isThreefoldRepetition(rep, rootHash))
    return { best: undefined, score: 0, nodes: 0, qnodes: 0, depth: 0, elapsedMs: Date.now() - startTime, timedOut: false, pv: [] };

  // Cap probe so it never eats into the search budget.
  // Default maxMs=3000 could exceed timeMs entirely, leaving no time for search.
  const totalRootPieces = bitCount(root.p1Men | root.p1Kings | root.p2Men | root.p2Kings);
  const allRootKings = root.p1Men === 0 && root.p2Men === 0;
  const probeMs = totalRootPieces <= 3
    ? Math.min(1200, Math.max(500, Math.floor(timeMs * 0.65)))
    : allRootKings && totalRootPieces <= 4
      ? Math.min(900, Math.max(350, Math.floor(timeMs * 0.45)))
      : Math.min(500, Math.floor(timeMs * 0.3));
  const eg = isDeterministic ? undefined : probeSmallEndgame(root, normHist, probeMs);
  // Only shortcut when we have an actual move — draw positions at the depth
  // limit store bestMoveKey = NO_MOVE_KEY so eg.best would be undefined.
  // Fall through to regular search so the engine still picks a legal move.
  if (eg?.best) {
    onInfo?.({ depth: eg.dtm, score: eg.score, nodes: 0, pv: [eg.best] });
    return { best: eg.best, score: eg.score, nodes: 0, qnodes: 0, depth: eg.dtm, elapsedMs: Date.now() - startTime, timedOut: false, pv: [eg.best] };
  }

  let best: Move | undefined, bestScore = 0, nodes = 0, qnodes = 0, reached = 0;
  let overrideReason: string | undefined;
  let finalMoveSource: RootMoveSource = 'normalSearch';
  let lastRootCandidates: RootCandidate[] | undefined;
  let lastRootScore = 0;
  let lastScore = 0, haveLast = false;
  const acc = { n: 0, q: 0 };
  // Adaptive time: track how many consecutive depths produced the same best move
  let stableDepths = 0, lastBestKey = -1;

  for (let depth = 1; depth <= effectiveMaxDepth; depth++) {
    // Yield to the JS event loop between depth iterations so React Native can
    // process layout/input events and the UI doesn't freeze.
    await new Promise<void>(r => setTimeout(r, 0));
    if (cancel?.cancelled || Date.now() > deadline) break;

    let winSize = haveLast ? 150 : INF;
    let alpha   = haveLast ? lastScore - winSize : -INF;
    let beta    = haveLast ? lastScore + winSize : INF;
    let result: { move?: Move; score: number; candidates?: RootCandidate[] } = { score: 0 };

    while (true) {
      // Cooperative yield between aspiration retries.
      await new Promise<void>(r => setTimeout(r, 0));
      // Run root search (first move full window, rest PVS)
      const moves = generateMoves(root);
      if (!moves.length) break;
      const rootTTKey = hashSearchState(root, rep);
      const rootTTVerifyKey = verifyHashSearchState(root, rep);
      const ordered = orderMoves(root, moves, tt.get(rootTTKey, rootTTVerifyKey)?.move ?? -1, 0, -1, rootMoveScores);
      const rootLowMobility = ordered.length <= 3 && totalRootPieces <= 8 && moves[0].captured.length === 0;
      const rootLowMobilityExtension = rootLowMobility && depth >= 4
        ? (totalRootPieces <= 6 ? 4 : 2)
        : 0;
      const exactTieTiebreak = allowLowMobilityExactTiebreak(root, moves);

      let rootBest = -INF, rootMove: Move | undefined;
      let rootTieChildStatic: number | undefined;
      const rootCandidates: RootCandidate[] = [];
      acc.n = 0;
      acc.q = 0;

      for (let i = 0; i < ordered.length; i++) {
        if ((i & 1) === 1) {
          await new Promise<void>(r => setTimeout(r, 0));
        }
        if (cancel?.cancelled || stop.flag) break;
        if ((acc.n & TC_MASK) === 0 && Date.now() > deadline) break;
        const m      = ordered[i];
        const child  = applyMove(root, m);
        const ch     = hashPosition(child);
        const mk     = moveKey(m);
        const single = ordered.length === 1;
        const total  = bitCount(child.p1Men|child.p1Kings|child.p2Men|child.p2Kings);
        const isQ    = m.captured.length === 0;

        let d = depth - 1;
        if (activeFeatures.extensions && single) d = Math.min(depth, d+1);
        if (activeFeatures.extensions && total <= 5) d = Math.min(depth, d+1);
        // LMR at root — skip on tactical positions (opponent has forced captures)
        const opCapRoot = isQ && hasCapturesAvailable(child);
        if (activeFeatures.extensions) d = extendTacticalDepth(depth, d, m, opCapRoot, 0);
        if (activeFeatures.extensions && rootLowMobilityExtension > 0) d = Math.min(depth + rootLowMobilityExtension, d + rootLowMobilityExtension);
        if (activeFeatures.extensions && isSoundForcedTrap(root, m)) d = Math.min(depth + 2, d + 2);
        const fullD = d;
        if (activeFeatures.lmr && i >= 3 && d >= 2 && isQ && !single && !opCapRoot) {
          d = Math.max(1, d - (LMR[Math.min(31,i)][Math.min(31,d)] | 0));
        }

        pushRepetition(rep, ch);
        let score: number;
        if (i === 0) {
          score = -negamax(child, d, -beta, -alpha, tt, deadline, acc, 1, rep, true, mk);
        } else {
          score = -negamax(child, d, -(alpha+1), -alpha, tt, deadline, acc, 1, rep, true, mk);
          if (score > alpha && score < beta)
            score = -negamax(child, fullD, -beta, -alpha, tt, deadline, acc, 1, rep, true, mk);
        }
        popRepetition(rep, ch);
        rootCandidates.push({ move: m, score });
        if (score > rootBest) {
          rootBest = score;
          rootMove = m;
          rootTieChildStatic = exactTieTiebreak ? _eval(child) : undefined;
        } else if (exactTieTiebreak && score === rootBest && rootMove) {
          const challengerChildStatic = _eval(child);
          const incumbentChildStatic = rootTieChildStatic ?? _eval(applyMove(root, rootMove));
          rootTieChildStatic = incumbentChildStatic;
          // Child eval is from the child side to move, so lower is better for the root side.
          if (challengerChildStatic < incumbentChildStatic) {
            rootMove = m;
            rootTieChildStatic = challengerChildStatic;
          }
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
      }

      nodes += acc.n;
      qnodes += acc.q;
      result = { move: rootMove, score: rootBest, candidates: rootCandidates };

      if (cancel?.cancelled || stop.flag || Date.now() > deadline) break;
      if (rootBest <= (haveLast ? lastScore - winSize : -INF)) {
        winSize = Math.min(INF, winSize * 2); alpha = Math.max(-INF, (haveLast ? lastScore : 0) - winSize); continue;
      }
      if (rootBest >= (haveLast ? lastScore + winSize : INF)) {
        winSize = Math.min(INF, winSize * 2); beta = Math.min(INF, (haveLast ? lastScore : 0) + winSize); continue;
      }
      break;
    }

    if (cancel?.cancelled || stop.flag || Date.now() > deadline) break;
    if (result.move) { best = result.move; bestScore = result.score; reached = depth; }
    if (result.candidates?.length) {
      lastRootCandidates = result.candidates;
      lastRootScore = result.score;
    }
    lastScore = result.score; haveLast = true;
    onInfo?.({ depth, score: bestScore, nodes, pv: getRootPV(root, best, tt, rep) });

    // Adaptive time: if the best move hasn't changed for 3+ depths and we've
    // used ≥50% of the time budget, the search has converged — stop early.
    const newBestKey = result.move ? moveKey(result.move) : -1;
    if (newBestKey === lastBestKey) { stableDepths++; } else { stableDepths = 0; lastBestKey = newBestKey; }
    if (!isDeterministic && stableDepths >= 3 && Date.now() > startTime + timeMs * 0.5) break;
  }

  // Disabling this override caused tactical regression during Phase B.1 experiment.
  const ENABLE_LOW_MOBILITY_RECAPTURE_OVERRIDE = true;

  if (lastRootCandidates?.length) {
    // Before adding opening variety, re-check the top root alternatives with a
    // wide window. This catches occasional aspiration / ordering artifacts at
    // the root without paying the cost for every legal move.
    if (reached >= 5 && activeNodeLimit === undefined && Date.now() + 50 < deadline && !cancel?.cancelled) {
      const verifyAcc = { n: 0, q: 0 };
      const updated = [...lastRootCandidates];
      for (const candidate of candidateWindow(lastRootCandidates, bestScore, 95)) {
        if (Date.now() > deadline || cancel?.cancelled || stop.flag) break;
        const child = applyMove(root, candidate.move);
        const ch = hashPosition(child);
        const isQ = candidate.move.captured.length === 0;
        let d = Math.max(1, reached - 1);
        const total = bitCount(child.p1Men | child.p1Kings | child.p2Men | child.p2Kings);
        if (activeFeatures.extensions && total <= 5) d = Math.min(reached, d + 1);
        if (activeFeatures.extensions) d = extendTacticalDepth(reached, d, candidate.move, isQ && hasCapturesAvailable(child), 0);
        if (activeFeatures.extensions && isSoundForcedTrap(root, candidate.move)) d = Math.min(reached + 2, d + 2);
        if (activeFeatures.extensions && candidate.move.captured.length >= 2) d = Math.min(reached + 1, d + 1);

        pushRepetition(rep, ch);
        const score = -negamax(child, d, -INF, INF, tt, deadline, verifyAcc, 1, rep, true, moveKey(candidate.move));
        popRepetition(rep, ch);
        const idx = updated.findIndex(c => moveKey(c.move) === moveKey(candidate.move));
        if (idx >= 0) updated[idx] = { move: candidate.move, score };
        if (score > bestScore) {
          best = candidate.move;
          bestScore = score;
        }
      }
      nodes += verifyAcc.n;
      qnodes += verifyAcc.q;
      lastRootCandidates = updated;
      lastRootScore = bestScore;
    }

    if (diversifyRoot) {
      rootOverrideStats.openingDiversification.attempts++;
      const selected = pickDiversifiedRoot(lastRootCandidates, lastRootScore);
      if (selected) {
        rootOverrideStats.openingDiversification.accepted++;
        best = selected.move;
        bestScore = selected.score;
        overrideReason = 'opening diversification';
        finalMoveSource = 'openingDiversification';
      }
    }

    rootOverrideStats.trapOverride.attempts++;
    const trap = pickSoundForcedTrap(root, lastRootCandidates, bestScore);
    if (trap) {
      rootOverrideStats.trapOverride.accepted++;
      best = trap.move;
      bestScore = trap.score;
      overrideReason = 'forced recapture trap';
      finalMoveSource = 'trapOverride';
    }

    const lowMobilityRecapture = ENABLE_LOW_MOBILITY_RECAPTURE_OVERRIDE
      ? (rootOverrideStats.lowMobilityRecaptureOverride.attempts++, pickLowMobilityRecaptureCandidate(root, lastRootCandidates, bestScore))
      : undefined;
    if (lowMobilityRecapture) {
      rootOverrideStats.lowMobilityRecaptureOverride.accepted++;
      best = lowMobilityRecapture.move;
      bestScore = lowMobilityRecapture.score;
      overrideReason = 'low-mobility recapture';
      finalMoveSource = 'lowMobilityRecaptureOverride';
    }

    rootOverrideStats.promotionOverride.attempts++;
    const promotion = pickEndgamePromotionCandidate(root, lastRootCandidates, bestScore);
    if (promotion) {
      rootOverrideStats.promotionOverride.accepted++;
      best = promotion.move;
      bestScore = promotion.score;
      overrideReason = 'endgame promotion race';
      finalMoveSource = 'promotionOverride';
    }

    // Tactical blunder guard at root:
    // if the top-eval move hangs an immediate multi-capture and there is a
    // near-equal safer move, prefer the safer move.
    const safer = reached >= 2
      ? (rootOverrideStats.rootTacticalSafetyOverride.attempts++, pickSaferRootCandidate(root, lastRootCandidates, bestScore, best))
      : undefined;
    if (safer) {
      rootOverrideStats.rootTacticalSafetyOverride.accepted++;
      best = safer.move;
      bestScore = safer.score;
      overrideReason = 'root tactical safety';
      finalMoveSource = 'rootTacticalSafetyOverride';
    }
  }

  // Safety fallback: if the search somehow produced no best move (e.g. time
  // expired before depth-1 completed, or threefold detected mid-search) but
  // legal moves exist, return the first one so the game never stalls.
  if (!best) {
    const fallback = generateMoves(root);
    if (fallback.length) {
      best = fallback[0];
      finalMoveSource = 'fallbackLegalMove';
    }
  }

  rootOverrideStats.antiHangSafetyOverride.attempts++;
  const antiHang = pickAbsoluteAntiHangMove(root, best);
  if (antiHang) {
    rootOverrideStats.antiHangSafetyOverride.accepted++;
    best = antiHang;
    overrideReason = 'absolute anti-hang safety';
    finalMoveSource = 'antiHangSafetyOverride';
  }

  rootOverrideStats.finalMoveSource[finalMoveSource]++;

  return {
    best,
    score: bestScore,
    nodes,
    qnodes,
    depth: reached,
    elapsedMs: Date.now() - startTime,
    timedOut: !isDeterministic && (stop.flag || Date.now() > deadline),
    pv: getRootPV(root, best, tt, rep),
    limitReached: activeNodeLimit !== undefined && activeNodeCount >= activeNodeLimit ? 'nodes' : undefined,
    overrideReason,
    rootCandidates: lastRootCandidates,
  };
}

export function fixedDepthSearch(
  root: Position,
  depth: number,
  tt = new TT(),
  historyHashes: number[] = [],
  onInfo?: OnInfo,
  featureOverrides: Partial<SearchFeatureFlags> = {},
): Promise<SearchResult> {
  if (!Number.isInteger(depth) || depth < 1) throw new Error(`depth must be a positive integer, got ${depth}`);
  return iterativeDeepening(root, 0, tt, onInfo, historyHashes, undefined, depth, undefined, false, { depth }, featureOverrides);
}

export function fixedNodeSearch(
  root: Position,
  nodes: number,
  tt = new TT(),
  historyHashes: number[] = [],
  maxDepth = 64,
  onInfo?: OnInfo,
  featureOverrides: Partial<SearchFeatureFlags> = {},
): Promise<SearchResult> {
  if (!Number.isInteger(nodes) || nodes < 1) throw new Error(`nodes must be a positive integer, got ${nodes}`);
  return iterativeDeepening(root, 0, tt, onInfo, historyHashes, undefined, maxDepth, undefined, false, { nodes }, featureOverrides);
}
