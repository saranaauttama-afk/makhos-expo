import { bitCount } from '../bitboards';
import { applyMove, generateMoves, Move } from '../movegen';
import { Position } from '../position';
import { moveKey, SearchResult } from './alphabeta';

export type StrictDifficulty = 'easy' | 'normal' | 'hard' | 'expert';
export type EngineDifficulty = StrictDifficulty | 'master';

export interface StrictLevelPolicy {
  baseDepth: number;
  depthCap: number;
  baseBudgetMs: number;
  budgetCapMs: number;
}

export const STRICT_LEVELS: StrictDifficulty[] = ['easy', 'normal', 'hard', 'expert'];

export const STRICT_LEVEL_POLICY: Record<StrictDifficulty, StrictLevelPolicy> = {
  easy: { baseDepth: 3, depthCap: 6, baseBudgetMs: 500, budgetCapMs: 1400 },
  normal: { baseDepth: 6, depthCap: 9, baseBudgetMs: 1200, budgetCapMs: 3200 },
  hard: { baseDepth: 9, depthCap: 11, baseBudgetMs: 2800, budgetCapMs: 4500 },
  expert: { baseDepth: 12, depthCap: 14, baseBudgetMs: 5600, budgetCapMs: 7600 },
};

export function isStrictDifficulty(difficulty: EngineDifficulty): difficulty is StrictDifficulty {
  return difficulty !== 'master';
}

export function countTotalPieces(pos: Position): number {
  return bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
}

export function pickAdaptiveStrictDepth(difficulty: StrictDifficulty, pos: Position): number {
  const policy = STRICT_LEVEL_POLICY[difficulty];
  const moves = generateMoves(pos);
  if (moves.length <= 1) return Math.min(policy.depthCap, policy.baseDepth + 1);

  const forcedCapture = moves[0].captured.length > 0;
  const hasMultiCapture = forcedCapture && moves.some(m => m.captured.length >= 2);
  const total = countTotalPieces(pos);

  let bonus = 0;
  if (forcedCapture) bonus += 1;
  if (hasMultiCapture) bonus += 1;
  if (total <= 10) bonus += 1;
  if (total <= 7) bonus += 1;

  return Math.min(policy.depthCap, policy.baseDepth + bonus);
}

export function pickAdaptiveStrictBudgetMs(difficulty: StrictDifficulty, pos: Position): number {
  const policy = STRICT_LEVEL_POLICY[difficulty];
  const moves = generateMoves(pos);
  if (moves.length <= 1) return Math.min(policy.budgetCapMs, policy.baseBudgetMs + 100);

  const forcedCapture = moves[0].captured.length > 0;
  const hasMultiCapture = forcedCapture && moves.some(m => m.captured.length >= 2);
  const total = countTotalPieces(pos);
  const lowMobility = moves.length <= 3;

  let extra = 0;
  if (forcedCapture) extra += 220;
  if (hasMultiCapture) extra += 280;
  if (lowMobility) extra += difficulty === 'easy' ? 520 : difficulty === 'normal' ? 900 : 180;
  if (total <= 10) extra += 220;
  if (total <= 7) extra += 280;

  return Math.min(policy.budgetCapMs, policy.baseBudgetMs + extra);
}

export function pickStrictHintBudgetMs(difficulty: StrictDifficulty, minMs: number, maxMs: number): number {
  return Math.min(maxMs, Math.max(minMs, STRICT_LEVEL_POLICY[difficulty].baseBudgetMs + 1000));
}

export function pickStrictHintDepth(difficulty: StrictDifficulty): number {
  return Math.min(13, STRICT_LEVEL_POLICY[difficulty].baseDepth + 2);
}

function immediateCaptureRisk(pos: Position, move: Move): number {
  const child = applyMove(pos, move);
  const replies = generateMoves(child);
  if (!replies.length || replies[0].captured.length === 0) return 0;

  let maxCap = 0;
  let hangsMovedPiece = false;
  for (const reply of replies) {
    maxCap = Math.max(maxCap, reply.captured.length);
    if (reply.captured.includes(move.to)) hangsMovedPiece = true;
  }
  return maxCap * 140 + (maxCap >= 2 ? 180 : 0) + (hangsMovedPiece ? 320 : 0);
}

function stablePickIndex(pos: Position, level: StrictDifficulty, count: number): number {
  const salt = level === 'easy' ? 0x9e37 : level === 'normal' ? 0x51ed : 0x2c1b;
  const raw = (
    (pos.p1Men * 31) ^
    (pos.p1Kings * 131) ^
    (pos.p2Men * 8191) ^
    (pos.p2Kings * 524287) ^
    (pos.side * salt) ^
    (pos.halfmoveClock * 17)
  ) >>> 0;
  return count <= 1 ? 0 : raw % count;
}

export function selectStrictLevelMove(
  difficulty: StrictDifficulty,
  pos: Position,
  result: SearchResult,
): Move | undefined {
  const best = result.best;
  if (!best || difficulty === 'hard' || difficulty === 'expert') return best;

  const legal = generateMoves(pos);
  if (legal.length <= 3) return best;
  if (legal[0].captured.length > 0) return best;
  if (result.overrideReason) return best;

  const total = countTotalPieces(pos);
  if (total <= 5) return best;

  const candidates = result.rootCandidates;
  if (!candidates || candidates.length < 3) return best;

  const bestKey = moveKey(best);
  const bestScore = candidates.find(candidate => moveKey(candidate.move) === bestKey)?.score
    ?? Math.max(...candidates.map(candidate => candidate.score));

  const margin = difficulty === 'easy' ? 260 : difficulty === 'normal' ? 45 : 45;
  const maxRisk = difficulty === 'easy' ? 110 : difficulty === 'normal' ? 70 : 60;
  const poolSize = difficulty === 'easy' ? 5 : difficulty === 'normal' ? 2 : 2;
  const rankBias = difficulty === 'easy' ? 1 : 0;

  const pool = [...candidates]
    .filter(candidate => {
      if (candidate.move.captured.length > 0) return false;
      if (candidate.score < bestScore - margin) return false;
      if (immediateCaptureRisk(pos, candidate.move) > maxRisk) return false;
      return true;
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, poolSize);

  if (pool.length < 2) return best;
  const idx = Math.min(pool.length - 1, stablePickIndex(pos, difficulty, pool.length) + rankBias);
  return pool[idx]?.move ?? best;
}
