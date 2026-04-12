import { bitCount } from '../bitboards';
import { azBestMove } from '../azMcts';
import { Move, generateMoves } from '../movegen';
import { Position } from '../position';
import { CancelToken, iterativeDeepening, SearchInfo } from './alphabeta';
import { lookupOpeningBook } from './openingBook';
import { TT } from './tt';

export type HybridDifficulty = 'easy' | 'medium' | 'hard';
export type HybridMode = 'book' | 'az' | 'alphabeta' | 'forced';

export interface HybridPlan {
  mode: HybridMode;
  reason: string;
}

export interface HybridResult {
  move?: Move;
  info: SearchInfo | null;
  plan: HybridPlan;
}

const BOOK_SKIP_RATE: Record<HybridDifficulty, number> = {
  easy: 0.70,
  medium: 0.35,
  hard: 0.15,
};

const AZ_SIMS: Record<HybridDifficulty, number> = {
  easy: 80,
  medium: 160,
  hard: 280,
};

const AB_BUDGET_MS: Record<HybridDifficulty, { tactical: number; endgame: number }> = {
  easy: { tactical: 500, endgame: 700 },
  medium: { tactical: 1200, endgame: 1600 },
  hard: { tactical: 2200, endgame: 3200 },
};

function pieceCounts(pos: Position) {
  const men = bitCount(pos.p1Men | pos.p2Men);
  const kings = bitCount(pos.p1Kings | pos.p2Kings);
  return { men, kings, total: men + kings };
}

function classifyPosition(pos: Position, difficulty: HybridDifficulty, historyHashes: number[]): HybridPlan {
  const moves = generateMoves(pos);
  if (moves.length <= 1) return { mode: 'forced', reason: 'single legal move' };

  const ply = Math.max(0, historyHashes.length - 1);
  const { men, kings, total } = pieceCounts(pos);
  const allKings = men === 0;
  const forcedCapture = moves[0].captured.length > 0;
  const multiCapture = forcedCapture && moves.some(m => m.captured.length >= 2);
  const lowMobility = moves.length <= 3;
  const promotionRace = moves.some(m => m.promote);
  const endgame = total <= 8 || (allKings && total <= 10);

  if (lookupOpeningBook(pos) && ply <= 10 && Math.random() >= BOOK_SKIP_RATE[difficulty]) {
    return { mode: 'book', reason: 'opening book hit' };
  }

  if (endgame) {
    return { mode: 'alphabeta', reason: allKings ? 'all-kings endgame' : 'small-piece endgame' };
  }

  if (forcedCapture && (multiCapture || lowMobility || promotionRace)) {
    return { mode: 'alphabeta', reason: 'forced tactical sequence' };
  }

  if (lowMobility && total <= 12) {
    return { mode: 'alphabeta', reason: 'low-mobility technical position' };
  }

  return { mode: 'az', reason: 'midgame pattern search' };
}

function makeSyntheticInfo(move: Move | undefined, mode: HybridMode, reason: string, nodes: number): SearchInfo | null {
  if (!move) return null;
  const score = mode === 'book' ? 0 : mode === 'forced' ? 1 : 0;
  return { depth: 0, score, nodes, pv: [move] };
}

export async function hybridBestMove(
  pos: Position,
  ms: number,
  tt: TT,
  historyHashes: number[] = [],
  cancel?: CancelToken,
  difficulty: HybridDifficulty = 'medium',
): Promise<HybridResult> {
  const plan = classifyPosition(pos, difficulty, historyHashes);
  const moves = generateMoves(pos);

  if (moves.length === 0) {
    return { move: undefined, info: null, plan };
  }

  if (plan.mode === 'forced') {
    return {
      move: moves[0],
      info: makeSyntheticInfo(moves[0], 'forced', plan.reason, 1),
      plan,
    };
  }

  if (plan.mode === 'book') {
    const bookHit = lookupOpeningBook(pos);
    return {
      move: bookHit?.move,
      info: makeSyntheticInfo(bookHit?.move, 'book', plan.reason, 0),
      plan,
    };
  }

  if (plan.mode === 'alphabeta') {
    const budget = plan.reason.includes('endgame')
      ? AB_BUDGET_MS[difficulty].endgame
      : AB_BUDGET_MS[difficulty].tactical;
    const result = await iterativeDeepening(pos, Math.max(ms, budget), tt, undefined, historyHashes, cancel);
    return {
      move: result.best,
      info: result.depth > 0 ? { depth: result.depth, score: result.score, nodes: result.nodes, pv: result.best ? [result.best] : [] } : null,
      plan,
    };
  }

  const move = await azBestMove(pos, AZ_SIMS[difficulty]);
  return {
    move,
    info: makeSyntheticInfo(move, 'az', plan.reason, AZ_SIMS[difficulty]),
    plan,
  };
}
