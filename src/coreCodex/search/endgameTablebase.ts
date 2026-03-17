import { bitCount } from '../bitboards';
import { applyMove, generateMoves, Move } from '../movegen';
import { isDrawByInactivity, Position } from '../position';
import {
  buildRepetitionCounts,
  getRepetitionCount,
  isThreefoldRepetition,
  popRepetition,
  pushRepetition,
  RepetitionCounts,
} from './repetition';
import { hashPosition } from './zobrist';

type Outcome = -1 | 0 | 1;

interface SolveResult {
  outcome: Outcome;
  dtm: number;
  bestMoveKey: number;
}

export interface EndgameProbe {
  score: number;
  best?: Move;
  dtm: number;
  exact: boolean;
}

const TABLEBASE_WIN = 500_000;
const NO_MOVE_KEY = -1;
const sharedMemo = new Map<string, SolveResult>();

function keyMove(move: Move) {
  return (move.from << 5) | move.to;
}

function stateKey(pos: Position, repetitionCount: number) {
  return [
    pos.side,
    pos.p1Men >>> 0,
    pos.p1Kings >>> 0,
    pos.p2Men >>> 0,
    pos.p2Kings >>> 0,
    pos.halfmoveClock,
    repetitionCount,
  ].join(':');
}

function chooseBetter(current: SolveResult | undefined, candidate: SolveResult): SolveResult {
  if (!current) return candidate;
  if (candidate.outcome !== current.outcome) {
    return candidate.outcome > current.outcome ? candidate : current;
  }
  if (candidate.outcome === 1) {
    return candidate.dtm < current.dtm ? candidate : current;
  }
  if (candidate.outcome === -1) {
    return candidate.dtm > current.dtm ? candidate : current;
  }
  return candidate.dtm < current.dtm ? candidate : current;
}

function scoreFromSolve(result: SolveResult): number {
  if (result.outcome > 0) return TABLEBASE_WIN - result.dtm;
  if (result.outcome < 0) return -TABLEBASE_WIN + result.dtm;
  return 0;
}

function canProbe(pos: Position): boolean {
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const totalMen = bitCount(pos.p1Men | pos.p2Men);
  return (totalPieces <= 4 && totalMen === 0) || totalPieces <= 3;
}

// Max recursion depth for the endgame solver.  Flying-king positions can have
// very long forced mates; cap at 60 plies to prevent stack overflow on mobile.
const SOLVE_MAX_DEPTH = 60;

function solveNode(
  pos: Position,
  repetitionCounts: RepetitionCounts,
  visiting: Set<string>,
  depth = 0,
): SolveResult {
  // Hard depth cap — return "unknown draw" rather than overflowing the stack
  if (depth >= SOLVE_MAX_DEPTH) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const hash = hashPosition(pos);
  const repCount = Math.min(3, getRepetitionCount(repetitionCounts, hash));
  const key = stateKey(pos, repCount);

  if (isDrawByInactivity(pos) || isThreefoldRepetition(repetitionCounts, hash)) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const cached = sharedMemo.get(key);
  if (cached) return cached;
  if (visiting.has(key)) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const moves = generateMoves(pos);
  if (moves.length === 0) {
    const terminal = { outcome: -1 as Outcome, dtm: 0, bestMoveKey: NO_MOVE_KEY };
    sharedMemo.set(key, terminal);
    return terminal;
  }

  visiting.add(key);
  let best: SolveResult | undefined;

  for (const move of moves) {
    const child = applyMove(pos, move);
    const childHash = hashPosition(child);
    pushRepetition(repetitionCounts, childHash);
    const childResult = solveNode(child, repetitionCounts, visiting, depth + 1);
    popRepetition(repetitionCounts, childHash);

    const candidate: SolveResult = {
      outcome: (-childResult.outcome) as Outcome,
      dtm: childResult.dtm + 1,
      bestMoveKey: keyMove(move),
    };
    best = chooseBetter(best, candidate);

    if (best.outcome === 1 && best.dtm === 1) break;
  }

  visiting.delete(key);
  const resolved = best ?? { outcome: 0 as Outcome, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  sharedMemo.set(key, resolved);
  return resolved;
}

function findBestMove(pos: Position, bestMoveKey: number): Move | undefined {
  if (bestMoveKey < 0) return undefined;
  const from = (bestMoveKey >> 5) & 31;
  const to = bestMoveKey & 31;
  return generateMoves(pos).find((move) => move.from === from && move.to === to);
}

export function probeSmallEndgame(pos: Position, historyHashes: number[] = []): EndgameProbe | undefined {
  if (!canProbe(pos)) return undefined;

  const hash = hashPosition(pos);
  const normalizedHistory = historyHashes.length ? historyHashes : [hash];
  const repetitionCounts = buildRepetitionCounts(normalizedHistory);
  if ((repetitionCounts.get(hash) ?? 0) === 0) {
    repetitionCounts.set(hash, 1);
  }

  const result = solveNode(pos, repetitionCounts, new Set<string>());
  return {
    score: scoreFromSolve(result),
    best: findBestMove(pos, result.bestMoveKey),
    dtm: result.dtm,
    exact: true,
  };
}

export function probeSmallEndgameFromCounts(pos: Position, repetitionCounts: RepetitionCounts): EndgameProbe | undefined {
  if (!canProbe(pos)) return undefined;
  const result = solveNode(pos, repetitionCounts, new Set<string>());
  return {
    score: scoreFromSolve(result),
    best: findBestMove(pos, result.bestMoveKey),
    dtm: result.dtm,
    exact: true,
  };
}
