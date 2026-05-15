// timeManager.ts - Adaptive time management for minimax search
//
// Allocates search time based on:
// - Game phase (opening/midgame/endgame)
// - Position complexity (legal moves, forced captures)
// - Opening book availability
// - Move stability (how often best move changes)

import { Position } from '../position';
import { bitCount } from '../bitboards';
import { generateMoves } from '../movegen';
import { lookupOpeningBook } from './openingBook';

export interface TimeAllocation {
  targetMs: number;      // Target time for this move
  minMs: number;         // Minimum time (for quick obvious moves)
  maxMs: number;         // Maximum time (for critical positions)
  phase: GamePhase;      // Game phase classification
  complexity: number;    // Position complexity score (0-1)
}

export type GamePhase = 'opening' | 'midgame' | 'endgame';

export interface TimeManagerState {
  totalTimeMs: number;        // Total time remaining
  movesPlayed: number;        // Number of moves played so far
  timePerMove: number[];      // History of time spent per move
  expectedMovesRemaining: number; // Estimated moves until game end
}

/**
 * Classify game phase based on material count
 */
function classifyGamePhase(pos: Position): GamePhase {
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);

  if (totalPieces >= 12) return 'opening';
  if (totalPieces >= 6) return 'midgame';
  return 'endgame';
}

/**
 * Calculate position complexity (0-1 scale)
 * Higher = more complex position requiring more time
 */
function calculateComplexity(pos: Position): number {
  const moves = generateMoves(pos);
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const hasForcedCaptures = moves.length > 0 && moves[0].captured.length > 0;

  let complexity = 0;

  // More legal moves = more complex
  if (moves.length >= 10) complexity += 0.4;
  else if (moves.length >= 6) complexity += 0.3;
  else if (moves.length >= 3) complexity += 0.2;
  else complexity += 0.1; // Very few moves = simpler

  // Forced captures = tactical complexity
  if (hasForcedCaptures) {
    if (moves[0].captured.length >= 2) complexity += 0.3; // Multi-captures
    else complexity += 0.2;
  }

  // More pieces = more complex
  if (totalPieces >= 12) complexity += 0.2;
  else if (totalPieces >= 8) complexity += 0.1;

  // Kings on board = more complex
  const kings = bitCount(pos.p1Kings | pos.p2Kings);
  if (kings >= 2) complexity += 0.1;

  return Math.min(1.0, complexity);
}

/**
 * Allocate time for current move based on game state
 */
export function allocateTime(
  pos: Position,
  state: TimeManagerState,
  baseTimeMs: number, // Base time budget for this move (e.g., from difficulty setting)
): TimeAllocation {
  const phase = classifyGamePhase(pos);
  const complexity = calculateComplexity(pos);

  // Check if opening book has this position
  const bookHit = lookupOpeningBook(pos, { enabled: true });

  // Base allocation depends on phase
  let phaseMultiplier = 1.0;
  switch (phase) {
    case 'opening':
      phaseMultiplier = bookHit ? 0.1 : 0.7; // Very fast if in book, otherwise normal
      break;
    case 'midgame':
      phaseMultiplier = 1.2; // Spend more time in middlegame
      break;
    case 'endgame':
      phaseMultiplier = 0.8; // Less time in endgame (often obvious or tablebase)
      break;
  }

  // Adjust based on complexity
  const complexityMultiplier = 0.5 + (complexity * 1.0); // Range: 0.5x to 1.5x

  // Calculate target time
  let targetMs = baseTimeMs * phaseMultiplier * complexityMultiplier;

  // Opening book hit = instant move (but allow minimal search for verification)
  if (bookHit) {
    targetMs = Math.min(targetMs, baseTimeMs * 0.15);
  }

  // Set min/max bounds
  const minMs = Math.max(10, baseTimeMs * 0.05);   // At least 5% of base time
  const maxMs = Math.min(baseTimeMs * 2.5, targetMs * 2); // Up to 2.5x base time

  // Clamp target to bounds
  targetMs = Math.max(minMs, Math.min(maxMs, targetMs));

  return {
    targetMs,
    minMs,
    maxMs,
    phase,
    complexity,
  };
}

/**
 * Update time manager state after a move
 */
export function updateTimeManagerState(
  state: TimeManagerState,
  timeSpentMs: number,
): TimeManagerState {
  return {
    ...state,
    totalTimeMs: Math.max(0, state.totalTimeMs - timeSpentMs),
    movesPlayed: state.movesPlayed + 1,
    timePerMove: [...state.timePerMove, timeSpentMs],
    expectedMovesRemaining: Math.max(1, state.expectedMovesRemaining - 1),
  };
}

/**
 * Create initial time manager state
 */
export function createTimeManagerState(
  totalTimeMs: number,
  expectedMoves: number = 40, // Average game length
): TimeManagerState {
  return {
    totalTimeMs,
    movesPlayed: 0,
    timePerMove: [],
    expectedMovesRemaining: expectedMoves,
  };
}

/**
 * Calculate time budget with remaining time awareness
 */
export function calculateTimeBudget(
  state: TimeManagerState,
  moveTimeMs: number, // Time per move from difficulty setting
): number {
  // If we have explicit move time (from difficulty level), use it directly
  // but adjust based on remaining time if in a timed match
  if (state.totalTimeMs === Infinity || state.totalTimeMs <= 0) {
    return moveTimeMs;
  }

  // Calculate average time per remaining move
  const avgTimePerMove = state.totalTimeMs / Math.max(1, state.expectedMovesRemaining);

  // Use the smaller of: difficulty setting or remaining time average
  // But allow up to 2x average for critical positions
  return Math.min(moveTimeMs, avgTimePerMove * 1.5);
}
