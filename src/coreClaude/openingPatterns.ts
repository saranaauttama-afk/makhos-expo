// Thai Checkers Opening Patterns
// Based on research from traditional Thai Checkers books and online resources
// Main source: PIGGYMAN007.COM opening formulas

import { FreshOpeningBookEntry } from './search/openingBookFresh';
import { hashPosition } from './search/zobrist';
import { initialPosition, Position } from './position';
import { applyMove } from './movegen';

// Note: Thai checkers uses 32-square notation (0-31 bit indices)
// Starting position: P1 men on 21-28, P2 men on 1-8

// Helper to create position after moves
function positionAfterMoves(moves: Array<{ from: number; to: number }>): Position {
  let pos = initialPosition();
  for (const move of moves) {
    const legalMoves = require('./movegen').generateMoves(pos);
    const moveObj = legalMoves.find((m: any) => m.from === move.from && m.to === move.to);
    if (!moveObj) throw new Error(`Illegal move: ${move.from}->${move.to}`);
    pos = applyMove(pos, moveObj);
  }
  return pos;
}

export const THAI_OPENING_PATTERNS: FreshOpeningBookEntry[] = [
  // === PLY 0: Initial position - P1 opening moves ===
  {
    key: hashPosition(initialPosition()),
    ply: 0,
    side: 1,
    moves: [
      { from: 25, to: 22, weight: 100, note: 'Standard opening' },
      { from: 26, to: 23, weight: 80, note: 'Three-in-line' },
      { from: 27, to: 23, weight: 75, note: 'Dragon head' },
      { from: 24, to: 20, weight: 70, note: 'Five points' },
      { from: 25, to: 21, weight: 60, note: 'Center control' },
      { from: 26, to: 22, weight: 55, note: 'Flank development' },
      { from: 24, to: 21, weight: 50, note: 'Solid opening' },
    ],
  },

  // === PLY 1: After 25-22 (Standard opening) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 25, to: 22 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 7, to: 11, weight: 100, note: 'Standard response (from PIGGYMAN007)' },
      { from: 6, to: 10, weight: 80, note: 'Five points defense' },
      { from: 7, to: 10, weight: 75, note: 'Double corner response' },
      { from: 4, to: 8, weight: 70, note: 'Solid defense' },
    ],
  },

  // === PLY 2: After 25-22, 7-11 (Standard vs Standard) - P1 continuation ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 22, to: 18, weight: 100, note: 'Standard continuation (PIGGYMAN007)' },
      { from: 29, to: 25, weight: 80, note: 'Flexible development' },
      { from: 26, to: 23, weight: 70, note: 'Central pressure' },
    ],
  },

  // === PLY 3: After 25-22, 7-11, 22-18 - P2 continuation ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
      { from: 22, to: 18 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 4, to: 8, weight: 100, note: 'Standard line (PIGGYMAN007)' },
      { from: 6, to: 10, weight: 80, note: 'Alternative development' },
      { from: 11, to: 15, weight: 70, note: 'Push center' },
    ],
  },

  // === PLY 1: After 26-23 (Three-in-line) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 26, to: 23 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 7, to: 10, weight: 100, note: 'Standard three-in-line response' },
      { from: 7, to: 11, weight: 80, note: 'Flexible response' },
      { from: 6, to: 10, weight: 70, note: 'Solid defense' },
    ],
  },

  // === PLY 1: After 24-20 (Five points) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 24, to: 20 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 4, to: 8, weight: 100, note: 'Counter five points' },
      { from: 7, to: 11, weight: 80, note: 'Central response' },
      { from: 6, to: 10, weight: 70, note: 'Solid development' },
    ],
  },

  // === PLY 1: After 27-23 (Dragon head) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 27, to: 23 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 7, to: 10, weight: 100, note: 'Meet dragon head' },
      { from: 7, to: 11, weight: 80, note: 'Flexible response' },
      { from: 6, to: 10, weight: 70, note: 'Solid response' },
    ],
  },

  // === PLY 4: After 25-22, 7-11, 22-18, 4-8 (Standard line continues) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
      { from: 22, to: 18 },
      { from: 4, to: 8 },
    ])),
    ply: 4,
    side: 1,
    moves: [
      { from: 29, to: 25, weight: 100, note: 'Standard development' },
      { from: 26, to: 22, weight: 85, note: 'Strengthen center' },
      { from: 27, to: 23, weight: 75, note: 'Flexible option' },
    ],
  },

  // === PLY 2: After 26-23, 7-10 (Three-in-line main) - P1 continuation ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 23 },
      { from: 7, to: 10 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 23, to: 19, weight: 100, note: 'Three-in-line advance' },
      { from: 25, to: 22, weight: 85, note: 'Mixed opening' },
      { from: 29, to: 25, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 2: After 24-20, 4-8 (Five points main) - P1 continuation ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 24, to: 20 },
      { from: 4, to: 8 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 20, to: 16, weight: 100, note: 'Five points continuation' },
      { from: 25, to: 22, weight: 85, note: 'Standard development' },
      { from: 29, to: 25, weight: 70, note: 'Flexible approach' },
    ],
  },
];

export function generateOpeningBookEntries(): FreshOpeningBookEntry[] {
  return THAI_OPENING_PATTERNS;
}
