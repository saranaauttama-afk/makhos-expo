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

  // === PLY 5: After 25-22, 7-11, 22-18, 4-8, 29-25 (Standard line deep) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
      { from: 22, to: 18 },
      { from: 4, to: 8 },
      { from: 29, to: 25 },
    ])),
    ply: 5,
    side: -1,
    moves: [
      { from: 11, to: 15, weight: 100, note: 'Push center' },
      { from: 6, to: 10, weight: 85, note: 'Solid development' },
      { from: 3, to: 7, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 3: After 26-23, 7-10, 23-19 (Three-in-line advance) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 10, to: 14, weight: 100, note: 'Counter advance' },
      { from: 6, to: 10, weight: 80, note: 'Solid' },
      { from: 4, to: 8, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 3: After 24-20, 4-8, 20-16 (Five points continuation) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 24, to: 20 },
      { from: 4, to: 8 },
      { from: 20, to: 16 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 8, to: 12, weight: 100, note: 'Counter five points' },
      { from: 6, to: 10, weight: 80, note: 'Solid development' },
      { from: 7, to: 11, weight: 70, note: 'Central play' },
    ],
  },

  // === PLY 2: After 25-22, 6-10 (Alternative P2 response) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 6, to: 10 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 22, to: 18, weight: 100, note: 'Standard continuation' },
      { from: 29, to: 25, weight: 85, note: 'Development' },
      { from: 26, to: 23, weight: 70, note: 'Central pressure' },
    ],
  },

  // === PLY 2: After 27-23, 7-10 (Dragon head main) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 27, to: 23 },
      { from: 7, to: 10 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 23, to: 19, weight: 100, note: 'Dragon advance' },
      { from: 25, to: 22, weight: 85, note: 'Standard development' },
      { from: 29, to: 25, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 3: After 27-23, 7-10, 23-19 (Dragon head advance) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 27, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 10, to: 14, weight: 100, note: 'Meet advance' },
      { from: 6, to: 10, weight: 80, note: 'Solid' },
      { from: 4, to: 8, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 1: After 25-21 (Center control) - P2 response ===
  {
    key: hashPosition(positionAfterMoves([{ from: 25, to: 21 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 6, to: 10, weight: 100, note: 'Standard response' },
      { from: 7, to: 11, weight: 85, note: 'Flexible' },
      { from: 4, to: 8, weight: 70, note: 'Solid' },
    ],
  },

  // === PLY 6: After 25-22, 7-11, 22-18, 4-8, 29-25, 11-15 (Standard line PLY 6) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
      { from: 22, to: 18 },
      { from: 4, to: 8 },
      { from: 29, to: 25 },
      { from: 11, to: 15 },
    ])),
    ply: 6,
    side: 1,
    moves: [
      { from: 26, to: 22, weight: 100, note: 'Develop flank' },
      { from: 27, to: 23, weight: 85, note: 'Dragon head development' },
      { from: 24, to: 20, weight: 75, note: 'Five points approach' },
    ],
  },

  // === PLY 4: After 26-23, 7-10, 23-19, 10-14 (Three-in-line P2 counter) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
      { from: 10, to: 14 },
    ])),
    ply: 4,
    side: 1,
    moves: [
      { from: 19, to: 15, weight: 100, note: 'Continue advance' },
      { from: 25, to: 22, weight: 85, note: 'Develop standard' },
      { from: 29, to: 25, weight: 75, note: 'Reinforce' },
    ],
  },

  // === PLY 5: After 26-23, 7-10, 23-19, 10-14, 19-15 (Three-in-line deep) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
      { from: 10, to: 14 },
      { from: 19, to: 15 },
    ])),
    ply: 5,
    side: -1,
    moves: [
      { from: 10, to: 19, weight: 100, note: 'Capture advance' },
      { from: 6, to: 10, weight: 80, note: 'Develop side' },
      { from: 4, to: 8, weight: 70, note: 'Solid development' },
    ],
  },

  // === PLY 4: After 24-20, 4-8, 20-16, 8-12 (Five points P2 counter) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 24, to: 20 },
      { from: 4, to: 8 },
      { from: 20, to: 16 },
      { from: 8, to: 12 },
    ])),
    ply: 4,
    side: 1,
    moves: [
      { from: 16, to: 11, weight: 100, note: 'Five points deep push' },
      { from: 25, to: 22, weight: 85, note: 'Standard development' },
      { from: 29, to: 25, weight: 75, note: 'Flexible' },
    ],
  },

  // === PLY 5: After 24-20, 4-8, 20-16, 8-12, 16-11 (Five points deep) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 24, to: 20 },
      { from: 4, to: 8 },
      { from: 20, to: 16 },
      { from: 8, to: 12 },
      { from: 16, to: 11 },
    ])),
    ply: 5,
    side: -1,
    moves: [
      { from: 7, to: 16, weight: 100, note: 'Capture five points' },
      { from: 6, to: 10, weight: 80, note: 'Develop before capture' },
      { from: 12, to: 16, weight: 70, note: 'Push before capture' },
    ],
  },

  // === PLY 4: After 27-23, 7-10, 23-19, 10-14 (Dragon head P2 counter) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 27, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
      { from: 10, to: 14 },
    ])),
    ply: 4,
    side: 1,
    moves: [
      { from: 19, to: 15, weight: 100, note: 'Dragon deep advance' },
      { from: 25, to: 22, weight: 85, note: 'Standard development' },
      { from: 29, to: 25, weight: 75, note: 'Flexible approach' },
    ],
  },

  // === PLY 3: After 25-22, 6-10, 22-18 (Alternative P2 - P1 continues) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 6, to: 10 },
      { from: 22, to: 18 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 7, to: 11, weight: 100, note: 'Central development' },
      { from: 4, to: 8, weight: 85, note: 'Solid approach' },
      { from: 10, to: 14, weight: 70, note: 'Push flank' },
    ],
  },

  // === PLY 2: After 25-21, 6-10 (Center control main response) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 21 },
      { from: 6, to: 10 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 21, to: 17, weight: 100, note: 'Center control advance' },
      { from: 26, to: 23, weight: 85, note: 'Three-in-line development' },
      { from: 29, to: 25, weight: 75, note: 'Standard development' },
    ],
  },

  // === PLY 1: After 26-22 (Flank development) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 26, to: 22 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 6, to: 10, weight: 100, note: 'Standard flank response' },
      { from: 7, to: 11, weight: 85, note: 'Central response' },
      { from: 4, to: 8, weight: 70, note: 'Solid response' },
    ],
  },

  // === PLY 1: After 24-21 (Solid opening) - P2 responses ===
  {
    key: hashPosition(positionAfterMoves([{ from: 24, to: 21 }])),
    ply: 1,
    side: -1,
    moves: [
      { from: 4, to: 8, weight: 100, note: 'Meet solid with solid' },
      { from: 7, to: 11, weight: 85, note: 'Central approach' },
      { from: 6, to: 10, weight: 70, note: 'Flank development' },
    ],
  },

  // === PLY 2: After 26-22, 6-10 (Flank development main) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 22 },
      { from: 6, to: 10 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 25, to: 22, weight: 100, note: 'Standard development' },
      { from: 29, to: 25, weight: 85, note: 'Flexible approach' },
      { from: 27, to: 23, weight: 70, note: 'Dragon head mix' },
    ],
  },

  // === PLY 2: After 24-21, 4-8 (Solid opening main) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 24, to: 21 },
      { from: 4, to: 8 },
    ])),
    ply: 2,
    side: 1,
    moves: [
      { from: 21, to: 17, weight: 100, note: 'Solid advance' },
      { from: 25, to: 22, weight: 85, note: 'Standard development' },
      { from: 29, to: 25, weight: 75, note: 'Flexible' },
    ],
  },

  // === PLY 3: After 25-21, 6-10, 21-17 (Center control advance) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 21 },
      { from: 6, to: 10 },
      { from: 21, to: 17 },
    ])),
    ply: 3,
    side: -1,
    moves: [
      { from: 7, to: 11, weight: 100, note: 'Central development' },
      { from: 10, to: 14, weight: 85, note: 'Push flank' },
      { from: 4, to: 8, weight: 70, note: 'Solid response' },
    ],
  },

  // === PLY 7: After 25-22, 7-11, 22-18, 4-8, 29-25, 11-15, 26-22 (Standard line PLY 7) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 25, to: 22 },
      { from: 7, to: 11 },
      { from: 22, to: 18 },
      { from: 4, to: 8 },
      { from: 29, to: 25 },
      { from: 11, to: 15 },
      { from: 26, to: 22 },
    ])),
    ply: 7,
    side: -1,
    moves: [
      { from: 6, to: 10, weight: 100, note: 'Complete development' },
      { from: 8, to: 12, weight: 85, note: 'Push center' },
      { from: 3, to: 7, weight: 70, note: 'Flexible' },
    ],
  },

  // === PLY 6: After 26-23, 7-10, 23-19, 10-14, 19-15, 10-19 (Three-in-line capture) ===
  {
    key: hashPosition(positionAfterMoves([
      { from: 26, to: 23 },
      { from: 7, to: 10 },
      { from: 23, to: 19 },
      { from: 10, to: 14 },
      { from: 19, to: 15 },
      { from: 10, to: 19 },
    ])),
    ply: 6,
    side: 1,
    moves: [
      { from: 29, to: 25, weight: 100, note: 'Development after capture' },
      { from: 25, to: 22, weight: 85, note: 'Standard approach' },
      { from: 27, to: 23, weight: 70, note: 'Flexible' },
    ],
  },
];

export function generateOpeningBookEntries(): FreshOpeningBookEntry[] {
  return THAI_OPENING_PATTERNS;
}
