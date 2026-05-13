// Thai Checkers Opening Patterns
// Based on research from traditional Thai Checkers books and online resources
// Main source: PIGGYMAN007.COM opening formulas

import { FreshOpeningBookEntry } from './search/openingBookFresh';
import { hashPosition } from './search/zobrist';
import { initialPosition } from './position';

// Note: Thai checkers uses 32-square notation (0-31 bit indices)
// Starting position: P1 men on 21-28, P2 men on 1-8

export const THAI_OPENING_PATTERNS: FreshOpeningBookEntry[] = [
  // Opening move options from starting position
  {
    key: hashPosition(initialPosition()),
    ply: 0,
    side: 1,
    moves: [
      // Standard opening (มาตรฐาน): 25-22
      { from: 25, to: 22, weight: 100, note: 'Standard opening' },
      // Three-in-line (สามตัวเรียง): 26-23
      { from: 26, to: 23, weight: 80, note: 'Three-in-line' },
      // Dragon head variation: 27-23
      { from: 27, to: 23, weight: 75, note: 'Dragon head' },
      // Five points (ห้าแต้ม): 24-20
      { from: 24, to: 20, weight: 70, note: 'Five points' },
      // Center control: 25-21
      { from: 25, to: 21, weight: 60, note: 'Center control' },
      // Flank development: 26-22
      { from: 26, to: 22, weight: 55, note: 'Flank development' },
      // Solid: 24-21
      { from: 24, to: 21, weight: 50, note: 'Solid opening' },
    ],
  },
];

// TODO: Add subsequent moves for each opening line
// This is a minimal viable opening book - can be expanded with:
// - Response patterns for each opening
// - Multi-ply continuations
// - Common trap setups from research

export function generateOpeningBookEntries(): FreshOpeningBookEntry[] {
  return THAI_OPENING_PATTERNS;
}
