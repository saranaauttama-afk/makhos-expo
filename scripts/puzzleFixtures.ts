// Thai Checkers Tactical Puzzles (หมากกล)
// Based on research and common tactical patterns

import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';

export type PuzzleType =
  | 'forced-capture-trap'
  | 'promotion-race'
  | 'sacrifice-combination'
  | 'endgame-technique'
  | 'escape-from-trap'
  | 'king-vs-men'
  | 'tempo-gain';

export type PuzzleDifficulty = 'easy' | 'medium' | 'hard' | 'expert';

export interface PuzzleFixture {
  id: string;
  name: string;
  type: PuzzleType;
  difficulty: PuzzleDifficulty;
  pos: Position;
  expectedMove?: { from: number; to: number };  // Best move
  solution?: string;  // Description of winning line
  plysToSolve?: number;  // Expected depth to find solution
  note?: string;
}

function makePosition(fields: Partial<Position> & Pick<Position, 'side'>): Position {
  return {
    p1Men: 0,
    p1Kings: 0,
    p2Men: 0,
    p2Kings: 0,
    halfmoveClock: 0,
    ...fields,
  };
}

// Puzzle Collection
export const TACTICAL_PUZZLES: PuzzleFixture[] = [
  // === FORCED CAPTURE TRAPS (กับดักบังคับกิน) ===
  {
    id: 'trap-01-bait-sacrifice',
    name: 'Simple Bait Sacrifice',
    type: 'forced-capture-trap',
    difficulty: 'easy',
    pos: makePosition({
      side: 1,
      p1Men: B1(21) | B1(26),
      p2Men: B1(13) | B1(14),
    }),
    expectedMove: { from: 26, to: 23 },
    solution: 'Sacrifice at 23 to force double capture win',
    plysToSolve: 3,
    note: 'Basic trap pattern - bait and recapture',
  },

  {
    id: 'trap-02-double-bait',
    name: 'Double Bait Trap',
    type: 'forced-capture-trap',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Men: B1(21) | B1(26) | B1(27),
      p2Men: B1(13) | B1(14) | B1(17),
    }),
    expectedMove: { from: 27, to: 23 },
    solution: 'Create forced capture sequence leading to material advantage',
    plysToSolve: 5,
    note: 'Multiple pieces involved in trap',
  },

  // === PROMOTION RACES (แข่งขันขึ้นฮอส) ===
  {
    id: 'promo-01-race-to-king',
    name: 'Race to Promotion',
    type: 'promotion-race',
    difficulty: 'easy',
    pos: makePosition({
      side: 1,
      p1Men: B1(10),
      p2Men: B1(23),
    }),
    expectedMove: { from: 10, to: 6 },
    solution: 'Push forward to promote before opponent',
    plysToSolve: 2,
    note: 'Tempo matters in promotion races',
  },

  {
    id: 'promo-02-sacrifice-for-promotion',
    name: 'Sacrifice for Tempo',
    type: 'promotion-race',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Men: B1(10) | B1(14),
      p2Men: B1(17) | B1(23),
    }),
    expectedMove: { from: 14, to: 10 },
    solution: 'Sacrifice piece to gain tempo for promotion',
    plysToSolve: 4,
    note: 'Trading material for positional advantage',
  },

  // === KING VS MEN ENDGAMES (ฮอสต่อเบี้ย) ===
  {
    id: 'endgame-01-king-vs-two',
    name: 'King vs Two Men',
    type: 'king-vs-men',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Kings: B1(18),
      p2Men: B1(13) | B1(14),
    }),
    expectedMove: { from: 18, to: 14 },
    solution: 'King captures forcing win',
    plysToSolve: 3,
    note: 'King mobility advantage',
  },

  {
    id: 'endgame-02-king-vs-three',
    name: 'King vs Three Men (Defense)',
    type: 'king-vs-men',
    difficulty: 'hard',
    pos: makePosition({
      side: -1,
      p1Kings: B1(18),
      p2Men: B1(5) | B1(6) | B1(9),
    }),
    expectedMove: { from: 5, to: 9 },
    solution: 'Form defensive wall to hold draw',
    plysToSolve: 5,
    note: 'Defensive technique against king',
  },

  // === SACRIFICE COMBINATIONS (การเสียสละเพื่อชัยชนะ) ===
  {
    id: 'sac-01-piece-for-position',
    name: 'Piece Sacrifice for Position',
    type: 'sacrifice-combination',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Men: B1(21) | B1(22) | B1(25),
      p2Men: B1(13) | B1(14) | B1(17) | B1(18),
    }),
    expectedMove: { from: 25, to: 22 },
    solution: 'Sacrifice to break opponent structure',
    plysToSolve: 4,
    note: 'Positional sacrifice',
  },

  {
    id: 'sac-02-two-for-promotion',
    name: 'Two Pieces for Promotion',
    type: 'sacrifice-combination',
    difficulty: 'hard',
    pos: makePosition({
      side: 1,
      p1Men: B1(10) | B1(14) | B1(15),
      p2Men: B1(6) | B1(17) | B1(18) | B1(22),
    }),
    expectedMove: { from: 14, to: 10 },
    solution: 'Sacrifice two pieces to ensure promotion and win',
    plysToSolve: 6,
    note: 'Complex calculation required',
  },

  // === ESCAPE FROM TRAP (การหนีกับดัก) ===
  {
    id: 'escape-01-find-the-exit',
    name: 'Escape the Squeeze',
    type: 'escape-from-trap',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Men: B1(24) | B1(25) | B1(29),
      p2Men: B1(16) | B1(17) | B1(20),
    }),
    expectedMove: { from: 29, to: 25 },
    solution: 'Find the only move to avoid losing position',
    plysToSolve: 3,
    note: 'Low mobility requires precise play',
  },

  {
    id: 'escape-02-counter-trap',
    name: 'Counter-trap Response',
    type: 'escape-from-trap',
    difficulty: 'hard',
    pos: makePosition({
      side: 1,
      p1Men: B1(21) | B1(25) | B1(26) | B1(30),
      p2Men: B1(13) | B1(14) | B1(17) | B1(18),
    }),
    expectedMove: { from: 30, to: 26 },
    solution: 'Turn opponent trap into your advantage',
    plysToSolve: 5,
    note: 'Defensive tactics',
  },

  // === TEMPO GAIN (การแย่งจังหวะ) ===
  {
    id: 'tempo-01-force-response',
    name: 'Forced Response Win',
    type: 'tempo-gain',
    difficulty: 'easy',
    pos: makePosition({
      side: 1,
      p1Men: B1(14) | B1(18),
      p2Men: B1(10) | B1(22),
    }),
    expectedMove: { from: 18, to: 14 },
    solution: 'Force opponent into losing tempo',
    plysToSolve: 2,
    note: 'Initiative matters',
  },

  {
    id: 'tempo-02-zugzwang',
    name: 'Zugzwang Position',
    type: 'tempo-gain',
    difficulty: 'expert',
    pos: makePosition({
      side: -1,
      p1Men: B1(18) | B1(22),
      p2Men: B1(10) | B1(14),
    }),
    expectedMove: { from: 10, to: 6 },
    solution: 'Any move worsens position - find best losing move',
    plysToSolve: 4,
    note: 'Advanced positional concept',
  },

  // === ENDGAME TECHNIQUES (เทคนิคท้ายเกม) ===
  {
    id: 'endgame-03-opposition',
    name: 'Opposition Technique',
    type: 'endgame-technique',
    difficulty: 'medium',
    pos: makePosition({
      side: 1,
      p1Men: B1(18),
      p2Men: B1(14),
    }),
    expectedMove: { from: 18, to: 14 },
    solution: 'Maintain opposition to force win',
    plysToSolve: 3,
    note: 'Chess-like opposition concept',
  },

  {
    id: 'endgame-04-triangulation',
    name: 'Triangulation Winning',
    type: 'endgame-technique',
    difficulty: 'hard',
    pos: makePosition({
      side: 1,
      p1Kings: B1(18),
      p2Men: B1(6) | B1(10),
    }),
    expectedMove: { from: 18, to: 22 },
    solution: 'Use triangulation to zugzwang opponent',
    plysToSolve: 7,
    note: 'Advanced king technique',
  },
];

export function getPuzzlesByDifficulty(difficulty: PuzzleDifficulty): PuzzleFixture[] {
  return TACTICAL_PUZZLES.filter(p => p.difficulty === difficulty);
}

export function getPuzzlesByType(type: PuzzleType): PuzzleFixture[] {
  return TACTICAL_PUZZLES.filter(p => p.type === type);
}

export function getPuzzle(id: string): PuzzleFixture | undefined {
  return TACTICAL_PUZZLES.find(p => p.id === id);
}
