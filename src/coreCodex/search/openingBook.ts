import { Move, generateMoves } from '../movegen';
import { Position } from '../position';
import { hashPosition } from './zobrist';

interface OpeningBookEntry {
  hash: number;
  move: { from: number; to: number };
  label: string;
}

const BOOK: OpeningBookEntry[] = [
  { hash: 2120013542, move: { from: 26, to: 22 }, label: 'Codex seed 1' },
  { hash: 2943209601, move: { from: 6, to: 9 }, label: 'Codex seed 2' },
  { hash: 3504885778, move: { from: 31, to: 26 }, label: 'Codex seed 3' },
  { hash: 3658850790, move: { from: 2, to: 6 }, label: 'Codex seed 4' },
  { hash: 2265423568, move: { from: 22, to: 17 }, label: 'Codex seed 5' },
  { hash: 1383768184, move: { from: 9, to: 13 }, label: 'Codex seed 6' },
  { hash: 1925486754, move: { from: 17, to: 8 }, label: 'Codex seed 7' },
  { hash: 205381744, move: { from: 5, to: 12 }, label: 'Codex seed 8' },
];

export function lookupOpeningBook(pos: Position): { move: Move; label: string } | undefined {
  const hash = hashPosition(pos);
  const entry = BOOK.find((candidate) => candidate.hash === hash);
  if (!entry) return undefined;

  const move = generateMoves(pos).find(
    (candidate) => candidate.from === entry.move.from && candidate.to === entry.move.to,
  );
  if (!move) return undefined;
  return { move, label: entry.label };
}
