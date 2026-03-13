// src/core/perft.ts
import { initialPosition } from './position';
import { generateMoves, applyMove } from './movegen';

export function perft(depth: number): number {
  const pos = initialPosition();
  return perftRec(pos, depth);
}

function perftRec(pos: any, d: number): number {
  if (d === 0) return 1;
  let nodes = 0;
  const moves = generateMoves(pos);
  for (const m of moves) nodes += perftRec(applyMove(pos, m), d - 1);
  return nodes;
}