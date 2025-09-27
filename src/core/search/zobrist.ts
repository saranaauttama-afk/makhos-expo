// src/core/search/zobrist.ts
import { BB, bits } from '../bitboards';
import { Position } from '../position';

// Zobrist: [pieceType(0..3)][square(0..31)] and sideToMove
// Types: 0=P1 man, 1=P1 king, 2=P2 man, 3=P2 king

const R = (seed: number) => () => {
  let t = (seed = (seed + 0x6D2B79F5) >>> 0);
  t = Math.imul(t ^ (t >>> 15), 1 | t);
  t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const rand32 = (rng: () => number) => (rng() * 0x100000000) >>> 0;

const rng = R(0xC0FFEE);
export const Z_PIECE: number[][] = Array.from({ length: 4 }, () => Array(32).fill(0));
export const Z_SIDE: number = rand32(rng);

(function init() {
  for (let t = 0; t < 4; t++) {
    for (let s = 0; s < 32; s++) Z_PIECE[t][s] = rand32(rng);
  }
})();

export function hashPosition(p: Position): number {
  let h = 0 >>> 0;
  for (const i of bits(p.p1Men))   h ^= Z_PIECE[0][i];
  for (const i of bits(p.p1Kings)) h ^= Z_PIECE[1][i];
  for (const i of bits(p.p2Men))   h ^= Z_PIECE[2][i];
  for (const i of bits(p.p2Kings)) h ^= Z_PIECE[3][i];
  if (p.side === 1) h ^= Z_SIDE;
  return h >>> 0;
}
