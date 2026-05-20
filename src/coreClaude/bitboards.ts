// src/core/bitboards.ts
// 32-square dark-tile indexing for checkers (row-major on dark squares)
// Index map (8x8): rows 0..7 from top; dark squares only -> 0..31
// We'll programmatically generate mapping to avoid off-by-one bugs.

export type BB = number; // use uint32 via >>> 0 for ops
export const B1 = (i: number): BB => (1 << i) >>> 0;

export interface RC { r: number; c: number; }

// Build mapping between 8x8 dark squares and 0..31 index
const darkSquares: number[] = []; // length 32, holds (r*8+c)
const SQUARE_TO_RC: RC[] = new Array(32);
const RC_TO_INDEX = new Int16Array(64).fill(-1);

(function buildMaps() {
  let idx = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const dark = ((r + c) & 1) === 1;
      if (dark) {
        const flat = r * 8 + c;
        darkSquares.push(flat);
        SQUARE_TO_RC[idx] = { r, c };
        RC_TO_INDEX[flat] = idx;
        idx++;
      }
    }
  }
})();

export const toRC = (i: number): RC => SQUARE_TO_RC[i];
export const toIndex = (r: number, c: number): number => {
  const flat = r * 8 + c;
  return RC_TO_INDEX[flat];
};

// Precompute single-step and jump targets per square
export type Dir = 'UL' | 'UR' | 'DL' | 'DR';
export interface Step { to: number; dir: Dir }
export interface Jump { over: number; to: number; dir: Dir }

export const STEPS: Step[][] = Array.from({ length: 32 }, () => []);
export const JUMPS: Jump[][] = Array.from({ length: 32 }, () => []);

(function buildAdjacency() {
  const dirs: [Dir, number, number][] = [
    ['UL', -1, -1],
    ['UR', -1, +1],
    ['DL', +1, -1],
    ['DR', +1, +1],
  ];

  for (let i = 0; i < 32; i++) {
    const { r, c } = toRC(i);
    for (const [dir, dr, dc] of dirs) {
      const r1 = r + dr, c1 = c + dc;
      const r2 = r + 2 * dr, c2 = c + 2 * dc;
      const stepIdx = (r1 >= 0 && r1 < 8 && c1 >= 0 && c1 < 8) ? toIndex(r1, c1) : -1;
      if (stepIdx >= 0) STEPS[i].push({ to: stepIdx, dir });
      const overIdx = stepIdx;
      const jumpIdx = (r2 >= 0 && r2 < 8 && c2 >= 0 && c2 < 8) ? toIndex(r2, c2) : -1;
      if (overIdx >= 0 && jumpIdx >= 0) JUMPS[i].push({ over: overIdx, to: jumpIdx, dir });
    }
  }
})();

export function bitCount(x: BB): number {
  x = x >>> 0;
  let c = 0;
  while (x) { x &= (x - 1) >>> 0; c++; }
  return c;
}

export function *bits(bb: BB): Iterable<number> {
  // Handle both number and BigInt
  let x = typeof bb === 'bigint' ? Number(bb & 0xFFFFFFFFn) : (bb >>> 0);
  while (x) {
    const lsb = x & -x;                 // lowest set bit
    const i = 31 - Math.clz32(lsb);     // <-- ที่ถูกต้อง (เดิมใช้ ^ 31)
    yield i;
    x = (x ^ lsb) >>> 0;
  }
}

// ── Fast directional tables for movegen optimization (Phase 3) ──────────────
export const DIRS = ['UL', 'UR', 'DL', 'DR'] as const;
export type DirIndex = typeof DIRS[number];

export const DIR_INDEX: Record<Dir, number> = {
  UL: 0,
  UR: 1,
  DL: 2,
  DR: 3,
};

// NEXT[sq][dirIndex] = next square in that direction, or -1 if off-board
export const NEXT: Int8Array[] = Array.from({ length: 32 }, () => new Int8Array([-1, -1, -1, -1]));

// RAYS[sq][dirIndex] = list of all squares outward in that direction (for king fly)
export const RAYS: number[][][] = Array.from({ length: 32 }, () =>
  Array.from({ length: 4 }, () => [] as number[])
);

(function buildFastDirectionalTables() {
  const dirs: [number, number][] = [
    [-1, -1], // UL
    [-1, +1], // UR
    [+1, -1], // DL
    [+1, +1], // DR
  ];

  for (let sq = 0; sq < 32; sq++) {
    const { r, c } = toRC(sq);

    for (let d = 0; d < 4; d++) {
      const [dr, dc] = dirs[d];

      // NEXT: just one step
      let nr = r + dr;
      let nc = c + dc;
      if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
        const nextIdx = toIndex(nr, nc);
        if (nextIdx >= 0) {
          NEXT[sq][d] = nextIdx;
        }
      }

      // RAYS: all steps outward until edge
      nr = r + dr;
      nc = c + dc;
      while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
        const idx = toIndex(nr, nc);
        if (idx >= 0) RAYS[sq][d].push(idx);
        nr += dr;
        nc += dc;
      }
    }
  }
})();
