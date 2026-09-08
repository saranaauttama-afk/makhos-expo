// Transposition Table — typed array backend, no GC pressure
export const enum Bound { EXACT = 0, LOWER = 1, UPPER = 2 }

export interface TTEntry {
  key: number; verifyKey: number; depth: number; score: number; move?: number; bound: Bound;
}

const SIZE = 1 << 20; // 1M slots
const MASK = SIZE - 1;
const NO_MOVE = 0x3fffff;
export const TT_CAPACITY = SIZE;

export class TT {
  private data = new Int32Array(SIZE * 4); // Added slot for verifyKey
  // Keys are valid across the entire uint32 range, including 0/0. Without an
  // occupancy bit a freshly zero-filled table fabricated an EXACT key-0 hit.
  private occupied = new Uint8Array(SIZE);

  get(key: number, verifyKey: number): TTEntry | undefined {
    const i = (key & MASK) * 4;
    if (this.occupied[key & MASK] === 0) return undefined;
    if (this.data[i] !== (key | 0)) return undefined;
    if (this.data[i + 1] !== (verifyKey | 0)) return undefined; // Verify collision check
    const p = this.data[i + 2];
    return {
      key,
      verifyKey,
      depth:  (p >>> 24) & 0xff,
      bound:  ((p >>> 22) & 0x3) as Bound,
      move:   (p & NO_MOVE) === NO_MOVE ? undefined : (p & NO_MOVE),
      score:  this.data[i + 3],
    };
  }

  put(e: TTEntry): void {
    const i = (e.key & MASK) * 4;
    if (this.data[i] === (e.key | 0) && this.data[i+1] === (e.verifyKey | 0) && e.depth < ((this.data[i+2] >>> 24) & 0xff)) return;
    const mv = e.move !== undefined ? (e.move & NO_MOVE) : NO_MOVE;
    this.data[i]     = e.key | 0;
    this.data[i + 1] = e.verifyKey | 0;
    this.data[i + 2] = (Math.min(255, e.depth) << 24) | ((e.bound & 3) << 22) | mv;
    this.data[i + 3] = e.score;
    this.occupied[e.key & MASK] = 1;
  }

  clear() { this.data.fill(0); this.occupied.fill(0); }
}
