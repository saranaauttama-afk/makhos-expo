// Transposition Table — typed array backend, no GC pressure
export const enum Bound { EXACT = 0, LOWER = 1, UPPER = 2 }

export interface TTEntry {
  key: number; depth: number; score: number; move?: number; bound: Bound;
}

const SIZE = 1 << 20; // 1M slots
const MASK = SIZE - 1;
const NO_MOVE = 0x3fffff;

export class TT {
  private data = new Int32Array(SIZE * 3);

  get(key: number): TTEntry | undefined {
    const i = (key & MASK) * 3;
    if (this.data[i] !== (key | 0)) return undefined;
    const p = this.data[i + 1];
    return {
      key,
      depth:  (p >>> 24) & 0xff,
      bound:  ((p >>> 22) & 0x3) as Bound,
      move:   (p & NO_MOVE) === NO_MOVE ? undefined : (p & NO_MOVE),
      score:  this.data[i + 2],
    };
  }

  put(e: TTEntry): void {
    const i = (e.key & MASK) * 3;
    if (this.data[i] === (e.key | 0) && e.depth < ((this.data[i+1] >>> 24) & 0xff)) return;
    const mv = e.move !== undefined ? (e.move & NO_MOVE) : NO_MOVE;
    this.data[i]     = e.key | 0;
    this.data[i + 1] = (Math.min(255, e.depth) << 24) | ((e.bound & 3) << 22) | mv;
    this.data[i + 2] = e.score;
  }

  clear() { this.data.fill(0); }
}
