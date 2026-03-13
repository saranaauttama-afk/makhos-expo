// src/core/search/tt.ts
export enum Bound { EXACT=0, LOWER=1, UPPER=2 }
export interface TTEntry {
  key: number; depth: number; score: number; move?: number; bound: Bound; }

export class TT {
  private m = new Map<number, TTEntry>();
  get(k: number): TTEntry | undefined { return this.m.get(k); }
  put(e: TTEntry) {
    const prev = this.m.get(e.key);
    if (!prev || e.depth >= prev.depth) this.m.set(e.key, e);
  }
  clear() { this.m.clear(); }
}