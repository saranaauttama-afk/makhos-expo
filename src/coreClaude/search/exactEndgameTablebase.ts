import { createHash } from 'crypto';
import { B1, bitCount } from '../bitboards';
import { applyMove, generateMoves, Move } from '../movegen';
import { isDrawByInactivity, Position, Side } from '../position';
import { buildRepetitionCounts, isThreefoldRepetition } from './repetition';
import { hashPosition } from './zobrist';

/** This module is deliberately not imported by production search. */
export const EXACT_TABLEBASE_VERSION = 'makhos-board-theoretic-2-3-v1';
export const CANONICAL_ENCODING_VERSION = 'makhos-board-v1';
export type CanonicalOutcome = 'WIN' | 'DRAW' | 'LOSS';

export interface CanonicalEntry {
  key: string;
  outcome: CanonicalOutcome;
  dtm?: number;
  bestMoveKeys: string[];
  materialSignature: string;
}

export interface CanonicalBuild {
  entries: Map<string, CanonicalEntry>;
  positions: Map<string, Position>;
  fingerprint: string;
  counts: Record<CanonicalOutcome, number>;
  materialCounts: Record<string, number>;
  maxDtm: number;
  generationMs: number;
  estimatedCompactBytes: number;
}

export interface HistoryAwareExactResult {
  status: 'EXACT' | 'UNKNOWN';
  outcome?: CanonicalOutcome;
  reason: 'inactivity' | 'repetition' | 'terminal' | 'mate-in-one' | 'history-can-intervene';
  dtm?: number;
  best?: Move;
}

const promotionIllegal = (category: number, square: number) =>
  (category === 0 && square < 4) || (category === 2 && square >= 28);

export function canonicalStateKey(pos: Position): string {
  return `${CANONICAL_ENCODING_VERSION}:${pos.side}:${pos.p1Men >>> 0}:${pos.p1Kings >>> 0}:${pos.p2Men >>> 0}:${pos.p2Kings >>> 0}`;
}

export function materialSignature(pos: Position): string {
  return `P1:${bitCount(pos.p1Men)}M${bitCount(pos.p1Kings)}K-P2:${bitCount(pos.p2Men)}M${bitCount(pos.p2Kings)}K`;
}

export function canonicalMoveKey(move: Move): string {
  return `${move.from}>${move.to}x${move.captured.join('.')}:${move.path?.join('.') ?? ''}:${move.promote ? 1 : 0}`;
}

export function validateCanonicalPosition(pos: Position): void {
  const boards = [pos.p1Men, pos.p1Kings, pos.p2Men, pos.p2Kings].map(x => x >>> 0);
  let seen = 0;
  for (const board of boards) {
    if ((seen & board) !== 0) throw new Error('overlapping bitboards are not canonical');
    seen = (seen | board) >>> 0;
  }
  if ((pos.p1Men & 0xf) !== 0) throw new Error('unpromoted P1 man on promotion rank');
  if ((pos.p2Men & 0xf0000000) !== 0) throw new Error('unpromoted P2 man on promotion rank');
  if (pos.side !== 1 && pos.side !== -1) throw new Error('invalid side');
}

function positionFrom(squares: number[], categories: number[], side: Side): Position {
  const p: Position = { side, p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0 };
  const fields = ['p1Men', 'p1Kings', 'p2Men', 'p2Kings'] as const;
  for (let i = 0; i < squares.length; i++) p[fields[categories[i]]] = (p[fields[categories[i]]] | B1(squares[i])) >>> 0;
  return p;
}

/** Enumerates every structurally reachable 2/3-piece board plus closed terminal sinks. */
export function enumerateCanonicalPositions(): Map<string, Position> {
  const result = new Map<string, Position>();
  const add = (p: Position) => result.set(canonicalStateKey(p), p);
  // Empty boards are explicit synthetic no-move terminal sinks. They are not
  // reachable from legal play, but make the zero-piece representation defined.
  add({ side: 1, p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0 });
  add({ side: -1, p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0 });
  // A capture of the last enemy produces exactly these one-piece sinks: the
  // empty side is to move. Other one-piece boards are unreachable after a move.
  for (let sq = 0; sq < 32; sq++) for (const category of [0, 1, 2, 3]) {
    if (promotionIllegal(category, sq)) continue;
    const owner: Side = category < 2 ? 1 : -1;
    add(positionFrom([sq], [category], owner === 1 ? -1 : 1));
  }
  // A multi-capture from a three-piece position can leave two friendly pieces
  // and no opponent, so retain those terminal sinks as well.
  for (let a = 0; a < 32; a++) for (let b = a + 1; b < 32; b++)
    for (const owner of [1, -1] as const) for (let ta = 0; ta < 2; ta++) for (let tb = 0; tb < 2; tb++) {
      const ca = owner === 1 ? ta : ta + 2, cb = owner === 1 ? tb : tb + 2;
      if (promotionIllegal(ca, a) || promotionIllegal(cb, b)) continue;
      add(positionFrom([a, b], [ca, cb], owner === 1 ? -1 : 1));
    }
  for (const total of [2, 3]) {
    const squares: number[] = [];
    const visitSquares = (start: number) => {
      if (squares.length === total) {
        const variants = 4 ** total;
        for (let mask = 0; mask < variants; mask++) {
          let value = mask;
          const cats: number[] = [];
          let p1 = 0, p2 = 0, legal = true;
          for (let i = 0; i < total; i++) {
            const cat = value & 3; value >>>= 2; cats.push(cat);
            if (promotionIllegal(cat, squares[i])) legal = false;
            if (cat < 2) p1++; else p2++;
          }
          if (!legal || p1 === 0 || p2 === 0) continue;
          add(positionFrom(squares, cats, 1));
          add(positionFrom(squares, cats, -1));
        }
        return;
      }
      for (let sq = start; sq <= 32 - (total - squares.length); sq++) {
        squares.push(sq); visitSquares(sq + 1); squares.pop();
      }
    };
    visitSquares(0);
  }
  return result;
}

interface Work { pos: Position; keys: string[]; moves: Move[]; predecessors: number[]; remaining: number; outcome?: CanonicalOutcome; dtm?: number }

export function buildCanonicalTablebase(): CanonicalBuild {
  const started = Date.now();
  const positions = enumerateCanonicalPositions();
  const keys = [...positions.keys()].sort();
  const index = new Map(keys.map((key, i) => [key, i]));
  const work: Work[] = keys.map(key => ({ pos: positions.get(key)!, keys: [], moves: [], predecessors: [], remaining: 0 }));
  for (let i = 0; i < work.length; i++) {
    const moves = generateMoves(work[i].pos);
    work[i].moves = moves;
    work[i].keys = moves.map(m => canonicalStateKey({ ...applyMove(work[i].pos, m), halfmoveClock: 0 }));
    work[i].remaining = moves.length;
    for (const childKey of work[i].keys) {
      const child = index.get(childKey);
      if (child === undefined) throw new Error(`canonical graph is not closed: ${keys[i]} -> ${childKey}`);
      work[child].predecessors.push(i);
    }
  }
  // FIFO propagation is valid: a WIN is first reached through the minimum
  // LOSS distance; a LOSS waits for every WIN and therefore takes their max.
  const queue: number[] = [];
  for (let i = 0; i < work.length; i++) if (work[i].remaining === 0) {
    work[i].outcome = 'LOSS'; work[i].dtm = 0; queue.push(i);
  }
  for (let head = 0; head < queue.length; head++) {
    const child = work[queue[head]];
    for (const pi of child.predecessors) {
      const parent = work[pi];
      if (parent.outcome) continue;
      if (child.outcome === 'LOSS') {
        parent.outcome = 'WIN'; parent.dtm = child.dtm! + 1; queue.push(pi);
      } else if (child.outcome === 'WIN') {
        parent.remaining--;
        parent.dtm = Math.max(parent.dtm ?? 0, child.dtm! + 1);
        if (parent.remaining === 0) { parent.outcome = 'LOSS'; queue.push(pi); }
      }
    }
  }
  const entries = new Map<string, CanonicalEntry>();
  const counts = { WIN: 0, DRAW: 0, LOSS: 0 };
  const materialCounts: Record<string, number> = {};
  let maxDtm = 0;
  for (let i = 0; i < work.length; i++) {
    const w = work[i]; const outcome = w.outcome ?? 'DRAW'; counts[outcome]++;
    const signature = materialSignature(w.pos); materialCounts[signature] = (materialCounts[signature] ?? 0) + 1;
    const candidates = w.moves.map((move, mi) => ({ move, child: work[index.get(w.keys[mi])!] }));
    const optimal = candidates.filter(({ child }) => outcome === 'WIN' ? child.outcome === 'LOSS' && child.dtm! + 1 === w.dtm :
      outcome === 'LOSS' ? child.outcome === 'WIN' && child.dtm! + 1 === w.dtm : child.outcome === undefined);
    const entry: CanonicalEntry = { key: keys[i], outcome, bestMoveKeys: optimal.map(x => canonicalMoveKey(x.move)).sort(), materialSignature: signature };
    if (outcome !== 'DRAW') { entry.dtm = w.dtm!; maxDtm = Math.max(maxDtm, w.dtm!); }
    entries.set(keys[i], entry);
  }
  const stable = keys.map(k => { const e = entries.get(k)!; return `${k}|${e.outcome}|${e.dtm ?? '-'}|${e.bestMoveKeys.join(',')}`; }).join('\n');
  return { entries, positions, fingerprint: createHash('sha256').update(stable).digest('hex'), counts, materialCounts,
    maxDtm, generationMs: Date.now() - started, estimatedCompactBytes: Buffer.byteLength(stable) };
}

export function probeHistoryAwareExact(pos: Position, historyHashes: number[] = []): HistoryAwareExactResult {
  validateCanonicalPosition(pos);
  const hash = hashPosition(pos);
  const repetitions = buildRepetitionCounts(historyHashes.length ? historyHashes : [hash]);
  if (!repetitions.has(hash)) repetitions.set(hash, 1);
  if (isDrawByInactivity(pos)) return { status: 'EXACT', outcome: 'DRAW', reason: 'inactivity', dtm: 0 };
  if (isThreefoldRepetition(repetitions, hash)) return { status: 'EXACT', outcome: 'DRAW', reason: 'repetition', dtm: 0 };
  const moves = generateMoves(pos);
  if (!moves.length) return { status: 'EXACT', outcome: 'LOSS', reason: 'terminal', dtm: 0 };
  for (const move of moves) if (generateMoves(applyMove(pos, move)).length === 0)
    return { status: 'EXACT', outcome: 'WIN', reason: 'mate-in-one', dtm: 1, best: move };
  // Canonical W/D/L cannot be promoted to live-game truth: either draw clock
  // or the complete repetition multiset may change every longer result.
  return { status: 'UNKNOWN', reason: 'history-can-intervene' };
}
