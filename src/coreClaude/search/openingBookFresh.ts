import { Move, generateMoves } from '../movegen';
import { Position } from '../position';
import { hashPosition } from './zobrist';

export type FreshOpeningBookSource = 'ui' | 'matchup' | 'benchmark' | 'other';

export interface FreshOpeningBookMove {
  from: number;
  to: number;
  weight: number;
  scoreCp?: number;
  rank?: number;
  note?: string;
}

export interface FreshOpeningBookEntry {
  key: number;
  verify?: number;
  ply: number;
  side: 1 | -1;
  moves: FreshOpeningBookMove[];
}

export interface FreshOpeningBookFile {
  version: 1;
  format: 'makhos-opening-book';
  generatedAt: string;
  generator: {
    name: string;
    settings: {
      maxPly: number;
      topN: number;
      thinkMs: number;
      sidePolicy: 'p1' | 'p2' | 'both';
    };
  };
  entries: FreshOpeningBookEntry[];
}

export interface FreshOpeningBookCandidate {
  move: Move;
  weight: number;
  scoreCp?: number;
  rank?: number;
  note?: string;
}

export interface FreshOpeningBookLookupOptions {
  source?: FreshOpeningBookSource;
  enabled?: boolean;
}

export interface FreshOpeningBookStats {
  lookupAttempts: number;
  successfulHits: number;
  disabledRejects: number;
  benchmarkBypassRejects: number;
  hashMisses: number;
  verifyMisses: number;
  illegalMoves: number;
  selectedDeterministic: number;
  selectedRandomized: number;
  candidateCountTotal: number;
}

// Phase D.4 scaffold only. This stays off until a later integration step
// explicitly enables the fresh book path.
export const ENABLE_FRESH_OPENING_BOOK: boolean = false;

export const FRESH_OPENING_BOOK: FreshOpeningBookFile = {
  version: 1,
  format: 'makhos-opening-book',
  generatedAt: 'scaffold',
  generator: {
    name: 'phase-d4-scaffold',
    settings: {
      maxPly: 0,
      topN: 0,
      thinkMs: 0,
      sidePolicy: 'both',
    },
  },
  entries: [],
};

const freshOpeningBookStats: FreshOpeningBookStats = {
  lookupAttempts: 0,
  successfulHits: 0,
  disabledRejects: 0,
  benchmarkBypassRejects: 0,
  hashMisses: 0,
  verifyMisses: 0,
  illegalMoves: 0,
  selectedDeterministic: 0,
  selectedRandomized: 0,
  candidateCountTotal: 0,
};

const FRESH_OPENING_BOOK_MAP = new Map<number, FreshOpeningBookEntry[]>();
for (const entry of FRESH_OPENING_BOOK.entries) {
  const bucket = FRESH_OPENING_BOOK_MAP.get(entry.key);
  if (bucket) bucket.push(entry);
  else FRESH_OPENING_BOOK_MAP.set(entry.key, [entry]);
}

function computeFreshOpeningBookVerify(pos: Position): number {
  let value = pos.side === 1 ? 0x13579bdf : 0x2468ace0;
  value = Math.imul((value ^ pos.p1Men) >>> 0, 0x45d9f3b);
  value = Math.imul((value ^ pos.p1Kings) >>> 0, 0x45d9f3b);
  value = Math.imul((value ^ pos.p2Men) >>> 0, 0x45d9f3b);
  value = Math.imul((value ^ pos.p2Kings) >>> 0, 0x45d9f3b);
  value ^= pos.halfmoveClock >>> 0;
  return value >>> 0;
}

function compareFreshOpeningBookCandidates(
  a: FreshOpeningBookCandidate,
  b: FreshOpeningBookCandidate,
): number {
  return (
    b.weight - a.weight ||
    (b.scoreCp ?? Number.NEGATIVE_INFINITY) - (a.scoreCp ?? Number.NEGATIVE_INFINITY) ||
    (a.rank ?? Number.POSITIVE_INFINITY) - (b.rank ?? Number.POSITIVE_INFINITY) ||
    a.move.from - b.move.from ||
    a.move.to - b.move.to
  );
}

function pickDeterministicFreshOpeningBookCandidate(
  candidates: FreshOpeningBookCandidate[],
): FreshOpeningBookCandidate {
  let best = candidates[0];
  for (let i = 1; i < candidates.length; i++) {
    if (compareFreshOpeningBookCandidates(candidates[i], best) < 0) best = candidates[i];
  }
  return best;
}

export function resetFreshOpeningBookStats(): void {
  freshOpeningBookStats.lookupAttempts = 0;
  freshOpeningBookStats.successfulHits = 0;
  freshOpeningBookStats.disabledRejects = 0;
  freshOpeningBookStats.benchmarkBypassRejects = 0;
  freshOpeningBookStats.hashMisses = 0;
  freshOpeningBookStats.verifyMisses = 0;
  freshOpeningBookStats.illegalMoves = 0;
  freshOpeningBookStats.selectedDeterministic = 0;
  freshOpeningBookStats.selectedRandomized = 0;
  freshOpeningBookStats.candidateCountTotal = 0;
}

export function getFreshOpeningBookStats(): FreshOpeningBookStats {
  return { ...freshOpeningBookStats };
}

export function lookupFreshOpeningBookCandidates(
  pos: Position,
  options: FreshOpeningBookLookupOptions = {},
): { candidates: FreshOpeningBookCandidate[] } | undefined {
  freshOpeningBookStats.lookupAttempts++;

  if (options.source === 'benchmark') {
    freshOpeningBookStats.benchmarkBypassRejects++;
    return undefined;
  }

  if (!ENABLE_FRESH_OPENING_BOOK || options.enabled === false) {
    freshOpeningBookStats.disabledRejects++;
    return undefined;
  }

  const entries = FRESH_OPENING_BOOK_MAP.get(hashPosition(pos));
  if (!entries?.length) {
    freshOpeningBookStats.hashMisses++;
    return undefined;
  }

  const verify = computeFreshOpeningBookVerify(pos);
  const verifiedEntry = entries.find(entry => entry.verify == null || entry.verify === verify);
  if (!verifiedEntry) {
    freshOpeningBookStats.verifyMisses++;
    return undefined;
  }

  const legal = generateMoves(pos);
  let illegalEntryCount = 0;
  const candidates = verifiedEntry.moves
    .map((entry): FreshOpeningBookCandidate | undefined => {
      const move = legal.find(candidate => candidate.from === entry.from && candidate.to === entry.to);
      if (!move) {
        illegalEntryCount++;
        return undefined;
      }
      return {
        move,
        weight: entry.weight,
        scoreCp: entry.scoreCp,
        rank: entry.rank,
        note: entry.note,
      };
    })
    .filter((entry): entry is FreshOpeningBookCandidate => entry != null)
    .sort(compareFreshOpeningBookCandidates);

  freshOpeningBookStats.illegalMoves += illegalEntryCount;
  freshOpeningBookStats.candidateCountTotal += candidates.length;
  if (candidates.length) freshOpeningBookStats.successfulHits++;

  return candidates.length ? { candidates } : undefined;
}

export function lookupFreshOpeningBook(
  pos: Position,
  options: FreshOpeningBookLookupOptions = {},
): { move: Move } | undefined {
  const hit = lookupFreshOpeningBookCandidates(pos, options);
  if (!hit) return undefined;
  freshOpeningBookStats.selectedDeterministic++;
  return { move: pickDeterministicFreshOpeningBookCandidate(hit.candidates).move };
}
