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

export interface FreshOpeningBookRuntime {
  lookupCandidates: (
    pos: Position,
    options?: FreshOpeningBookLookupOptions,
  ) => { candidates: FreshOpeningBookCandidate[] } | undefined;
  lookup: (
    pos: Position,
    options?: FreshOpeningBookLookupOptions,
  ) => { move: Move } | undefined;
  resetStats: () => void;
  getStats: () => FreshOpeningBookStats;
}

import { generateOpeningBookEntries } from '../openingPatterns';

// TuneClaude Session 2: Enable fresh opening book with Thai patterns
export const ENABLE_FRESH_OPENING_BOOK: boolean = true;

export const FRESH_OPENING_BOOK: FreshOpeningBookFile = {
  version: 1,
  format: 'makhos-opening-book',
  generatedAt: new Date().toISOString(),
  generator: {
    name: 'tuneClaude-thai-patterns',
    settings: {
      maxPly: 10,
      topN: 5,
      thinkMs: 0,
      sidePolicy: 'both',
    },
  },
  entries: generateOpeningBookEntries(),
};

function makeEmptyFreshOpeningBookStats(): FreshOpeningBookStats {
  return {
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
}

function buildFreshOpeningBookMap(book: FreshOpeningBookFile): Map<number, FreshOpeningBookEntry[]> {
  const map = new Map<number, FreshOpeningBookEntry[]>();
  for (const entry of book.entries) {
    const bucket = map.get(entry.key);
    if (bucket) bucket.push(entry);
    else map.set(entry.key, [entry]);
  }
  return map;
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

export function createFreshOpeningBookRuntime(
  book: FreshOpeningBookFile,
  defaultEnabled = false,
): FreshOpeningBookRuntime {
  const stats = makeEmptyFreshOpeningBookStats();
  const bookMap = buildFreshOpeningBookMap(book);

  function resetStats(): void {
    stats.lookupAttempts = 0;
    stats.successfulHits = 0;
    stats.disabledRejects = 0;
    stats.benchmarkBypassRejects = 0;
    stats.hashMisses = 0;
    stats.verifyMisses = 0;
    stats.illegalMoves = 0;
    stats.selectedDeterministic = 0;
    stats.selectedRandomized = 0;
    stats.candidateCountTotal = 0;
  }

  function getStats(): FreshOpeningBookStats {
    return { ...stats };
  }

  function lookupCandidates(
    pos: Position,
    options: FreshOpeningBookLookupOptions = {},
  ): { candidates: FreshOpeningBookCandidate[] } | undefined {
    stats.lookupAttempts++;

    if (options.source === 'benchmark') {
      stats.benchmarkBypassRejects++;
      return undefined;
    }

    const enabled = options.enabled ?? defaultEnabled;
    if (!enabled) {
      stats.disabledRejects++;
      return undefined;
    }

    const entries = bookMap.get(hashPosition(pos));
    if (!entries?.length) {
      stats.hashMisses++;
      return undefined;
    }

    const verify = computeFreshOpeningBookVerify(pos);
    const verifiedEntry = entries.find(entry => entry.verify == null || entry.verify === verify);
    if (!verifiedEntry) {
      stats.verifyMisses++;
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

    stats.illegalMoves += illegalEntryCount;
    stats.candidateCountTotal += candidates.length;
    if (candidates.length) stats.successfulHits++;

    return candidates.length ? { candidates } : undefined;
  }

  function lookup(
    pos: Position,
    options: FreshOpeningBookLookupOptions = {},
  ): { move: Move } | undefined {
    const hit = lookupCandidates(pos, options);
    if (!hit) return undefined;
    stats.selectedDeterministic++;
    return { move: pickDeterministicFreshOpeningBookCandidate(hit.candidates).move };
  }

  return {
    lookupCandidates,
    lookup,
    resetStats,
    getStats,
  };
}

const freshOpeningBookRuntime = createFreshOpeningBookRuntime(FRESH_OPENING_BOOK, ENABLE_FRESH_OPENING_BOOK);

export function resetFreshOpeningBookStats(): void {
  freshOpeningBookRuntime.resetStats();
}

export function getFreshOpeningBookStats(): FreshOpeningBookStats {
  return freshOpeningBookRuntime.getStats();
}

export function lookupFreshOpeningBookCandidates(
  pos: Position,
  options: FreshOpeningBookLookupOptions = {},
): { candidates: FreshOpeningBookCandidate[] } | undefined {
  return freshOpeningBookRuntime.lookupCandidates(pos, options);
}

export function lookupFreshOpeningBook(
  pos: Position,
  options: FreshOpeningBookLookupOptions = {},
): { move: Move } | undefined {
  return freshOpeningBookRuntime.lookup(pos, options);
}
