import { generateMoves } from '../src/coreClaude/movegen';
import { initialPosition } from '../src/coreClaude/position';
import {
  createFreshOpeningBookRuntime,
  FRESH_OPENING_BOOK,
  getFreshOpeningBookStats,
  lookupFreshOpeningBook,
  resetFreshOpeningBookStats,
  type FreshOpeningBookFile,
} from '../src/coreClaude/search/openingBookFresh';
import { hashPosition } from '../src/coreClaude/search/zobrist';

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

function assertDefined<T>(value: T | null | undefined, message: string): T {
  if (value == null) throw new Error(message);
  return value;
}

function testEmptyScaffoldReturnsNoMove(): void {
  resetFreshOpeningBookStats();
  const pos = initialPosition();
  const result = lookupFreshOpeningBook(pos);
  assert(result == null, 'empty scaffold should return no move while disabled');

  const stats = getFreshOpeningBookStats();
  assert(stats.lookupAttempts === 1, 'empty scaffold should count one lookup');
  assert(stats.disabledRejects === 1, 'empty scaffold should reject while disabled');
  assert(stats.successfulHits === 0, 'empty scaffold should not record a hit');
}

function testIllegalMoveRejection(): void {
  const pos = initialPosition();
  const runtime = createFreshOpeningBookRuntime({
    version: 1,
    format: 'makhos-opening-book',
    generatedAt: 'validation',
    generator: {
      name: 'validation',
      settings: { maxPly: 1, topN: 1, thinkMs: 1, sidePolicy: 'both' },
    },
    entries: [{
      key: hashPosition(pos),
      ply: 0,
      side: pos.side,
      moves: [{ from: 0, to: 31, weight: 100 }],
    }],
  }, true);

  const result = runtime.lookup(pos);
  assert(result == null, 'illegal fresh-book move should be rejected');

  const stats = runtime.getStats();
  assert(stats.lookupAttempts === 1, 'illegal move test should count one lookup');
  assert(stats.illegalMoves === 1, 'illegal move should increment illegalMoves');
  assert(stats.successfulHits === 0, 'illegal move should not count as a hit');
}

function testDeterministicSelection(): void {
  const pos = initialPosition();
  const legal = generateMoves(pos);
  assert(legal.length >= 2, 'initial position should provide at least two legal moves');

  const [a, b] = [...legal].sort((left, right) => (
    left.from - right.from || left.to - right.to
  ));

  const book: FreshOpeningBookFile = {
    version: 1,
    format: 'makhos-opening-book',
    generatedAt: 'validation',
    generator: {
      name: 'validation',
      settings: { maxPly: 1, topN: 2, thinkMs: 1, sidePolicy: 'both' },
    },
    entries: [{
      key: hashPosition(pos),
      ply: 0,
      side: pos.side,
      moves: [
        { from: b.from, to: b.to, weight: 100 },
        { from: a.from, to: a.to, weight: 100 },
      ],
    }],
  };

  const runtime = createFreshOpeningBookRuntime(book, true);
  const first = assertDefined(runtime.lookup(pos), 'deterministic selection should return first move');
  const second = assertDefined(runtime.lookup(pos), 'deterministic selection should return second move');
  const third = assertDefined(runtime.lookup(pos), 'deterministic selection should return third move');

  assert(first.move.from === second.move.from && first.move.to === second.move.to, 'deterministic selection should repeat');
  assert(second.move.from === third.move.from && second.move.to === third.move.to, 'deterministic selection should stay stable');
  assert(first.move.from === a.from && first.move.to === a.to, 'deterministic tie-break should prefer lowest from/to');

  const stats = runtime.getStats();
  assert(stats.selectedDeterministic === 3, 'deterministic lookups should increment selectedDeterministic');
}

function testBenchmarkBypass(): void {
  const pos = initialPosition();
  const runtime = createFreshOpeningBookRuntime(FRESH_OPENING_BOOK, true);
  const result = runtime.lookup(pos, { source: 'benchmark' });
  assert(result == null, 'benchmark source should bypass fresh opening book');

  const stats = runtime.getStats();
  assert(stats.lookupAttempts === 1, 'benchmark bypass should count one lookup');
  assert(stats.benchmarkBypassRejects === 1, 'benchmark bypass should increment bypass rejects');
}

function testStatsResetAndRead(): void {
  resetFreshOpeningBookStats();
  lookupFreshOpeningBook(initialPosition());
  let stats = getFreshOpeningBookStats();
  assert(stats.lookupAttempts === 1, 'global scaffold should expose stats after a lookup');

  resetFreshOpeningBookStats();
  stats = getFreshOpeningBookStats();
  assert(stats.lookupAttempts === 0, 'reset should clear lookupAttempts');
  assert(stats.disabledRejects === 0, 'reset should clear disabledRejects');
  assert(stats.benchmarkBypassRejects === 0, 'reset should clear benchmarkBypassRejects');
}

function main(): void {
  testEmptyScaffoldReturnsNoMove();
  testIllegalMoveRejection();
  testDeterministicSelection();
  testBenchmarkBypass();
  testStatsResetAndRead();
  console.log('freshOpeningBookScaffoldValidation: 5 checks passed');
}

main();
