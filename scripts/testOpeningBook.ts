// Test that opening book is enabled and working

import { initialPosition } from '../src/coreClaude/position';
import {
  lookupFreshOpeningBook,
  getFreshOpeningBookStats,
  resetFreshOpeningBookStats,
  ENABLE_FRESH_OPENING_BOOK,
  FRESH_OPENING_BOOK,
} from '../src/coreClaude/search/openingBookFresh';

console.log('=== Opening Book Test ===');
console.log(`ENABLE_FRESH_OPENING_BOOK: ${ENABLE_FRESH_OPENING_BOOK}`);
console.log(`Book entries: ${FRESH_OPENING_BOOK.entries.length}`);
console.log(`Generated at: ${FRESH_OPENING_BOOK.generatedAt}`);
console.log('');

if (FRESH_OPENING_BOOK.entries.length > 0) {
  const entry = FRESH_OPENING_BOOK.entries[0];
  console.log('First entry:');
  console.log(`  ply: ${entry.ply}`);
  console.log(`  side: ${entry.side}`);
  console.log(`  moves: ${entry.moves.length}`);
  entry.moves.forEach((m, i) => {
    console.log(`    ${i + 1}. ${m.from}->${m.to} (weight=${m.weight}, note="${m.note}")`);
  });
  console.log('');
}

console.log('Testing lookup from initial position...');
resetFreshOpeningBookStats();
const pos = initialPosition();
const result = lookupFreshOpeningBook(pos, { enabled: true });

if (result) {
  console.log(`✓ Found opening move: ${result.move.from}->${result.move.to}`);
} else {
  console.log('✗ No opening move found');
}

const stats = getFreshOpeningBookStats();
console.log('');
console.log('Stats:');
console.log(`  lookupAttempts: ${stats.lookupAttempts}`);
console.log(`  successfulHits: ${stats.successfulHits}`);
console.log(`  hashMisses: ${stats.hashMisses}`);
console.log(`  illegalMoves: ${stats.illegalMoves}`);
console.log(`  disabledRejects: ${stats.disabledRejects}`);

console.log('');
console.log(result ? '✓ Opening book is WORKING' : '✗ Opening book NOT working');
