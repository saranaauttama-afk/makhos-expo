import { B1 } from '../src/coreClaude/bitboards';
import { isDrawByInactivity, Position } from '../src/coreClaude/position';
import { probeHistoryAwareExact, validateCanonicalPosition } from '../src/coreClaude/search/exactEndgameTablebase';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';

function referenceMateInOne(pos: Position, history: number[]): boolean {
  for (const move of generateMoves(pos)) {
    const child = applyMove(pos, move), childHash = hashPosition(child);
    const occurrences = history.filter(value => value === childHash).length + 1;
    if (isDrawByInactivity(child) || occurrences >= 3) continue;
    if (generateMoves(child).length === 0) return true;
  }
  return false;
}

function assert(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
const mixed: Position = { side: 1, p1Men: B1(20), p1Kings: 0, p2Men: 0, p2Kings: B1(9), halfmoveClock: 31 };
const at32 = { ...mixed, halfmoveClock: 32 };
assert(probeHistoryAwareExact(at32).outcome === 'DRAW', '32-ply inactivity not exact');
assert(probeHistoryAwareExact(mixed).status === 'UNKNOWN', 'one ply before 32 must not borrow board truth');
const kings: Position = { side: 1, p1Men: 0, p1Kings: B1(0), p2Men: 0, p2Kings: B1(1), halfmoveClock: 16 };
assert(probeHistoryAwareExact(kings).outcome === 'DRAW', '16-ply king inactivity not exact');
assert(probeHistoryAwareExact({ ...kings, halfmoveClock: 15 }).status === 'UNKNOWN', 'one ply before 16 must be unknown');
const fresh = { ...mixed, halfmoveClock: 0 };
const hash = hashPosition(fresh);
assert(probeHistoryAwareExact(fresh, [hash, hash]).status === 'UNKNOWN', 'second occurrence is not a draw');
assert(probeHistoryAwareExact(fresh, [hash, hash, hash]).outcome === 'DRAW', 'third occurrence must draw');
assert(probeHistoryAwareExact(fresh, [hash]).status === 'UNKNOWN', 'ordinary graph cycle must remain unknown');
let rejected = 0;
for (const bad of [
  { ...fresh, p1Kings: fresh.p2Kings },
  { ...fresh, p1Men: B1(0) },
]) try { validateCanonicalPosition(bad); } catch { rejected++; }
assert(rejected === 2, 'invalid canonical states accepted');

// Draw adjudication precedes no-legal-move adjudication in the child.
const quietMateAtLimit: Position = {
  side: 1, p1Men: 0, p1Kings: B1(4), p2Men: B1(27), p2Kings: 0, halfmoveClock: 31,
};
assert(generateMoves(quietMateAtLimit).some(m => m.captured.length === 0 &&
  generateMoves(applyMove(quietMateAtLimit, m)).length === 0), 'fixture must contain a quiet apparent mate');
assert(probeHistoryAwareExact(quietMateAtLimit).status === 'UNKNOWN',
  'apparent mate at the inactivity threshold must not be WIN');
assert(!referenceMateInOne(quietMateAtLimit, [hashPosition(quietMateAtLimit)]),
  'independent history adjudicator disagrees at inactivity threshold');

const genuineMate: Position = {
  side: 1, p1Men: 0, p1Kings: B1(18), p2Men: B1(14), p2Kings: 0, halfmoveClock: 0,
};
const matingMove = generateMoves(genuineMate)[0];
const matingChild = applyMove(genuineMate, matingMove);
assert(generateMoves(matingChild).length === 0, 'genuine mate fixture is not terminal');
assert(probeHistoryAwareExact(genuineMate).outcome === 'WIN', 'genuine mate-in-one was not proven');
assert(referenceMateInOne(genuineMate, [hashPosition(genuineMate)]), 'reference rejected genuine mate');
assert(probeHistoryAwareExact(genuineMate,
  [hashPosition(genuineMate), hashPosition(matingChild), hashPosition(matingChild)]).status === 'UNKNOWN',
  'child third repetition must take precedence over mate');
assert(!referenceMateInOne(genuineMate,
  [hashPosition(genuineMate), hashPosition(matingChild), hashPosition(matingChild)]),
  'independent history adjudicator did not prioritize third repetition');
console.log('phase5aHistorySuite: 17 assertions passed');
