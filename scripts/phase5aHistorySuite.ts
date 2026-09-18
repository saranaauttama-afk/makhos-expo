import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';
import { probeHistoryAwareExact, validateCanonicalPosition } from '../src/coreClaude/search/exactEndgameTablebase';
import { hashPosition } from '../src/coreClaude/search/zobrist';

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
console.log('phase5aHistorySuite: 9 assertions passed');
