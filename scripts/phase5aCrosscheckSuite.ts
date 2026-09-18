import { buildCanonicalTablebase, canonicalMoveKey, canonicalStateKey } from '../src/coreClaude/search/exactEndgameTablebase';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';
import { bitCount } from '../src/coreClaude/bitboards';

function assert(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
const tb = buildCanonicalTablebase();
// Independent direct Bellman cross-check: it reads raw entries and production
// successors, never recursively probes through the tablebase API.
let twoPiece = 0, sampledThree = 0;
for (const [key, pos] of tb.positions) {
  const pieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const selected = pieces <= 2 || (pieces === 3 && sampledThree < 100_000);
  if (!selected) continue;
  if (pieces === 2) twoPiece++; else if (pieces === 3) sampledThree++;
  const entry = tb.entries.get(key)!;
  const moves = generateMoves(pos);
  const childEntries = moves.map(m => tb.entries.get(canonicalStateKey({ ...applyMove(pos, m), halfmoveClock: 0 }))!);
  const derived = !moves.length ? 'LOSS' : childEntries.some(c => c.outcome === 'LOSS') ? 'WIN' :
    childEntries.every(c => c.outcome === 'WIN') ? 'LOSS' : 'DRAW';
  assert(derived === entry.outcome, `independent outcome mismatch ${key}`);
  assert(entry.bestMoveKeys.every(k => moves.some(m => canonicalMoveKey(m) === k)), `best move mismatch ${key}`);
}
console.log(JSON.stringify({ exhaustiveTwoPiece: twoPiece, deterministicThreePieceSample: sampledThree }));
