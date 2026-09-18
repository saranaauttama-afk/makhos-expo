import { writeFileSync } from 'fs';
import { buildCanonicalTablebase, canonicalMoveKey, canonicalStateKey } from '../src/coreClaude/search/exactEndgameTablebase';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';

function assert(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }

const first = buildCanonicalTablebase();
let checked = 0;
for (const [key, entry] of first.entries) {
  const pos = first.positions.get(key)!;
  const moves = generateMoves(pos);
  const children = moves.map(move => first.entries.get(canonicalStateKey({ ...applyMove(pos, move), halfmoveClock: 0 }))!);
  assert(children.every(Boolean), `missing child for ${key}`);
  if (entry.outcome === 'WIN') {
    assert(children.some(c => c.outcome === 'LOSS'), `WIN lacks LOSS child: ${key}`);
    assert(entry.bestMoveKeys.length > 0, `WIN lacks best move: ${key}`);
    assert(children.some(c => c.outcome === 'LOSS' && c.dtm! + 1 === entry.dtm), `WIN DTM mismatch: ${key}`);
  } else if (entry.outcome === 'LOSS' && moves.length) {
    assert(children.every(c => c.outcome === 'WIN'), `LOSS has non-WIN child: ${key}`);
    assert(Math.max(...children.map(c => c.dtm! + 1)) === entry.dtm, `LOSS DTM mismatch: ${key}`);
  } else if (entry.outcome === 'DRAW' && moves.length) {
    assert(children.every(c => c.outcome !== 'LOSS'), `DRAW has LOSS child: ${key}`);
    assert(children.some(c => c.outcome === 'DRAW'), `DRAW has no DRAW continuation: ${key}`);
  }
  for (const best of entry.bestMoveKeys)
    assert(moves.some(move => canonicalMoveKey(move) === best), `illegal stored move ${best}`);
  checked++;
}
assert(checked === first.entries.size, 'not all entries checked');
assert(first.counts.WIN + first.counts.DRAW + first.counts.LOSS === first.entries.size, 'classification incomplete');

if (process.argv.includes('--artifact')) {
  writeFileSync('.tmp/phase5a-summary.json', JSON.stringify({
    stateCount: first.entries.size, counts: first.counts, materialCounts: first.materialCounts,
    maxDtm: first.maxDtm, fingerprint: first.fingerprint, generationMs: first.generationMs,
    estimatedCompactBytes: first.estimatedCompactBytes,
  }, null, 2) + '\n');
}
console.log(JSON.stringify({ checked, counts: first.counts, maxDtm: first.maxDtm, fingerprint: first.fingerprint,
  generationMs: first.generationMs, estimatedCompactBytes: first.estimatedCompactBytes }));
