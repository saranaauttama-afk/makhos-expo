import { writeFileSync } from 'fs';
import { buildCanonicalTablebase, buildFreshRuleTablebase, canonicalMoveKey, canonicalStateKey } from '../src/coreClaude/search/exactEndgameTablebase';
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
// Determinism is a gate in one invocation, not an inference from two manual runs.
const freshFirst = buildFreshRuleTablebase();
const freshSecond = buildFreshRuleTablebase();
assert(freshFirst.fingerprint === freshSecond.fingerprint,
  `fresh-rule fingerprint changed: ${freshFirst.fingerprint} != ${freshSecond.fingerprint}`);
assert(freshFirst.fingerprint !== first.fingerprint,
  'fresh-rule result unexpectedly collapsed to the clock-reset board-only result');
assert(freshFirst.entries.size === freshSecond.entries.size, 'fresh-rule state count changed');
assert(freshFirst.counts.WIN + freshFirst.counts.DRAW + freshFirst.counts.LOSS === freshFirst.entries.size,
  'fresh-rule classification incomplete');

if (process.argv.includes('--artifact')) {
  writeFileSync('.tmp/phase5a-summary.json', JSON.stringify({
    stateCount: freshFirst.entries.size, counts: freshFirst.counts, materialCounts: freshFirst.materialCounts,
    maxDtm: freshFirst.maxDtm, fingerprint: freshFirst.fingerprint, generationMs: freshFirst.generationMs,
    estimatedCompactBytes: freshFirst.estimatedCompactBytes, boardTheoreticFingerprint: first.fingerprint,
  }, null, 2) + '\n');
}
console.log(JSON.stringify({ checked, freshRuleCounts: freshFirst.counts, maxDtm: freshFirst.maxDtm,
  fingerprint: freshFirst.fingerprint, repeatFingerprint: freshSecond.fingerprint,
  generationMs: freshFirst.generationMs, estimatedCompactBytes: freshFirst.estimatedCompactBytes,
  boardTheoreticFingerprint: first.fingerprint }));
