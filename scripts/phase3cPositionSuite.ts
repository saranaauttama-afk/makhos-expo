import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createPositionSuiteReport, runPositionSuite } from './positionSuiteRunner';

(async () => {
  const baselineRows = await runPositionSuite('nodes', 5_000, { smallEndgame: true });
  const candidateRows = await runPositionSuite('nodes', 5_000, { smallEndgame: false });
  const baselineBlinded = createPositionSuiteReport(baselineRows, 'nodes', 5_000, false);
  const candidateBlinded = createPositionSuiteReport(candidateRows, 'nodes', 5_000, false);
  assert(!baselineBlinded.rows.some(r => r.split === 'holdout'));
  assert(!candidateBlinded.rows.some(r => r.split === 'holdout'));
  // Selection and the independent corpus are frozen. Only now create evaluation artifacts containing holdout rows.
  const baseline = createPositionSuiteReport(baselineRows, 'nodes', 5_000, true);
  const candidate = createPositionSuiteReport(candidateRows, 'nodes', 5_000, true);
  assert.deepEqual(candidate.developmentVerified, baseline.developmentVerified, 'development regression');
  assert.deepEqual(candidate.holdoutVerified, baseline.holdoutVerified, 'holdout regression');
  const endgames = candidate.rows.filter(r => r.motifs.some(m => m.includes('endgame')));
  assert(endgames.every(r => r.pass !== false), 'verified endgame regression');
  mkdirSync('.tmp/phase3c/position-suite', { recursive: true });
  writeFileSync('.tmp/phase3c/position-suite/comparison.json', JSON.stringify({
    blinded: { baseline: baselineBlinded, candidate: candidateBlinded }, revealed: { baseline, candidate },
  }, null, 2) + '\n');
  console.log(`Phase 3C positions: development ${candidate.developmentVerified.correct}/${candidate.developmentVerified.total}; holdout ${candidate.holdoutVerified.correct}/${candidate.holdoutVerified.total}; ${endgames.length} endgame-tagged rows without regression`);
})().catch(error => { console.error(error); process.exitCode = 1; });
