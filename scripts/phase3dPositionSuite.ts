import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createPositionSuiteReport, runPositionSuite } from './positionSuiteRunner';
import { ExtensionFeatureFlags } from '../src/coreClaude/search/alphabeta';

(async () => {
  const baselineExtensions: ExtensionFeatureFlags = {
    singleLegalMove: true,
    smallEndgame: false,
    tacticalCapture: true,
    multiCapture: true,
    opponentForcedCapture: true,
    singleCaptureRecapture: true,
    rootLowMobility: true,
    soundForcedTrap: true,
  };
  const candidateExtensions: ExtensionFeatureFlags = { ...baselineExtensions, soundForcedTrap: false };
  const baselineRows = await runPositionSuite('nodes', 5_000, baselineExtensions);
  const candidateRows = await runPositionSuite('nodes', 5_000, candidateExtensions);
  const blinded = { baseline: createPositionSuiteReport(baselineRows, 'nodes', 5_000, false),
    candidate: createPositionSuiteReport(candidateRows, 'nodes', 5_000, false) };
  assert(!blinded.baseline.rows.some(r => r.split === 'holdout'));
  assert(!blinded.candidate.rows.some(r => r.split === 'holdout'));
  // Execute with --reveal-holdout only after configuration, corpus, protocol,
  // and strength decision are frozen and the blinded report has been checked.
  const reveal = process.argv.includes('--reveal-holdout');
  const revealed = reveal ? { baseline: createPositionSuiteReport(baselineRows, 'nodes', 5_000, true),
    candidate: createPositionSuiteReport(candidateRows, 'nodes', 5_000, true) } : undefined;
  if (revealed) {
    assert.deepEqual(revealed.candidate.developmentVerified, revealed.baseline.developmentVerified, 'development regression');
    assert.deepEqual(revealed.candidate.holdoutVerified, revealed.baseline.holdoutVerified, 'holdout regression');
    assert(revealed.candidate.rows.filter(r => r.motifs.some(m => m.includes('endgame'))).every(r => r.pass !== false), 'endgame regression');
  }
  mkdirSync('.tmp/phase3d/position-suite', { recursive: true });
  writeFileSync('.tmp/phase3d/position-suite/comparison.json', JSON.stringify({ blinded, revealed }, null, 2) + '\n');
  console.log(reveal ? 'PASS Phase 3D revealed regression comparison' : 'PASS Phase 3D blinded position comparison');
})().catch(error => { console.error(error); process.exitCode = 1; });
