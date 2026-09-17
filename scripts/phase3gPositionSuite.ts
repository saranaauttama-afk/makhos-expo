import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import {
  DEFAULT_EXTENSION_FEATURES,
  DEFAULT_MOVE_ORDERING_FEATURES,
  DEFAULT_SEARCH_FEATURES,
  PruningProfile,
  SearchMeasurementOptions,
} from '../src/coreClaude/search/alphabeta';
import { isVerified, POSITION_SUITE } from './positionSuite';
import { createPositionSuiteReport, PositionRunRow, runPositionSuite } from './positionSuiteRunner';

const BUDGET = 5_000;
const OUTPUT = 'benchmarks/phase3g/position-suite-comparison.json';
const baselineMeasurement: SearchMeasurementOptions = {
  lmrProfile: 'current',
  pruningProfile: 'current',
};
const candidateMeasurement: SearchMeasurementOptions = {
  lmrProfile: 'current',
  pruningProfile: 'razoring-conservative',
};

function assertOnlyPruningProfileDiffers(a: SearchMeasurementOptions, b: SearchMeasurementOptions): void {
  const withoutPruning = ({ pruningProfile: _ignored, ...rest }: SearchMeasurementOptions) => rest;
  assert.deepEqual(withoutPruning(a), withoutPruning(b));
  assert.equal(a.pruningProfile, 'current');
  assert.equal(b.pruningProfile, 'razoring-conservative');
}

function assertNoRegression(label: string, baseline: PositionRunRow[], candidate: PositionRunRow[]): void {
  const baselineById = new Map(baseline.map(row => [row.id, row]));
  for (const row of candidate) {
    const before = baselineById.get(row.id);
    assert(before, `${label}: candidate row ${row.id} has no baseline`);
    if (before.pass === true) assert.notEqual(row.pass, false, `${label}: regression at ${row.id}`);
  }
  const correct = (rows: PositionRunRow[]) => rows.filter(row => row.pass === true).length;
  assert(correct(candidate) >= correct(baseline), `${label}: verified correct count regressed`);
}

async function main(): Promise<void> {
  assertOnlyPruningProfileDiffers(baselineMeasurement, candidateMeasurement);
  const common = {
    searchFeatures: { ...DEFAULT_SEARCH_FEATURES },
    moveOrdering: { ...DEFAULT_MOVE_ORDERING_FEATURES },
  };
  const baselineRows = await runPositionSuite('nodes', BUDGET, { ...DEFAULT_EXTENSION_FEATURES }, {
    ...common,
    measurement: baselineMeasurement,
  });
  const candidateRows = await runPositionSuite('nodes', BUDGET, { ...DEFAULT_EXTENSION_FEATURES }, {
    ...common,
    measurement: candidateMeasurement,
  });

  const blinded = {
    baseline: createPositionSuiteReport(baselineRows, 'nodes', BUDGET, false),
    candidate: createPositionSuiteReport(candidateRows, 'nodes', BUDGET, false),
  };
  assert(!blinded.baseline.rows.some(row => row.split === 'holdout'));
  assert(!blinded.candidate.rows.some(row => row.split === 'holdout'));

  const developmentIds = new Set(POSITION_SUITE.filter(c => isVerified(c) && c.split === 'development').map(c => c.id));
  assertNoRegression('development',
    baselineRows.filter(row => developmentIds.has(row.id)),
    candidateRows.filter(row => developmentIds.has(row.id)));

  const revealHoldout = process.argv.includes('--reveal-holdout');
  const revealed = revealHoldout ? {
    baseline: createPositionSuiteReport(baselineRows, 'nodes', BUDGET, true),
    candidate: createPositionSuiteReport(candidateRows, 'nodes', BUDGET, true),
  } : undefined;
  if (revealed) {
    const holdoutIds = new Set(POSITION_SUITE.filter(c => isVerified(c) && c.split === 'holdout').map(c => c.id));
    assertNoRegression('holdout',
      baselineRows.filter(row => holdoutIds.has(row.id)),
      candidateRows.filter(row => holdoutIds.has(row.id)));
    const endgameIds = new Set(POSITION_SUITE.filter(c => isVerified(c) &&
      (c.split === 'development' || c.split === 'holdout') && c.motifs.some(m => m.includes('endgame'))).map(c => c.id));
    for (const rows of [baselineRows, candidateRows]) {
      assert(rows.filter(row => endgameIds.has(row.id)).every(row => row.pass === true), 'endgame-tagged verified regression');
    }
  }

  const profiles: { baseline: PruningProfile; candidate: PruningProfile } = {
    baseline: 'current', candidate: 'razoring-conservative',
  };
  const artifact = {
    schemaVersion: 'makhos-phase3g-position-comparison-v1',
    tournamentDecision: 'NEEDS MORE DATA',
    search: { mode: 'nodes', budget: BUDGET, maxDepth: 64 },
    pinned: {
      searchFeatures: common.searchFeatures,
      extensions: DEFAULT_EXTENSION_FEATURES,
      moveOrdering: common.moveOrdering,
      lmrProfile: 'current',
      pruningProfiles: profiles,
      pruningInstrumentation: false,
    },
    blinded,
    ...(revealed ? { revealed } : {}),
  };
  mkdirSync('benchmarks/phase3g', { recursive: true });
  writeFileSync(OUTPUT, JSON.stringify(artifact, null, 2) + '\n');
  console.log(revealHoldout
    ? `PASS Phase 3G revealed baseline/candidate position comparison; artifact ${OUTPUT}`
    : `PASS Phase 3G blinded baseline/candidate position comparison; artifact ${OUTPUT}`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
