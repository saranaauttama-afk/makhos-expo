import { strict as assert } from 'assert';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition } from '../src/coreClaude/position';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { acceptedMoves, canonicalStateKey, isVerified, LEGACY_PUZZLE_AUDIT, moveMatches,
  POSITION_CASE_SCHEMA_VERSION, POSITION_SUITE, POSITION_SUITE_VERSION, positionStructurallyLegal } from './positionSuite';
import { POSITION_SUITE_V1_FINGERPRINT, suiteContentFingerprint } from './positionSuite';
import { createPositionSuiteReport, fullMoveSignature, resolveWdlEvidence, runPositionSuite } from './positionSuiteRunner';

async function main() {
  let checks = 0;
  assert.equal(POSITION_CASE_SCHEMA_VERSION, 'makhos-position-case-schema-v1');
  assert.equal(POSITION_SUITE_VERSION, 'makhos-position-suite-v1');
  assert.equal(new Set(POSITION_SUITE.map(c => c.id)).size, POSITION_SUITE.length, 'stable IDs must be unique');
  assert(POSITION_SUITE.every(c => c.schemaVersion === POSITION_CASE_SCHEMA_VERSION && c.suiteVersion === POSITION_SUITE_VERSION));
  assert.equal(suiteContentFingerprint(), POSITION_SUITE_V1_FINGERPRINT,
    'frozen v1 content changed without a reviewed version/fingerprint update');
  const mutated = POSITION_SUITE.map(c => c.id === 'exact-dev-forced-king-capture-p1'
    ? { ...c, split: 'holdout' as const } : c);
  assert.notEqual(suiteContentFingerprint(mutated), POSITION_SUITE_V1_FINGERPRINT,
    'suite content mutation did not change fingerprint');
  checks += 6;

  const dev = new Set(POSITION_SUITE.filter(c => c.split === 'development').map(canonicalStateKey));
  const holdout = new Set(POSITION_SUITE.filter(c => c.split === 'holdout').map(canonicalStateKey));
  assert([...dev].every(key => !holdout.has(key)), 'development/holdout exact state/history leakage');
  checks++;

  for (const c of POSITION_SUITE) {
    assert(positionStructurallyLegal(c.position), `${c.id}: illegal structural position`);
    assert.equal(c.historyHashes.at(-1), hashPosition(c.position), `${c.id}: history does not preserve root`);
    const legal = generateMoves(c.position);
    if (isVerified(c)) for (const expected of acceptedMoves(c.expected))
      assert(legal.some(move => moveMatches(move, expected)), `${c.id}: verified accepted move is illegal`);
    if (c.expected.type === 'forced-legal-only') {
      assert.equal(legal.length, 1, `${c.id}: legal-only evidence has ${legal.length} moves`);
      assert.equal(generateMoves(applyMove(c.position, legal[0])).length, 0, `${c.id}: forced move is not immediate terminal`);
    }
  }
  checks += POSITION_SUITE.length;

  const multi = POSITION_SUITE.find(c => c.id === 'exact-dev-two-equivalent-captures')!;
  assert.equal(acceptedMoves(multi.expected).length, 2, 'multiple acceptable moves were collapsed');
  checks++;
  assert.equal(LEGACY_PUZZLE_AUDIT.length, 14, 'all 14 legacy puzzles must be audited');
  assert(LEGACY_PUZZLE_AUDIT.every(c => !isVerified(c)), 'legacy labels entered verified tier');
  assert(LEGACY_PUZZLE_AUDIT.every(c => c.audit && !c.audit.uniqueProven && !c.audit.bestMoveEvidence));
  checks += 3;

  const historyCase = POSITION_SUITE.find(c => c.id === 'exact-diagnostic-threefold-root')!;
  assert.equal(historyCase.historyHashes.length, 3);
  assert(new Set(historyCase.historyHashes).size === 1, 'repetition history was not preserved');
  assert.deepEqual(resolveWdlEvidence(historyCase, 12345), { outcome: 'draw', source: 'rule-repetition' },
    'threefold must be resolved from history/rules, not score zero');
  const ordinary = initialPosition();
  const heuristicWdl = { ...historyCase, id: 'synthetic-heuristic-wdl', position: ordinary,
    historyHashes: [hashPosition(ordinary)], expected: { type: 'wdl' as const, outcome: 'win' as const } };
  assert.equal(resolveWdlEvidence(heuristicWdl, 500), undefined,
    'positive heuristic score was incorrectly promoted to a proven win');
  assert.equal(resolveWdlEvidence({ ...heuristicWdl, expected: { type: 'wdl', outcome: 'loss' } }, -500), undefined,
    'negative heuristic score was incorrectly promoted to a proven loss');
  checks += 5;

  const sameEndpointsA: Move = { from: 3, to: 20, captured: [7, 16], path: [11, 20], promote: false };
  const sameEndpointsB: Move = { from: 3, to: 20, captured: [8, 17], path: [12, 20], promote: true };
  assert.notEqual(fullMoveSignature(sameEndpointsA), fullMoveSignature(sameEndpointsB),
    'artifact move identity collapsed distinct capture/path/promotion data');
  checks++;

  const first = await runPositionSuite('nodes', 1_000);
  const second = await runPositionSuite('nodes', 1_000);
  const deterministic = (rows: typeof first) => rows.map(({ elapsedMs: _e, nps: _n, ...row }) => row);
  assert.deepEqual(deterministic(first), deterministic(second), 'same suite/version fixed-node result changed');
  const verifiedDenominator = first.filter(r => r.pass !== undefined).length;
  assert.equal(verifiedDenominator, POSITION_SUITE.filter(isVerified).length,
    'unverified case entered verified accuracy denominator');
  const blinded = createPositionSuiteReport(first, 'nodes', 1_000);
  assert.equal(blinded.holdoutVerified.total, 2, 'blinded report lost aggregate holdout accuracy');
  assert(!blinded.rows.some(row => row.split === 'holdout'), 'default report leaked holdout case details');
  const changedHoldout = first.map(row => row.split === 'holdout' ? { ...row, pass: !row.pass } : row);
  const blindedChanged = createPositionSuiteReport(changedHoldout, 'nodes', 1_000);
  assert.deepEqual(blindedChanged.byMotif, blinded.byMotif,
    'a holdout verdict changed blinded per-motif statistics');
  assert.equal(blinded.byMotif['low mobility']?.total ?? 0, 0,
    'holdout-only low-mobility verdict leaked through blinded motif statistics');
  const revealed = createPositionSuiteReport(first, 'nodes', 1_000, true);
  assert(revealed.rows.some(row => row.split === 'holdout' && row.pass !== undefined),
    'explicit evaluation mode did not reveal holdout verdicts');
  assert.deepEqual(revealed.byMotif['low mobility'], { correct: 1, total: 1 },
    'explicit evaluation mode did not restore full holdout motif statistics');
  assert.notDeepEqual(createPositionSuiteReport(changedHoldout, 'nodes', 1_000, true).byMotif,
    revealed.byMotif, 'revealed motif statistics ignored changed holdout verdicts');
  checks += 9;
  console.log(`positionSuiteTest: ${checks} checks passed; ${verifiedDenominator} verified cases`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
