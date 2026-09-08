import { strict as assert } from 'assert';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { acceptedMoves, canonicalStateKey, isVerified, LEGACY_PUZZLE_AUDIT, moveMatches,
  POSITION_CASE_SCHEMA_VERSION, POSITION_SUITE, POSITION_SUITE_VERSION, positionStructurallyLegal } from './positionSuite';
import { runPositionSuite } from './positionSuiteRunner';

async function main() {
  let checks = 0;
  assert.equal(POSITION_CASE_SCHEMA_VERSION, 'makhos-position-case-schema-v1');
  assert.equal(POSITION_SUITE_VERSION, 'makhos-position-suite-v1');
  assert.equal(new Set(POSITION_SUITE.map(c => c.id)).size, POSITION_SUITE.length, 'stable IDs must be unique');
  assert(POSITION_SUITE.every(c => c.schemaVersion === POSITION_CASE_SCHEMA_VERSION && c.suiteVersion === POSITION_SUITE_VERSION));
  checks += 4;

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
  checks += 2;

  const first = await runPositionSuite('nodes', 1_000);
  const second = await runPositionSuite('nodes', 1_000);
  const deterministic = (rows: typeof first) => rows.map(({ elapsedMs: _e, nps: _n, ...row }) => row);
  assert.deepEqual(deterministic(first), deterministic(second), 'same suite/version fixed-node result changed');
  const verifiedDenominator = first.filter(r => r.pass !== undefined).length;
  assert.equal(verifiedDenominator, POSITION_SUITE.filter(isVerified).length,
    'unverified case entered verified accuracy denominator');
  checks += 2;
  console.log(`positionSuiteTest: ${checks} checks passed; ${verifiedDenominator} verified cases`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
