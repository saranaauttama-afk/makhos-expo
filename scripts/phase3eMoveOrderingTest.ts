import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { DEFAULT_MOVE_ORDERING_FEATURES } from '../src/coreClaude/search/alphabeta';
import { EXTENSION_CONFIRMATION_START_SUITE, PHASE3C_CONFIRMATION_START_SUITE, PHASE3C_FINAL_CONFIRMATION_START_SUITE, PHASE3D_CONFIRMATION_START_SUITE, PHASE3E_MOVE_ORDERING_SEED, PHASE3E_MOVE_ORDERING_START_SUITE, PHASE3E_MOVE_ORDERING_V1_FINGERPRINT, SEARCH_ABLATION_START_SUITE, searchAblationSuiteFingerprint } from './tournamentStartSuite';
const config=JSON.parse(readFileSync('config/phase3e-move-ordering-v1.json','utf8'));
assert.equal(PHASE3E_MOVE_ORDERING_SEED,128184402);assert.equal(PHASE3E_MOVE_ORDERING_START_SUITE.starts.length,64);assert.equal(PHASE3E_MOVE_ORDERING_V1_FINGERPRINT,'00db4d9cedfe76202dae15373d33bfdc89f31726ad2b9b4e0fe3438701e97091');assert.equal(searchAblationSuiteFingerprint(PHASE3E_MOVE_ORDERING_START_SUITE),config.fingerprint);
const key=(p:any)=>[p.side,p.p1Men>>>0,p.p1Kings>>>0,p.p2Men>>>0,p.p2Kings>>>0,p.halfmoveClock].join(':');const prior=new Set([...SEARCH_ABLATION_START_SUITE.starts,...EXTENSION_CONFIRMATION_START_SUITE.starts,...PHASE3C_CONFIRMATION_START_SUITE.starts,...PHASE3C_FINAL_CONFIRMATION_START_SUITE.starts,...PHASE3D_CONFIRMATION_START_SUITE.starts].map(s=>key(s.position)));assert.equal(new Set(PHASE3E_MOVE_ORDERING_START_SUITE.starts.map(s=>key(s.position))).size,64);assert(PHASE3E_MOVE_ORDERING_START_SUITE.starts.every(s=>!prior.has(key(s.position))));assert.deepEqual(config.baselineMoveOrdering,DEFAULT_MOVE_ORDERING_FEATURES);assert.deepEqual(config.inventory,Object.keys(DEFAULT_MOVE_ORDERING_FEATURES));assert.deepEqual(config.screening,{startOffset:0,pairedStarts:32,nodesPerMove:5000,maxDepth:64,maxPlies:160});assert.deepEqual(config.confirmation,{startOffset:32,pairedStarts:32,nodeBudgets:[20000,50000],timeBudgetsMs:[100],maxDepth:64,maxPlies:160});console.log('PASS Phase 3E frozen corpus, disjointness, fingerprint, controls, and partitions');

import { generateMoves } from '../src/coreClaude/movegen';
import { initialPosition } from '../src/coreClaude/position';
import { fixedNodeSearch, SearchResult } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

function searchSignature(result: SearchResult) {
  return {
    best: result.best,
    score: result.score,
    depth: result.depth,
    pv: result.pv,
    nodes: result.nodes,
    qnodes: result.qnodes,
  };
}

async function assertOrderingInstrumentationIsSemanticOnly() {
  const quiet = initialPosition();
  const capture = PHASE3E_MOVE_ORDERING_START_SUITE.starts
    .map(start => start.position)
    .find(position => generateMoves(position)[0]?.captured.length > 0);
  assert(capture, 'frozen Phase 3E suite must contain a representative capture position');

  for (const [name, position] of [['quiet', quiet], ['capture', capture]] as const) {
    const history = [hashPosition(position)];
    const defaultResult = await fixedNodeSearch(position, 5_000, new TT(), history);
    const explicitResult = await fixedNodeSearch(
      position, 5_000, new TT(), history, 64, undefined, {}, {},
      { ...DEFAULT_MOVE_ORDERING_FEATURES },
    );
    const collectedResult = await fixedNodeSearch(
      position, 5_000, new TT(), history, 64, undefined, {}, {},
      { ...DEFAULT_MOVE_ORDERING_FEATURES }, { collectMoveOrderingStats: true },
    );

    assert.equal(defaultResult.moveOrderingStats, undefined, `${name}: default search collected ordering stats`);
    assert.equal(explicitResult.moveOrderingStats, undefined, `${name}: explicit flags collected ordering stats`);
    assert(collectedResult.moveOrderingStats, `${name}: opt-in search omitted ordering stats`);
    assert.deepEqual(searchSignature(defaultResult), searchSignature(explicitResult),
      `${name}: default differs from explicit all-ordering-flags=true`);
    assert.deepEqual(searchSignature(defaultResult), searchSignature(collectedResult),
      `${name}: statistics collection changed search semantics`);
  }
}

assertOrderingInstrumentationIsSemanticOnly()
  .then(() => console.log('PASS Phase 3E ordering statistics are opt-in and semantics-preserving'))
  .catch(error => { console.error(error); process.exitCode = 1; });
