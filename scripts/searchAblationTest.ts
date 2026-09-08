import { strict as assert } from 'node:assert';
import { DEFAULT_SEARCH_FEATURES } from '../src/coreClaude/search/alphabeta';
import { classifyAblation, SEARCH_ABLATION_FEATURES } from './searchAblationRunner';
import { generateSearchAblationStartSuite } from './tournamentStartSuite';

const a=generateSearchAblationStartSuite(), b=generateSearchAblationStartSuite();
assert.deepEqual(a,b); assert.equal(a.starts.length,64);
assert.equal(new Set(a.starts.map(s=>JSON.stringify(s.position))).size,64);
assert.ok(a.starts.every(s=>s.openingMoves.length>=2&&s.openingMoves.length<=17));
assert.deepEqual(SEARCH_ABLATION_FEATURES,Object.keys(DEFAULT_SEARCH_FEATURES));
assert.equal(classifyAblation([.40,.49]),'likely beneficial');
assert.equal(classifyAblation([.51,.60]),'likely harmful');
assert.equal(classifyAblation([.45,.55]),'inconclusive');
console.log('PASS search ablation infrastructure: frozen 64-start corpus, flags, and verdict boundaries');
