import { B1 } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';
import {
  fixedDepthSearch, fixedNodeSearch, moveKey, SearchResult,
} from '../src/coreClaude/search/alphabeta';
import { buildRepetitionCounts } from '../src/coreClaude/search/repetition';
import { TT } from '../src/coreClaude/search/tt';
import {
  hashPosition, hashSearchState, verifyHashSearchState,
} from '../src/coreClaude/search/zobrist';

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function signature(result: SearchResult): string {
  return JSON.stringify({
    best: result.best ? moveKey(result.best) : null,
    score: result.score,
    nodes: result.nodes,
    qnodes: result.qnodes,
    depth: result.depth,
    pv: result.pv.map(moveKey),
    limitReached: result.limitReached ?? null,
  });
}

async function repeated(
  name: string,
  runs: number,
  search: () => Promise<SearchResult>,
): Promise<SearchResult> {
  const results: SearchResult[] = [];
  for (let i = 0; i < runs; i++) results.push(await search());
  const expected = signature(results[0]);
  for (let i = 1; i < results.length; i++) {
    assert(signature(results[i]) === expected,
      `${name} run ${i + 1} changed\nexpected ${expected}\nactual   ${signature(results[i])}`);
  }
  console.log(`${name}: ${runs}/${runs} identical ${expected}`);
  return results[0];
}

async function main() {
  let checks = 0;
  const opening = initialPosition();

  const fixedDepth = await repeated('fixed-depth', 5,
    () => fixedDepthSearch(opening, 4, new TT()));
  assert(fixedDepth.depth === 4, `fixed-depth stopped at ${fixedDepth.depth}`);
  assert(fixedDepth.pv.length > 0 && moveKey(fixedDepth.pv[0]) === moveKey(fixedDepth.best!),
    'PV must start with bestMove');
  assert(!fixedDepth.timedOut, 'fixed-depth must not report a wall-clock timeout');
  checks += 3;

  const nodeBudget = 5_000;
  const fixedNodes = await repeated('fixed-nodes', 5,
    () => fixedNodeSearch(opening, nodeBudget, new TT()));
  assert(fixedNodes.nodes + fixedNodes.qnodes === nodeBudget,
    `fixed-node used ${fixedNodes.nodes + fixedNodes.qnodes}, expected ${nodeBudget}`);
  assert(fixedNodes.limitReached === 'nodes', 'fixed-node must report its stopping reason');
  assert(!fixedNodes.timedOut, 'fixed-node must not report a wall-clock timeout');
  checks += 3;

  // Board hashes intentionally ignore draw state for repetition identity; TT
  // search-state hashes must distinguish both inactivity and prior history.
  const clock31 = { ...opening, halfmoveClock: 31 };
  const rootHistory = buildRepetitionCounts([hashPosition(opening)]);
  const repeatedHistory = buildRepetitionCounts([
    hashPosition(opening), hashPosition(opening),
  ]);
  assert(hashPosition(opening) === hashPosition(clock31), 'board hash should ignore halfmoveClock');
  assert(hashSearchState(opening, rootHistory) !== hashSearchState(clock31, rootHistory),
    'TT key must include halfmoveClock');
  assert(hashSearchState(opening, rootHistory) !== hashSearchState(opening, repeatedHistory),
    'TT key must include repetition/history context');
  assert(verifyHashSearchState(opening, rootHistory) !== verifyHashSearchState(clock31, rootHistory),
    'TT verification key must include draw state');
  checks += 4;

  // Regression: a TT warmed with the same board at a different inactivity
  // clock must give the same answer as a clean TT.
  const sharedTT = new TT();
  await fixedDepthSearch(opening, 3, sharedTT);
  const clockShared = await fixedDepthSearch(clock31, 3, sharedTT);
  const clockFresh = await fixedDepthSearch(clock31, 3, new TT());
  assert(signature(clockShared) === signature(clockFresh),
    `draw-state TT contamination: shared=${signature(clockShared)} fresh=${signature(clockFresh)}`);
  checks++;

  // Regression/audit proof: at depth zero a mandatory capture is still played
  // in qsearch; stand-pat cannot terminate the forced-capture node.
  const forcedCapture: Position = {
    side: 1,
    p1Men: B1(13) | B1(16) | B1(22) | B1(25),
    p1Kings: 0,
    p2Men: B1(12) | B1(19) | B1(20),
    p2Kings: 0,
    halfmoveClock: 0,
  };
  const forced = await fixedDepthSearch(forcedCapture, 1, new TT());
  const rootMoves = generateMoves(forcedCapture);
  assert(rootMoves[0].captured.length === 0, 'fixture root must be quiet');
  assert(rootMoves.some(move => generateMoves(applyMove(forcedCapture, move))[0]?.captured.length > 0),
    'fixture must reach an opponent mandatory capture');
  assert(forced.qnodes > rootMoves.length,
    `forced capture qsearch used stand-pat: qnodes=${forced.qnodes}, rootMoves=${rootMoves.length}`);
  checks += 3;

  console.log(`searchDeterminismSuite: ${checks} assertions passed`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
