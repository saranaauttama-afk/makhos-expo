import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';
import { probeSmallEndgameDeterministic } from '../src/coreClaude/search/endgameTablebase';
import { hashPosition } from '../src/coreClaude/search/zobrist';

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function signature(result: ReturnType<typeof probeSmallEndgameDeterministic>): string {
  const best = result.probe?.best;
  return JSON.stringify({
    score: result.probe?.score ?? null,
    best: best ? `${best.from}->${best.to}/${best.captured.join(',')}` : null,
    dtm: result.probe?.dtm ?? null,
    nodes: result.nodes,
    limitReached: result.limitReached,
    limitReason: result.limitReason ?? null,
  });
}

function repeat(name: string, pos: Position, history: number[], limit: number, runs = 5, maxDepth?: number) {
  const results = Array.from({ length: runs }, () =>
    probeSmallEndgameDeterministic(pos, history, limit, maxDepth));
  const expected = signature(results[0]);
  for (const result of results.slice(1))
    assert(signature(result) === expected, `${name} changed: ${expected} != ${signature(result)}`);
  console.log(`${name}: ${runs}/${runs} identical ${expected}`);
  return results[0];
}

function main() {
  let checks = 0;
  const forced: Position = {
    side: 1, p1Men: 0, p1Kings: B1(18), p2Men: B1(14), p2Kings: 0, halfmoveClock: 0,
  };
  const solved = repeat('forced-capture', forced, [hashPosition(forced)], 10_000);
  assert(solved.probe?.best?.captured.length === 1 && !solved.limitReached, 'forced capture was not solved');
  checks++;

  const repeated = repeat('threefold-root', forced,
    [hashPosition(forced), hashPosition(forced), hashPosition(forced)], 10_000);
  assert(repeated.probe?.score === 0 && repeated.probe.best === undefined,
    'threefold history did not override board win/loss result');
  checks++;

  const oracleRisk: Position = {
    side: 1, p1Men: 0, p1Kings: B1(21), p2Men: B1(13) | B1(6), p2Kings: 0, halfmoveClock: 0,
  };
  const limited = repeat('fixed-node-timeout-separation', oracleRisk, [hashPosition(oracleRisk)], 1);
  assert(limited.limitReached && limited.probe === undefined && limited.nodes === 1,
    'node-limited oracle did not report deterministic incomplete result');
  checks++;

  const depthLimited = repeat('fixed-depth-horizon-separation', oracleRisk,
    [hashPosition(oracleRisk)], 10_000, 5, 0);
  assert(depthLimited.limitReached && depthLimited.limitReason === 'depth' &&
    depthLimited.probe === undefined && depthLimited.nodes === 1,
  'depth-limited oracle incorrectly reported an exact result');
  checks++;

  console.log(`tablebaseDeterminismSuite: ${checks} assertions passed`);
}

main();
