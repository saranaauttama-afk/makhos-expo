import { initialPosition } from '../src/coreClaude/position';
import {
  DEFAULT_SEARCH_FEATURES, fixedDepthSearch, fixedNodeSearch, moveKey, scoreFromTT, scoreToTT,
} from '../src/coreClaude/search/alphabeta';
import { Bound, TT, TT_CAPACITY } from '../src/coreClaude/search/tt';

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function main() {
  let checks = 0;
  const empty = new TT();
  assert(empty.get(0, 0) === undefined, 'fresh TT fabricated a key=0/verify=0 EXACT hit');
  checks++;

  for (const bound of [Bound.EXACT, Bound.LOWER, Bound.UPPER]) {
    const tt = new TT();
    tt.put({ key: 123, verifyKey: 456, depth: 7, score: -321, move: 17, bound });
    const hit = tt.get(123, 456);
    assert(hit?.bound === bound && hit.depth === 7 && hit.score === -321 && hit.move === 17,
      `TT bound ${bound} round-trip failed`);
    checks++;
  }

  const collision = new TT();
  collision.put({ key: 19, verifyKey: 91, depth: 4, score: 10, bound: Bound.EXACT });
  collision.put({ key: 19 + TT_CAPACITY, verifyKey: 92, depth: 5, score: 20, bound: Bound.LOWER });
  assert(collision.get(19, 91) === undefined, 'replaced index survived primary-key collision');
  assert(collision.get(19 + TT_CAPACITY, 91) === undefined, 'verification-key collision was accepted');
  assert(collision.get(19 + TT_CAPACITY, 92)?.score === 20, 'colliding replacement was not retrievable');
  checks += 3;

  const replacement = new TT();
  replacement.put({ key: 7, verifyKey: 8, depth: 8, score: 80, bound: Bound.EXACT });
  replacement.put({ key: 7, verifyKey: 8, depth: 3, score: 30, bound: Bound.UPPER });
  assert(replacement.get(7, 8)?.depth === 8, 'shallower same-position entry replaced deeper entry');
  replacement.clear();
  assert(replacement.get(7, 8) === undefined, 'clear left an occupied entry');
  checks += 2;

  for (const score of [999_995, -999_995, 1234, -1234, 0]) {
    for (const storedPly of [0, 3, 19]) {
      const stored = scoreToTT(score, storedPly);
      assert(scoreFromTT(stored, storedPly) === score, `mate score round-trip failed for ${score}/${storedPly}`);
      if (Math.abs(score) > 999_900) {
        const otherPly = 7;
        const expected = score > 0 ? score + storedPly - otherPly : score - storedPly + otherPly;
        assert(scoreFromTT(stored, otherPly) === expected, `mate ply normalization failed for ${score}`);
      }
      checks++;
    }
  }

  const pos = initialPosition();
  const warmTT = new TT();
  await fixedDepthSearch(pos, 3, warmTT);
  const reused = await fixedDepthSearch(pos, 5, warmTT);
  const fresh = await fixedDepthSearch(pos, 5, new TT());
  assert(moveKey(reused.best!) === moveKey(fresh.best!) && reused.score === fresh.score,
    `iterative TT reuse changed result: reused=${moveKey(reused.best!)}/${reused.score} fresh=${moveKey(fresh.best!)}/${fresh.score}`);
  checks++;

  const stoppedTT = new TT();
  await fixedNodeSearch(pos, 300, stoppedTT);
  const afterStopped = await fixedDepthSearch(pos, 4, stoppedTT);
  const cleanAfterStopped = await fixedDepthSearch(pos, 4, new TT());
  assert(afterStopped.score === cleanAfterStopped.score,
    `partial fixed-node iteration changed root score: warm=${moveKey(afterStopped.best!)}/${afterStopped.score} clean=${moveKey(cleanAfterStopped.best!)}/${cleanAfterStopped.score}`);
  checks++;

  for (const feature of Object.keys(DEFAULT_SEARCH_FEATURES) as Array<keyof typeof DEFAULT_SEARCH_FEATURES>) {
    const result = await fixedDepthSearch(pos, 3, new TT(), [], undefined, { [feature]: false });
    assert(result.best, `${feature}=false returned no legal move`);
    checks++;
  }

  console.log(`ttCorrectnessSuite: ${checks} assertions passed`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
