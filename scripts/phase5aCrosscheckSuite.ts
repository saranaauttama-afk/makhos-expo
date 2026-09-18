import { bitCount } from '../src/coreClaude/bitboards';
import {
  buildFreshRuleTablebase, canonicalMoveKey, canonicalStateKey, CanonicalOutcome,
} from '../src/coreClaude/search/exactEndgameTablebase';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { Position } from '../src/coreClaude/position';
import { ENDGAME_WEAKNESS_FIXTURES } from './endgameWeaknessFixtures';

function assert(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
interface Ref { outcome: CanonicalOutcome; dtm?: number; optimal: string[] }

// Build candidate first, but do not read any CanonicalEntry result until the
// independent reference truth below has been completely derived.
const candidate = buildFreshRuleTablebase();
const keys = [...candidate.positions.keys()].sort();
const index = new Map(keys.map((key, i) => [key, i]));
const rawMoves: Array<Move[] | undefined> = new Array(keys.length);
const rawChildren: Array<number[] | undefined> = new Array(keys.length);
const memo = new Map<number, Ref>();

function transitions(i: number): [Move[], number[]] {
  if (!rawMoves[i]) {
    const pos = candidate.positions.get(keys[i])!;
    rawMoves[i] = generateMoves(pos);
    rawChildren[i] = rawMoves[i]!.map(move => {
      const child = index.get(canonicalStateKey({ ...applyMove(pos, move), halfmoveClock: 0 }));
      assert(child !== undefined, `reference graph not closed: ${keys[i]}`);
      return child;
    });
  }
  return [rawMoves[i]!, rawChildren[i]!];
}

/** Separate top-down memoized algorithm; it never reads candidate entries. */
function solveReference(i: number, clock: number): Ref {
  const memoKey = clock * keys.length + i;
  const cached = memo.get(memoKey); if (cached) return cached;
  const pos = candidate.positions.get(keys[i])!;
  const limit = pos.p1Men === 0 && pos.p2Men === 0 ? 16 : 32;
  if (clock >= limit) { const result: Ref = { outcome: 'DRAW', optimal: [] }; memo.set(memoKey, result); return result; }
  const [moves, children] = transitions(i);
  if (!moves.length) { const result: Ref = { outcome: 'LOSS', dtm: 0, optimal: [] }; memo.set(memoKey, result); return result; }
  const childResults = moves.map((move, mi) => solveReference(children[mi], move.captured.length ? 0 : clock + 1));
  let result: Ref;
  const losses = childResults.map((child, mi) => ({ child, mi })).filter(x => x.child.outcome === 'LOSS');
  if (losses.length) {
    const distance = Math.min(...losses.map(x => x.child.dtm! + 1));
    result = { outcome: 'WIN', dtm: distance, optimal: losses.filter(x => x.child.dtm! + 1 === distance)
      .map(x => canonicalMoveKey(moves[x.mi])).sort() };
  } else if (childResults.some(child => child.outcome === 'DRAW')) {
    result = { outcome: 'DRAW', optimal: childResults.map((child, mi) => ({ child, mi }))
      .filter(x => x.child.outcome === 'DRAW').map(x => canonicalMoveKey(moves[x.mi])).sort() };
  } else {
    const distance = Math.max(...childResults.map(child => child.dtm! + 1));
    result = { outcome: 'LOSS', dtm: distance, optimal: childResults.map((child, mi) => ({ child, mi }))
      .filter(x => x.child.dtm! + 1 === distance).map(x => canonicalMoveKey(moves[x.mi])).sort() };
  }
  memo.set(memoKey, result); return result;
}

const requested = keys.map((key, i) => ({ key, i, pos: candidate.positions.get(key)! })).filter(({ pos }) => {
  const p1 = bitCount(pos.p1Men | pos.p1Kings), p2 = bitCount(pos.p2Men | pos.p2Kings);
  return p1 > 0 && p2 > 0 && p1 + p2 >= 2 && p1 + p2 <= 3;
});
const twoPiece = requested.filter(x => bitCount(x.pos.p1Men | x.pos.p1Kings | x.pos.p2Men | x.pos.p2Kings) === 2);
const threePiece = requested.filter(x => bitCount(x.pos.p1Men | x.pos.p1Kings | x.pos.p2Men | x.pos.p2Kings) === 3);
const selectedThree = Array.from({ length: 100_000 }, (_, n) => threePiece[Math.floor(n * threePiece.length / 100_000)]);
const selected = [...twoPiece, ...selectedThree];

// Truth is fully derived before the first candidate entry is read.
const truth = selected.map(x => ({ ...x, ref: solveReference(x.i, 0) }));
for (const { key, ref } of truth) {
  const actual = candidate.entries.get(key)!;
  assert(actual.outcome === ref.outcome, `independent W/D/L mismatch: ${key}`);
  assert(actual.dtm === ref.dtm, `independent DTM mismatch: ${key}`);
  assert(JSON.stringify(actual.bestMoveKeys) === JSON.stringify(ref.optimal),
    `independent optimal/outcome-preserving moves mismatch: ${key}`);
}

let fixtureChecks = 0;
for (const fixture of ENDGAME_WEAKNESS_FIXTURES) {
  const pieces = bitCount(fixture.pos.p1Men | fixture.pos.p1Kings | fixture.pos.p2Men | fixture.pos.p2Kings);
  if (pieces < 2 || pieces > 3) continue;
  const key = canonicalStateKey(fixture.pos), i = index.get(key);
  assert(i !== undefined, `in-scope regression fixture absent: ${fixture.id}`);
  const ref = solveReference(i, fixture.pos.halfmoveClock);
  const actual = candidate.entries.get(key)!;
  assert(fixture.pos.halfmoveClock === 0 && actual.outcome === ref.outcome && actual.dtm === ref.dtm,
    `regression fixture mismatch: ${fixture.id}`);
  fixtureChecks++;
}
assert(fixtureChecks > 0, 'no existing tiny endgame regression fixture was checked');
console.log(JSON.stringify({ exhaustiveRequestedTwoPiece: twoPiece.length,
  deterministicThreePieceSample: selectedThree.length, existingRegressionFixtures: fixtureChecks,
  independentMemoStates: memo.size }));
