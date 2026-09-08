import { B1, toIndex, toRC } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import type { Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import {
  pickAdaptiveStrictBudgetMs,
  pickAdaptiveStrictDepth,
  selectStrictLevelMove,
  STRICT_LEVELS,
  type StrictDifficulty,
} from '../src/coreClaude/search/levelPolicy';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { getEndgameWeaknessFixture, type EndgameWeaknessFixtureId } from './endgameWeaknessFixtures';

const DEFAULT_FIXTURE_ID: EndgameWeaknessFixtureId = 'small-piece-king-vs-men';
const BENCHMARK_MODE = 'quick';
const BENCHMARK_TIME_SCALE = 0.08;
const ORACLE_MS = 1500;
const ORACLE_DEPTH = 11;
const ORACLE_TABLEBASE_MS = 1500;

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
}

function moveEquals(a: Move | undefined, b: Move | undefined): boolean {
  if (!a || !b) return false;
  if (a.from !== b.from || a.to !== b.to || a.promote !== b.promote) return false;
  if (a.captured.length !== b.captured.length) return false;
  return a.captured.every((sq, idx) => sq === b.captured[idx]);
}

function mapBits(bb: number, mapper: (sq: number) => number): number {
  let out = 0;
  for (let sq = 0; sq < 32; sq++) if (bb & B1(sq)) out |= B1(mapper(sq));
  return out >>> 0;
}

function rotateSquare180(sq: number): number {
  const { r, c } = toRC(sq);
  const rotated = toIndex(7 - r, 7 - c);
  if (rotated < 0) throw new Error(`invalid rotated square for ${sq}`);
  return rotated;
}

function mirroredEquivalent(pos: Position): Position {
  return {
    side: (pos.side === 1 ? -1 : 1) as 1 | -1,
    p1Men: mapBits(pos.p2Men, rotateSquare180),
    p1Kings: mapBits(pos.p2Kings, rotateSquare180),
    p2Men: mapBits(pos.p1Men, rotateSquare180),
    p2Kings: mapBits(pos.p1Kings, rotateSquare180),
    halfmoveClock: pos.halfmoveClock,
  };
}

async function fallbackSearch(pos: Position) {
  return iterativeDeepening(
    pos,
    ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
}

async function scoreMoveWithOracle(pos: Position, move: Move): Promise<number> {
  const child = applyMove(pos, move);
  const history = [hashPosition(pos), hashPosition(child)];
  const exactEndgame = probeSmallEndgame(child, history, ORACLE_TABLEBASE_MS);
  if (exactEndgame) return -exactEndgame.score;
  const result = await iterativeDeepening(
    child,
    ORACLE_MS,
    new TT(),
    undefined,
    history,
    { cancelled: false },
    ORACLE_DEPTH,
  );
  return -result.score;
}

function scaledBenchmarkBudget(level: StrictDifficulty, pos: Position): number {
  return Math.max(60, Math.round(pickAdaptiveStrictBudgetMs(level, pos) * BENCHMARK_TIME_SCALE));
}

async function inspectBenchmarkLevels(pos: Position, oracleMove: Move | undefined, oracleScore: number | undefined): Promise<void> {
  console.log(`benchmarkLikeChosen (matches benchmark chosenMove path, mode=${BENCHMARK_MODE})`);
  for (const level of STRICT_LEVELS) {
    const budget = scaledBenchmarkBudget(level, pos);
    const depthLimit = pickAdaptiveStrictDepth(level, pos);
    const result = await iterativeDeepening(
      pos,
      budget,
      new TT(),
      undefined,
      [hashPosition(pos)],
      { cancelled: false },
      depthLimit,
    );
    const chosen = selectStrictLevelMove(level, pos, result);
    const chosenMatchesOracleMove = moveEquals(chosen, oracleMove);
    const chosenOracleScore = chosen && oracleMove && !chosenMatchesOracleMove
      ? await scoreMoveWithOracle(pos, chosen)
      : oracleScore;
    const scoreDrop = oracleScore !== undefined && chosenOracleScore !== undefined
      ? Math.max(0, oracleScore - chosenOracleScore)
      : undefined;

    console.log(
      `  ${level} chosenMove=${fmtMove(chosen)} oracleMove=${fmtMove(oracleMove)} ` +
      `scoreDrop=${scoreDrop ?? 'n/a'} nodes=${result.nodes} depth=${result.depth} ` +
      `budgetMs=${budget} depthLimit=${depthLimit}`,
    );
  }
}

async function inspectRoot(label: string, pos: Position): Promise<void> {
  const legal = generateMoves(pos);
  const rootProbe = probeSmallEndgame(pos, [hashPosition(pos)], ORACLE_TABLEBASE_MS);
  const fallback = await fallbackSearch(pos);
  const childRows: { move: Move; childProbe: ReturnType<typeof probeSmallEndgame>; childOracleScore: number }[] = [];

  for (const move of legal) {
    const child = applyMove(pos, move);
    const childHistory = [hashPosition(pos), hashPosition(child)];
    const childProbe = probeSmallEndgame(child, childHistory, ORACLE_TABLEBASE_MS);
    const childOracleScore = await scoreMoveWithOracle(pos, move);
    childRows.push({ move, childProbe, childOracleScore });
  }

  let oracleBest = childRows[0];
  for (const row of childRows) {
    if (!oracleBest || row.childOracleScore > oracleBest.childOracleScore) oracleBest = row;
  }
  const fallbackRow = childRows.find(row => moveEquals(row.move, fallback.best));
  const fallbackMatchesOracle = !!oracleBest && !!fallbackRow && fallbackRow.childOracleScore === oracleBest.childOracleScore;
  const scoreDropVsOracle = oracleBest && fallbackRow
    ? Math.max(0, oracleBest.childOracleScore - fallbackRow.childOracleScore)
    : undefined;

  console.log(`\n=== ${label} ===`);
  console.log(`side=${pos.side} legalMoves=${legal.length}`);
  console.log(
    `rootProbe best=${fmtMove(rootProbe?.best)} score=${rootProbe?.score ?? 'undefined'} ` +
    `dtm=${rootProbe?.dtm ?? 'n/a'} exact=${rootProbe?.exact ?? 'n/a'}`,
  );
  console.log(
    `fallback best=${fmtMove(fallback.best)} score=${fallback.score} depth=${fallback.depth} ` +
    `nodes=${fallback.nodes} qnodes=${fallback.qnodes} override=${fallback.overrideReason ?? '(none)'}`,
  );
  console.log(`comparisonHint=this oracle fallback view corresponds to benchmark oracleMove`);
  console.log(
    `oracleBestMove=${fmtMove(oracleBest?.move)} oracleBestScore=${oracleBest?.childOracleScore ?? 'n/a'} ` +
    `fallbackChosenMove=${fmtMove(fallback.best)} fallbackOracleScore=${fallbackRow?.childOracleScore ?? 'n/a'} ` +
    `scoreDropVsOracle=${scoreDropVsOracle ?? 'n/a'} ` +
    `fallbackMatchesOracle=${fallbackMatchesOracle ? 'yes' : 'no'}`,
  );
  await inspectBenchmarkLevels(pos, fallback.best, fallbackRow?.childOracleScore);

  for (const { move, childProbe, childOracleScore } of childRows) {
    console.log(
      `  move ${fmtMove(move)} | childProbe best=${fmtMove(childProbe?.best)} ` +
      `score=${childProbe?.score ?? 'undefined'} dtm=${childProbe?.dtm ?? 'n/a'} ` +
      `| oracleChildScore=${childOracleScore}`,
    );
  }
}

async function main(): Promise<void> {
  const fixtureId = (process.argv[2] as EndgameWeaknessFixtureId | undefined) ?? DEFAULT_FIXTURE_ID;
  const fixture = getEndgameWeaknessFixture(fixtureId);
  const mirrored = mirroredEquivalent(fixture.pos);

  console.log(`Endgame probe debug`);
  console.log(`fixture=${fixture.id} bucket=${fixture.bucket}`);
  console.log(`note=${fixture.note}`);

  await inspectRoot('original', fixture.pos);
  await inspectRoot('mirror', mirrored);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
