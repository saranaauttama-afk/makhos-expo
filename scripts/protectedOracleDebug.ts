import { B1 } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import type { Position } from '../src/coreClaude/position';
import { iterativeDeepening, moveKey } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

const ORACLE_MS = 1500;
const ORACLE_DEPTH = 11;
const ORACLE_TABLEBASE_MS = 1500;
const DEFAULT_REPEATS = 5;

interface TacticalCase {
  id: 'sac-two-win-three-p1' | 'sac-two-win-three-p2';
  pos: Position;
}

interface MoveScore {
  move: Move;
  score: number;
}

const CASES: TacticalCase[] = [
  {
    id: 'sac-two-win-three-p1',
    pos: {
      side: 1,
      p1Men: B1(6) | B1(7) | B1(10) | B1(19) | B1(30),
      p1Kings: B1(9),
      p2Men: B1(0) | B1(13) | B1(16) | B1(17) | B1(29),
      p2Kings: B1(5),
      halfmoveClock: 0,
    },
  },
  {
    id: 'sac-two-win-three-p2',
    pos: {
      side: -1,
      p1Men: B1(7) | B1(9) | B1(16) | B1(30),
      p1Kings: B1(31),
      p2Men: B1(14) | B1(19) | B1(22) | B1(28),
      p2Kings: B1(18),
      halfmoveClock: 0,
    },
  },
];

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
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

async function oracleRootDebug(pos: Position): Promise<void> {
  const legal = generateMoves(pos);
  const fallback = legal[0];
  if (!fallback) throw new Error('oracleRoot called on terminal position');

  console.log(`legalMoves=${legal.length}`);
  for (const move of legal) {
    console.log(`  legal ${fmtMove(move)}`);
  }

  const exactEndgame = probeSmallEndgame(pos, [hashPosition(pos)], ORACLE_TABLEBASE_MS);
  if (exactEndgame?.best) {
    console.log(`oraclePath=rootProbe`);
    console.log(`oracleMove=${fmtMove(exactEndgame.best)} oracleScore=${exactEndgame.score} dtm=${exactEndgame.dtm}`);
    return;
  }

  const total = ((pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings) >>> 0)
    .toString(2)
    .split('1').length - 1;
  const quietLowMobility = legal.length <= 3 && legal[0].captured.length === 0 && total <= 8;
  console.log(`oraclePath=${quietLowMobility ? 'quietLowMobilityDirect' : 'fallbackIterativeDeepening'}`);

  if (quietLowMobility) {
    let best: MoveScore | undefined;
    for (const move of legal) {
      const score = await scoreMoveWithOracle(pos, move);
      console.log(`  childScore ${fmtMove(move)} score=${score}`);
      if (!best || score > best.score) best = { move, score };
    }
    console.log(`oracleMove=${fmtMove(best?.move)} oracleScore=${best?.score ?? 'n/a'}`);
    return;
  }

  const result = await iterativeDeepening(
    pos,
    ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
  console.log(
    `fallback best=${fmtMove(result.best)} score=${result.score} depth=${result.depth} ` +
    `nodes=${result.nodes} qnodes=${result.qnodes} timedOut=${result.timedOut} ` +
    `override=${result.overrideReason ?? '(none)'}`,
  );
  for (const candidate of result.rootCandidates ?? []) {
    console.log(
      `  cand ${fmtMove(candidate.move)} score=${candidate.score}` +
      `${result.best && moveKey(candidate.move) === moveKey(result.best) ? ' <-best' : ''}`,
    );
  }
  for (const move of legal) {
    const score = await scoreMoveWithOracle(pos, move);
    console.log(`  childScore ${fmtMove(move)} score=${score}`);
  }
}

async function main(): Promise<void> {
  const caseId = (process.argv[2] as TacticalCase['id'] | undefined) ?? 'sac-two-win-three-p1';
  const repeats = Number(process.argv[3] ?? DEFAULT_REPEATS);
  const chosen = CASES.find(testCase => testCase.id === caseId);
  if (!chosen) throw new Error(`unknown case: ${caseId}`);

  console.log('Protected oracle debug');
  console.log(`case=${chosen.id}`);
  console.log(`repeats=${repeats}`);
  console.log(`oracleMs=${ORACLE_MS}`);
  console.log(`oracleDepth=${ORACLE_DEPTH}`);
  console.log(`oracleTablebaseMs=${ORACLE_TABLEBASE_MS}`);

  for (let run = 1; run <= repeats; run++) {
    console.log('');
    console.log(`run=${run}`);
    await oracleRootDebug(chosen.pos);
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
