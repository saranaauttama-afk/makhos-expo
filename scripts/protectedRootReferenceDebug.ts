import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { B1, bitCount } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import type { Position } from '../src/coreClaude/position';
import {
  iterativeDeepening,
  moveKey,
  type RootSearchCandidate,
  type SearchInfo,
} from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

const QUICK_ORACLE_MS = 1500;
const QUICK_ORACLE_DEPTH = 11;
const QUICK_TABLEBASE_MS = 1500;
const REFERENCE_MS = 10_000;
const REFERENCE_DEPTH = 15;
const REFERENCE_TABLEBASE_MS = 10_000;
const DEFAULT_PASSES = 2;
const OUTPUT_DIR = join('.tmp', 'protected-oracle-logs');

interface TacticalCase {
  id: 'sac-two-win-three-p1';
  pos: Position;
}

interface MoveScore {
  move: Move;
  score: number;
}

interface QuickOracleSummary {
  path: 'rootProbe' | 'quietLowMobilityDirect' | 'fallbackIterativeDeepening';
  rawFallbackMove?: string;
  rawFallbackScore?: number;
  finalOracleMove: string;
  finalOracleScore: number;
  overrideReason?: string;
  overrideChangedMove: boolean;
  forcedRecaptureTrapParticipated: boolean;
  timedOut: boolean;
  depth: number;
  nodes: number;
  qnodes: number;
}

interface ReferenceMoveResult {
  move: string;
  score: number;
  timedOut: boolean;
  depth: number;
  nodes: number;
  qnodes: number;
  elapsedMs: number;
  exactEndgame: boolean;
  pv: string[];
}

interface ReferencePass {
  pass: number;
  quickOracle: QuickOracleSummary;
  ranking: ReferenceMoveResult[];
}

const CASE: TacticalCase = {
  id: 'sac-two-win-three-p1',
  pos: {
    side: 1,
    p1Men: B1(6) | B1(7) | B1(10) | B1(19) | B1(30),
    p1Kings: B1(9),
    p2Men: B1(0) | B1(13) | B1(16) | B1(17) | B1(29),
    p2Kings: B1(5),
    halfmoveClock: 0,
  },
};

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
}

function sameMove(a: Move | undefined, b: Move | undefined): boolean {
  if (!a || !b) return false;
  return moveKey(a) === moveKey(b);
}

function inferRawFallback(candidates: RootSearchCandidate[] | undefined): RootSearchCandidate | undefined {
  if (!candidates?.length) return undefined;
  let best = candidates[0];
  for (const candidate of candidates) {
    if (candidate.score > best.score) best = candidate;
  }
  return best;
}

async function quickOracleRootShadow(pos: Position): Promise<QuickOracleSummary> {
  const legal = generateMoves(pos);
  const fallback = legal[0];
  if (!fallback) throw new Error('oracleRoot called on terminal position');

  const exactEndgame = probeSmallEndgame(pos, [hashPosition(pos)], QUICK_TABLEBASE_MS);
  if (exactEndgame?.best) {
    return {
      path: 'rootProbe',
      rawFallbackMove: fmtMove(exactEndgame.best),
      rawFallbackScore: exactEndgame.score,
      finalOracleMove: fmtMove(exactEndgame.best),
      finalOracleScore: exactEndgame.score,
      overrideChangedMove: false,
      forcedRecaptureTrapParticipated: false,
      timedOut: false,
      depth: exactEndgame.dtm,
      nodes: 0,
      qnodes: 0,
    };
  }

  const total = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const quietLowMobility = legal.length <= 3 && legal[0].captured.length === 0 && total <= 8;
  if (quietLowMobility) {
    let best: MoveScore | undefined;
    for (const move of legal) {
      const score = await scoreMoveReference(pos, move, QUICK_ORACLE_MS, QUICK_ORACLE_DEPTH, QUICK_TABLEBASE_MS);
      if (!best || score.score > best.score) best = { move, score: score.score };
    }
    const finalBest = best ?? { move: fallback, score: 0 };
    return {
      path: 'quietLowMobilityDirect',
      rawFallbackMove: fmtMove(finalBest.move),
      rawFallbackScore: finalBest.score,
      finalOracleMove: fmtMove(finalBest.move),
      finalOracleScore: finalBest.score,
      overrideChangedMove: false,
      forcedRecaptureTrapParticipated: false,
      timedOut: false,
      depth: 0,
      nodes: 0,
      qnodes: 0,
    };
  }

  const result = await iterativeDeepening(
    pos,
    QUICK_ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    QUICK_ORACLE_DEPTH,
  );
  const rawFallback = inferRawFallback(result.rootCandidates);
  const rawMove = rawFallback?.move ?? result.best ?? fallback;
  const rawScore = rawFallback?.score ?? result.score;
  const finalMove = result.best ?? fallback;
  return {
    path: 'fallbackIterativeDeepening',
    rawFallbackMove: fmtMove(rawMove),
    rawFallbackScore: rawScore,
    finalOracleMove: fmtMove(finalMove),
    finalOracleScore: result.score,
    overrideReason: result.overrideReason,
    overrideChangedMove: !sameMove(rawMove, finalMove),
    forcedRecaptureTrapParticipated: result.overrideReason === 'forced recapture trap',
    timedOut: result.timedOut,
    depth: result.depth,
    nodes: result.nodes,
    qnodes: result.qnodes,
  };
}

async function scoreMoveReference(
  pos: Position,
  move: Move,
  timeMs: number,
  maxDepth: number,
  tablebaseMs: number,
): Promise<ReferenceMoveResult> {
  const child = applyMove(pos, move);
  const history = [hashPosition(pos), hashPosition(child)];
  const exactEndgame = probeSmallEndgame(child, history, tablebaseMs);
  if (exactEndgame) {
    const pv = [fmtMove(move)];
    if (exactEndgame.best) pv.push(fmtMove(exactEndgame.best));
    return {
      move: fmtMove(move),
      score: -exactEndgame.score,
      timedOut: false,
      depth: exactEndgame.dtm,
      nodes: 0,
      qnodes: 0,
      elapsedMs: 0,
      exactEndgame: true,
      pv,
    };
  }

  let latestInfo: SearchInfo | undefined;
  const result = await iterativeDeepening(
    child,
    timeMs,
    new TT(),
    info => {
      latestInfo = info;
    },
    history,
    { cancelled: false },
    maxDepth,
  );

  const pv = [fmtMove(move), ...(latestInfo?.pv ?? []).map(fmtMove)];
  return {
    move: fmtMove(move),
    score: -result.score,
    timedOut: result.timedOut,
    depth: result.depth,
    nodes: result.nodes,
    qnodes: result.qnodes,
    elapsedMs: result.elapsedMs,
    exactEndgame: false,
    pv,
  };
}

async function runReferencePass(pass: number): Promise<ReferencePass> {
  const quickOracle = await quickOracleRootShadow(CASE.pos);
  const legal = generateMoves(CASE.pos);
  const ranking: ReferenceMoveResult[] = [];
  for (const move of legal) {
    ranking.push(
      await scoreMoveReference(
        CASE.pos,
        move,
        REFERENCE_MS,
        REFERENCE_DEPTH,
        REFERENCE_TABLEBASE_MS,
      ),
    );
  }
  ranking.sort((a, b) => b.score - a.score || a.move.localeCompare(b.move));
  return { pass, quickOracle, ranking };
}

function printPassSummary(entry: ReferencePass): void {
  console.log(`pass=${entry.pass}`);
  console.log(
    `  quick raw=${entry.quickOracle.rawFallbackMove ?? '(none)'} final=${entry.quickOracle.finalOracleMove} ` +
    `timedOut=${entry.quickOracle.timedOut} depth=${entry.quickOracle.depth} nodes=${entry.quickOracle.nodes} ` +
    `override=${entry.quickOracle.overrideReason ?? '(none)'} changed=${entry.quickOracle.overrideChangedMove}`,
  );
  console.log('  ranking');
  for (const row of entry.ranking) {
    console.log(
      `    ${row.move} score=${row.score} depth=${row.depth} nodes=${row.nodes} ` +
      `timedOut=${row.timedOut} exact=${row.exactEndgame} pv=${row.pv.join(' ')}`,
    );
  }
}

async function main(): Promise<void> {
  const passes = Number(process.argv[2] ?? DEFAULT_PASSES);
  console.log('Protected root reference debug');
  console.log(`case=${CASE.id}`);
  console.log(`passes=${passes}`);
  console.log(`quickOracleMs=${QUICK_ORACLE_MS}`);
  console.log(`referenceMs=${REFERENCE_MS}`);
  console.log(`referenceDepth=${REFERENCE_DEPTH}`);

  const results: ReferencePass[] = [];
  for (let pass = 1; pass <= passes; pass++) {
    const entry = await runReferencePass(pass);
    results.push(entry);
    printPassSummary(entry);
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });
  const outputPath = join(OUTPUT_DIR, `${CASE.id}-root-reference-debug.json`);
  writeFileSync(outputPath, JSON.stringify({
    caseId: CASE.id,
    quickOracleMs: QUICK_ORACLE_MS,
    referenceMs: REFERENCE_MS,
    referenceDepth: REFERENCE_DEPTH,
    results,
  }, null, 2));
  console.log(`saved=${outputPath}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
