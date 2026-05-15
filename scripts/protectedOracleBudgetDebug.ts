import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { B1, bitCount } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import type { Position } from '../src/coreClaude/position';
import { iterativeDeepening, moveKey } from '../src/coreClaude/search/alphabeta';
import {
  pickAdaptiveStrictBudgetMs,
  pickAdaptiveStrictDepth,
  selectStrictLevelMove,
  STRICT_LEVELS,
  type StrictDifficulty,
} from '../src/coreClaude/search/levelPolicy';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

const TIME_SCALE = 0.08;
const QUICK_MIN_BUDGET_MS = 60;
const ORACLE_DEPTH = 11;
const CATASTROPHIC_DROP = 500_000;
const OUTPUT_DIR = join('.tmp', 'protected-oracle-logs');

interface TacticalCase {
  id: 'sac-two-win-three-p1' | 'sac-two-win-three-p2';
  pos: Position;
}

interface MoveScore {
  move: Move;
  score: number;
}

interface RootCandidateLike {
  move: Move;
  score: number;
}

interface OracleRootDebugResult {
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

interface LevelDiagnostic {
  level: StrictDifficulty;
  selectedMove: string;
  searchDepth: number;
  elapsedMs: number;
  timedOut: boolean;
  overrideReason?: string;
  rawDrop: number;
  finalDrop: number;
  rawCatastrophic: boolean;
  finalCatastrophic: boolean;
}

interface RepeatDiagnostic {
  budgetMs: number;
  run: number;
  oracle: OracleRootDebugResult;
  levels: LevelDiagnostic[];
}

interface BudgetSummary {
  budgetMs: number;
  repeatCount: number;
  rawFallbackMoves: string[];
  finalOracleMoves: string[];
  rawStable: boolean;
  finalStable: boolean;
  timedOutCount: number;
  overrideChangedCount: number;
  forcedTrapCount: number;
  finalVectors: string[];
  rawVectors: string[];
  catastrophicFinalRuns: number;
  catastrophicRawRuns: number;
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

function sameMove(a: Move | undefined, b: Move | undefined): boolean {
  if (!a || !b) return false;
  return moveKey(a) === moveKey(b);
}

function inferRawFallback(candidates: RootCandidateLike[] | undefined): RootCandidateLike | undefined {
  if (!candidates?.length) return undefined;
  let best = candidates[0];
  for (const candidate of candidates) {
    if (candidate.score > best.score) best = candidate;
  }
  return best;
}

function scaledBudget(level: StrictDifficulty, pos: Position): number {
  return Math.max(QUICK_MIN_BUDGET_MS, Math.round(pickAdaptiveStrictBudgetMs(level, pos) * TIME_SCALE));
}

async function scoreMoveWithOracleBudget(
  pos: Position,
  move: Move,
  oracleMs: number,
  oracleTablebaseMs: number,
): Promise<number> {
  const child = applyMove(pos, move);
  const history = [hashPosition(pos), hashPosition(child)];
  const exactEndgame = probeSmallEndgame(child, history, oracleTablebaseMs);
  if (exactEndgame) return -exactEndgame.score;
  const result = await iterativeDeepening(
    child,
    oracleMs,
    new TT(),
    undefined,
    history,
    { cancelled: false },
    ORACLE_DEPTH,
  );
  return -result.score;
}

async function oracleRootShadow(
  pos: Position,
  oracleMs: number,
  oracleTablebaseMs: number,
): Promise<{ summary: OracleRootDebugResult; rawFallback?: MoveScore; finalOracle: MoveScore }> {
  const legal = generateMoves(pos);
  const fallback = legal[0];
  if (!fallback) throw new Error('oracleRoot called on terminal position');

  const exactEndgame = probeSmallEndgame(pos, [hashPosition(pos)], oracleTablebaseMs);
  if (exactEndgame?.best) {
    const exactMove = { move: exactEndgame.best, score: exactEndgame.score };
    return {
      summary: {
        path: 'rootProbe',
        rawFallbackMove: fmtMove(exactMove.move),
        rawFallbackScore: exactMove.score,
        finalOracleMove: fmtMove(exactMove.move),
        finalOracleScore: exactMove.score,
        overrideChangedMove: false,
        forcedRecaptureTrapParticipated: false,
        timedOut: false,
        depth: exactEndgame.dtm,
        nodes: 0,
        qnodes: 0,
      },
      rawFallback: exactMove,
      finalOracle: exactMove,
    };
  }

  const total = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const quietLowMobility = legal.length <= 3 && legal[0].captured.length === 0 && total <= 8;
  if (quietLowMobility) {
    let best: MoveScore | undefined;
    for (const move of legal) {
      const score = await scoreMoveWithOracleBudget(pos, move, oracleMs, oracleTablebaseMs);
      if (!best || score > best.score) best = { move, score };
    }
    const finalBest = best ?? { move: fallback, score: 0 };
    return {
      summary: {
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
      },
      rawFallback: finalBest,
      finalOracle: finalBest,
    };
  }

  const result = await iterativeDeepening(
    pos,
    oracleMs,
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
  const rawFallbackCandidate = inferRawFallback(result.rootCandidates);
  const rawFallback = rawFallbackCandidate
    ? { move: rawFallbackCandidate.move, score: rawFallbackCandidate.score }
    : result.best
      ? { move: result.best, score: result.score }
      : { move: fallback, score: result.score };
  const finalOracle = result.best
    ? { move: result.best, score: result.score }
    : { move: fallback, score: result.score };

  return {
    summary: {
      path: 'fallbackIterativeDeepening',
      rawFallbackMove: fmtMove(rawFallback.move),
      rawFallbackScore: rawFallback.score,
      finalOracleMove: fmtMove(finalOracle.move),
      finalOracleScore: finalOracle.score,
      overrideReason: result.overrideReason,
      overrideChangedMove: !sameMove(rawFallback.move, finalOracle.move),
      forcedRecaptureTrapParticipated: result.overrideReason === 'forced recapture trap',
      timedOut: result.timedOut,
      depth: result.depth,
      nodes: result.nodes,
      qnodes: result.qnodes,
    },
    rawFallback,
    finalOracle,
  };
}

async function runLevelDiagnostic(
  level: StrictDifficulty,
  pos: Position,
  rawFallback: MoveScore,
  finalOracle: MoveScore,
  oracleMs: number,
  oracleTablebaseMs: number,
): Promise<LevelDiagnostic> {
  const result = await iterativeDeepening(
    pos,
    scaledBudget(level, pos),
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    pickAdaptiveStrictDepth(level, pos),
  );
  const selected = selectStrictLevelMove(level, pos, result);
  const selectedMove = fmtMove(selected);
  let chosenScore = finalOracle.score;
  if (selected) {
    if (sameMove(selected, finalOracle.move)) {
      chosenScore = finalOracle.score;
    } else if (sameMove(selected, rawFallback.move)) {
      chosenScore = rawFallback.score;
    } else {
      chosenScore = await scoreMoveWithOracleBudget(pos, selected, oracleMs, oracleTablebaseMs);
    }
  }
  const rawDrop = Math.max(0, rawFallback.score - chosenScore);
  const finalDrop = Math.max(0, finalOracle.score - chosenScore);
  return {
    level,
    selectedMove,
    searchDepth: result.depth,
    elapsedMs: result.elapsedMs,
    timedOut: result.timedOut,
    overrideReason: result.overrideReason,
    rawDrop,
    finalDrop,
    rawCatastrophic: rawDrop >= CATASTROPHIC_DROP,
    finalCatastrophic: finalDrop >= CATASTROPHIC_DROP,
  };
}

function dropVector(levels: LevelDiagnostic[], key: 'rawDrop' | 'finalDrop'): string {
  return STRICT_LEVELS.map(level => {
    const row = levels.find(entry => entry.level === level);
    return row ? String(row[key]) : 'n/a';
  }).join('/');
}

function summarizeBudget(records: RepeatDiagnostic[], budgetMs: number): BudgetSummary {
  const rows = records.filter(row => row.budgetMs === budgetMs);
  const rawFallbackMoves = rows.map(row => row.oracle.rawFallbackMove ?? '(none)');
  const finalOracleMoves = rows.map(row => row.oracle.finalOracleMove);
  const rawVectors = rows.map(row => dropVector(row.levels, 'rawDrop'));
  const finalVectors = rows.map(row => dropVector(row.levels, 'finalDrop'));
  return {
    budgetMs,
    repeatCount: rows.length,
    rawFallbackMoves,
    finalOracleMoves,
    rawStable: new Set(rawFallbackMoves).size <= 1,
    finalStable: new Set(finalOracleMoves).size <= 1,
    timedOutCount: rows.filter(row => row.oracle.timedOut).length,
    overrideChangedCount: rows.filter(row => row.oracle.overrideChangedMove).length,
    forcedTrapCount: rows.filter(row => row.oracle.forcedRecaptureTrapParticipated).length,
    rawVectors,
    finalVectors,
    catastrophicFinalRuns: rows.filter(row => row.levels.some(level => level.finalCatastrophic)).length,
    catastrophicRawRuns: rows.filter(row => row.levels.some(level => level.rawCatastrophic)).length,
  };
}

async function runBudgetSweep(
  chosen: TacticalCase,
  budgetMs: number,
  repeats: number,
): Promise<RepeatDiagnostic[]> {
  const rows: RepeatDiagnostic[] = [];
  const oracleTablebaseMs = budgetMs;
  for (let run = 1; run <= repeats; run++) {
    const oracle = await oracleRootShadow(chosen.pos, budgetMs, oracleTablebaseMs);
    const levels: LevelDiagnostic[] = [];
    for (const level of STRICT_LEVELS) {
      levels.push(
        await runLevelDiagnostic(
          level,
          chosen.pos,
          oracle.rawFallback ?? oracle.finalOracle,
          oracle.finalOracle,
          budgetMs,
          oracleTablebaseMs,
        ),
      );
    }
    rows.push({
      budgetMs,
      run,
      oracle: oracle.summary,
      levels,
    });
  }
  return rows;
}

function defaultRepeatsForBudget(budgetMs: number): number {
  return budgetMs >= 10_000 ? 2 : 3;
}

async function main(): Promise<void> {
  const caseId = (process.argv[2] as TacticalCase['id'] | undefined) ?? 'sac-two-win-three-p1';
  const budgets = (process.argv[3] ?? '1500,3000,5000,10000')
    .split(',')
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value > 0);
  const chosen = CASES.find(testCase => testCase.id === caseId);
  if (!chosen) throw new Error(`unknown case: ${caseId}`);

  console.log('Protected oracle budget debug');
  console.log(`case=${chosen.id}`);
  console.log(`oracleDepth=${ORACLE_DEPTH}`);
  console.log(`budgets=${budgets.join(',')}`);

  const records: RepeatDiagnostic[] = [];
  for (const budgetMs of budgets) {
    const repeats = defaultRepeatsForBudget(budgetMs);
    console.log('');
    console.log(`budget=${budgetMs} repeats=${repeats}`);
    const rows = await runBudgetSweep(chosen, budgetMs, repeats);
    records.push(...rows);
    for (const row of rows) {
      console.log(
        `run=${row.run} raw=${row.oracle.rawFallbackMove ?? '(none)'} final=${row.oracle.finalOracleMove} ` +
        `timedOut=${row.oracle.timedOut} depth=${row.oracle.depth} nodes=${row.oracle.nodes} ` +
        `override=${row.oracle.overrideReason ?? '(none)'} changed=${row.oracle.overrideChangedMove} ` +
        `final=${dropVector(row.levels, 'finalDrop')} raw=${dropVector(row.levels, 'rawDrop')}`,
      );
    }
  }

  const summaries = budgets.map(budgetMs => summarizeBudget(records, budgetMs));
  console.log('');
  console.log('Budget summaries');
  for (const summary of summaries) {
    console.log(
      `budget=${summary.budgetMs} repeats=${summary.repeatCount} rawStable=${summary.rawStable} ` +
      `finalStable=${summary.finalStable} timedOut=${summary.timedOutCount}/${summary.repeatCount} ` +
      `overrideChanged=${summary.overrideChangedCount}/${summary.repeatCount} ` +
      `forcedTrap=${summary.forcedTrapCount}/${summary.repeatCount} ` +
      `catRaw=${summary.catastrophicRawRuns}/${summary.repeatCount} ` +
      `catFinal=${summary.catastrophicFinalRuns}/${summary.repeatCount}`,
    );
    console.log(`  rawMoves=${summary.rawFallbackMoves.join(' | ')}`);
    console.log(`  finalMoves=${summary.finalOracleMoves.join(' | ')}`);
    console.log(`  rawVectors=${summary.rawVectors.join(' | ')}`);
    console.log(`  finalVectors=${summary.finalVectors.join(' | ')}`);
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });
  const outputPath = join(OUTPUT_DIR, `${chosen.id}-budget-debug.json`);
  writeFileSync(outputPath, JSON.stringify({ caseId: chosen.id, oracleDepth: ORACLE_DEPTH, records, summaries }, null, 2));
  console.log('');
  console.log(`saved=${outputPath}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
