// scripts/aiBenchmark.ts
//
// Shared L1-L4 benchmark for tactical quality, blunder rate, timing, and a
// small head-to-head matrix. Run with:
//   npm run bench:ai
//   npm run bench:ai:full
//   npm run bench:ai:teacher

import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'fs';
import { join } from 'path';
import { B1, bitCount } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { iterativeDeepening, moveKey, SearchResult } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import {
  pickAdaptiveStrictBudgetMs,
  pickAdaptiveStrictDepth,
  selectStrictLevelMove,
  STRICT_LEVELS,
  StrictDifficulty,
} from '../src/coreClaude/search/levelPolicy';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

type BenchmarkMode = 'quick' | 'full' | 'teacher';

interface TacticalCase {
  id: string;
  bucket: string;
  pos: Position;
}

interface MoveScore {
  move: Move;
  score: number;
}

interface TacticalSample {
  caseId: string;
  bucket: string;
  level: StrictDifficulty;
  elapsedMs: number;
  p95BasisMs: number;
  depth: number;
  nodes: number;
  qnodes: number;
  pvLength: number;
  timedOut: boolean;
  legalMoves: number;
  forcedCapture: boolean;
  maxCaptureLen: number;
  chosenMove: string;
  oracleMove: string;
  scoreDrop: number;
  solved: boolean;
  severeBlunder: boolean;
  overrideReason?: string;
}

interface LevelSummary {
  level: StrictDifficulty;
  samples: number;
  solveRate: number;
  blunderRate: number;
  avgMs: number;
  p95Ms: number;
  avgDepth: number;
  avgNodes: number;
  avgQNodes: number;
  avgPvLength: number;
  timedOut: number;
  overrides: number;
}

interface BenchmarkReport {
  mode: BenchmarkMode;
  generatedAt: string;
  config: {
    openingBookBypassed: boolean;
    timeScale: number;
    oracleMs: number;
    oracleDepth: number;
    oracleTablebaseMs: number;
    headToHeadGames: number;
    headToHeadMaxPlies: number;
  };
  tacticalCases: Array<{ id: string; bucket: string; hash: number; side: 1 | -1; legalMoves: number }>;
  tacticalSummary: LevelSummary[];
  tacticalSamples: TacticalSample[];
  releaseGatePassed: boolean;
  headToHead?: HeadToHeadCheckpoint;
}

interface HeadToHeadCheckpoint {
  score: Record<StrictDifficulty, Record<StrictDifficulty, number>>;
  completed: string[];
}

interface BenchmarkCheckpoint {
  mode: BenchmarkMode;
  tacticalSamples: TacticalSample[];
  headToHead: HeadToHeadCheckpoint;
}

const OPENING_BOOK_BYPASSED = true;
const MODE: BenchmarkMode = process.argv.includes('--teacher')
  ? 'teacher'
  : process.argv.includes('--full')
    ? 'full'
    : 'quick';
const TACTICAL_ONLY = process.argv.includes('--tactical-only');
const TIME_SCALE = MODE === 'quick' ? 0.08 : 1;
const ORACLE_MS = MODE === 'teacher' ? 20_000 : MODE === 'full' ? 5000 : 1500;
const ORACLE_DEPTH = MODE === 'teacher' ? 19 : MODE === 'full' ? 15 : 11;
const ORACLE_TABLEBASE_MS = MODE === 'teacher' ? 10_000 : MODE === 'full' ? 5000 : 1500;
const HEAD_TO_HEAD_GAMES = MODE === 'full' && !TACTICAL_ONLY ? 6 : 0;
const HEAD_TO_HEAD_MAX_PLIES = MODE === 'full' ? 240 : 80;
const BENCHMARK_DIR = join('.tmp', 'benchmarks');
const CHECKPOINT_PATH = join(BENCHMARK_DIR, `ai-benchmark-${MODE}-checkpoint.json`);
const FRESH = process.argv.includes('--fresh');

function makePosition(fields: Partial<Position> & Pick<Position, 'side'>): Position {
  return {
    p1Men: 0,
    p1Kings: 0,
    p2Men: 0,
    p2Kings: 0,
    halfmoveClock: 0,
    ...fields,
  };
}

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[idx];
}

function emptyHeadToHeadCheckpoint(): HeadToHeadCheckpoint {
  return {
    score: Object.fromEntries(
      STRICT_LEVELS.map(a => [a, Object.fromEntries(STRICT_LEVELS.map(b => [b, 0]))]),
    ) as Record<StrictDifficulty, Record<StrictDifficulty, number>>,
    completed: [],
  };
}

function emptyCheckpoint(): BenchmarkCheckpoint {
  return {
    mode: MODE,
    tacticalSamples: [],
    headToHead: emptyHeadToHeadCheckpoint(),
  };
}

function loadCheckpoint(): BenchmarkCheckpoint {
  mkdirSync(BENCHMARK_DIR, { recursive: true });
  if (FRESH && existsSync(CHECKPOINT_PATH)) unlinkSync(CHECKPOINT_PATH);
  if (!existsSync(CHECKPOINT_PATH)) return emptyCheckpoint();

  try {
    const parsed = JSON.parse(readFileSync(CHECKPOINT_PATH, 'utf8')) as BenchmarkCheckpoint;
    if (parsed.mode !== MODE) return emptyCheckpoint();
    return {
      mode: parsed.mode,
      tacticalSamples: parsed.tacticalSamples ?? [],
      headToHead: parsed.headToHead ?? emptyHeadToHeadCheckpoint(),
    };
  } catch {
    return emptyCheckpoint();
  }
}

function saveCheckpoint(checkpoint: BenchmarkCheckpoint) {
  mkdirSync(BENCHMARK_DIR, { recursive: true });
  writeFileSync(CHECKPOINT_PATH, JSON.stringify(checkpoint, null, 2));
}

function sampleKey(caseId: string, level: StrictDifficulty): string {
  return `${caseId}:${level}`;
}

function scaledBudget(level: StrictDifficulty, pos: Position): number {
  return Math.max(MODE === 'full' ? 400 : 60, Math.round(pickAdaptiveStrictBudgetMs(level, pos) * TIME_SCALE));
}

function immediateCaptureRisk(pos: Position, move: Move): number {
  const child = applyMove(pos, move);
  const replies = generateMoves(child);
  if (!replies.length || replies[0].captured.length === 0) return 0;

  let maxCap = 0;
  let hangsMovedPiece = false;
  for (const reply of replies) {
    maxCap = Math.max(maxCap, reply.captured.length);
    if (reply.captured.includes(move.to)) hangsMovedPiece = true;
  }
  return maxCap * 140 + (maxCap >= 2 ? 180 : 0) + (hangsMovedPiece ? 320 : 0);
}

async function oracleRoot(pos: Position): Promise<MoveScore> {
  const legal = generateMoves(pos);
  const fallback = legal[0];
  if (!fallback) throw new Error('oracleRoot called on terminal position');
  const exactEndgame = probeSmallEndgame(pos, [hashPosition(pos)], ORACLE_TABLEBASE_MS);
  if (exactEndgame?.best) {
    return { move: exactEndgame.best, score: exactEndgame.score };
  }

  const total = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const quietLowMobility = legal.length <= 3 && legal[0].captured.length === 0 && total <= 8;
  if (quietLowMobility) {
    let best: MoveScore | undefined;
    for (const move of legal) {
      const score = await scoreMoveWithOracle(pos, move);
      if (!best || score > best.score) best = { move, score };
    }
    if (best) return best;
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
  return { move: result.best ?? fallback, score: result.score };
}

async function scoreMoveWithOracle(pos: Position, move: Move): Promise<number> {
  const child = applyMove(pos, move);
  const exactEndgame = probeSmallEndgame(child, [hashPosition(pos), hashPosition(child)], ORACLE_TABLEBASE_MS);
  if (exactEndgame) return -exactEndgame.score;
  const result = await iterativeDeepening(
    child,
    ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos), hashPosition(child)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
  return -result.score;
}

async function runLevelOnCase(level: StrictDifficulty, testCase: TacticalCase, oracleBest: MoveScore): Promise<TacticalSample> {
  const legal = generateMoves(testCase.pos);
  const result = await iterativeDeepening(
    testCase.pos,
    scaledBudget(level, testCase.pos),
    new TT(),
    undefined,
    [hashPosition(testCase.pos)],
    { cancelled: false },
    pickAdaptiveStrictDepth(level, testCase.pos),
  );
  const selected = selectStrictLevelMove(level, testCase.pos, result);

  const chosenKey = selected ? moveKey(selected) : -1;
  const chosenScore = selected && chosenKey !== moveKey(oracleBest.move)
    ? await scoreMoveWithOracle(testCase.pos, selected)
    : oracleBest.score;
  const scoreDrop = Math.max(0, oracleBest.score - chosenScore);
  const risk = selected ? immediateCaptureRisk(testCase.pos, selected) : 10_000;
  const forcedCapture = legal.length > 0 && legal[0].captured.length > 0;
  const maxCaptureLen = legal.reduce((max, move) => Math.max(max, move.captured.length), 0);
  const validatedTrapOverride = result.overrideReason === 'forced recapture trap' && scoreDrop <= 120;

  return {
    caseId: testCase.id,
    bucket: testCase.bucket,
    level,
    elapsedMs: result.elapsedMs,
    p95BasisMs: result.elapsedMs,
    depth: result.depth,
    nodes: result.nodes,
    qnodes: result.qnodes,
    pvLength: result.pv.length,
    timedOut: result.timedOut,
    legalMoves: legal.length,
    forcedCapture,
    maxCaptureLen,
    chosenMove: fmtMove(selected),
    oracleMove: fmtMove(oracleBest.move),
    scoreDrop,
    solved: chosenKey === moveKey(oracleBest.move) || scoreDrop <= 80 || validatedTrapOverride,
    severeBlunder: validatedTrapOverride ? false : (scoreDrop >= 180 || risk >= 500),
    overrideReason: result.overrideReason,
  };
}

function tacticalCases(): TacticalCase[] {
  return [
    {
      id: 'forced-double-capture',
      bucket: 'forced multi-capture',
      pos: makePosition({ side: 1, p1Men: B1(22) | B1(30), p2Men: B1(17) | B1(9) }),
    },
    {
      id: 'max-capture-choice',
      bucket: 'max-capture choice',
      pos: makePosition({ side: 1, p1Men: B1(22) | B1(25), p2Men: B1(17) | B1(9) | B1(20) }),
    },
    {
      id: 'quiet-promotion',
      bucket: 'promotion race',
      pos: makePosition({ side: 1, p1Men: B1(5) | B1(24), p2Men: B1(31) | B1(27) }),
    },
    {
      id: 'king-fly-capture',
      bucket: 'king fly capture',
      pos: makePosition({ side: 1, p1Kings: B1(22), p2Men: B1(17), p2Kings: B1(2) }),
    },
    {
      id: 'avoid-immediate-recapture',
      bucket: 'recapture risk',
      pos: makePosition({ side: 1, p1Men: B1(21) | B1(26), p2Men: B1(17) | B1(9) | B1(14) }),
    },
    {
      id: 'small-endgame',
      bucket: 'small-piece endgame',
      pos: makePosition({ side: -1, p1Kings: B1(18), p1Men: B1(25), p2Kings: B1(10), p2Men: B1(6) }),
    },
    {
      id: 'p2-forced-double-capture',
      bucket: 'forced multi-capture',
      pos: makePosition({ side: -1, p1Men: B1(22) | B1(30), p2Men: B1(9) | B1(14) }),
    },
    {
      id: 'p2-promotion-race',
      bucket: 'promotion race',
      pos: makePosition({ side: -1, p1Men: B1(0) | B1(4), p2Men: B1(26) | B1(27) }),
    },
    {
      id: 'king-corner-trap',
      bucket: 'king trap',
      pos: makePosition({ side: 1, p1Kings: B1(29), p1Men: B1(25), p2Kings: B1(20), p2Men: B1(16) }),
    },
    {
      id: 'all-kings-technical',
      bucket: 'all-kings endgame',
      pos: makePosition({ side: 1, p1Kings: B1(6) | B1(22), p2Kings: B1(13) }),
    },
    {
      id: 'all-kings-defensive',
      bucket: 'all-kings endgame',
      pos: makePosition({ side: -1, p1Kings: B1(9), p2Kings: B1(18) | B1(26) }),
    },
    {
      id: 'all-kings-2v1-corner-win',
      bucket: 'all-kings 2v1 finisher',
      pos: makePosition({ side: 1, p1Kings: B1(6) | B1(14), p2Kings: B1(29) }),
    },
    {
      id: 'all-kings-2v1-center-win',
      bucket: 'all-kings 2v1 finisher',
      pos: makePosition({ side: -1, p1Kings: B1(13), p2Kings: B1(5) | B1(22) }),
    },
    {
      id: 'two-kings-vs-man-finisher',
      bucket: 'small-piece 2v1 finisher',
      pos: makePosition({ side: 1, p1Kings: B1(10) | B1(21), p2Men: B1(5) }),
    },
    {
      id: 'king-and-man-vs-king-finisher',
      bucket: 'small-piece 2v1 finisher',
      pos: makePosition({ side: -1, p1Kings: B1(18), p2Kings: B1(6), p2Men: B1(24) }),
    },
    {
      id: 'quiet-hanging-piece-p1',
      bucket: 'quiet hanging piece',
      pos: makePosition({ side: 1, p1Men: B1(21) | B1(25) | B1(30), p2Men: B1(13) | B1(14) | B1(17) }),
    },
    {
      id: 'quiet-hanging-piece-p2',
      bucket: 'quiet hanging piece',
      pos: makePosition({ side: -1, p1Men: B1(14) | B1(18) | B1(21), p2Men: B1(1) | B1(6) | B1(10) }),
    },
    {
      id: 'capture-recapture-choice-p1',
      bucket: 'recapture risk',
      pos: makePosition({ side: 1, p1Men: B1(20) | B1(24) | B1(29), p2Men: B1(16) | B1(12) | B1(9) }),
    },
    {
      id: 'capture-recapture-choice-p2',
      bucket: 'recapture risk',
      pos: makePosition({ side: -1, p1Men: B1(19) | B1(22) | B1(26), p2Men: B1(10) | B1(14) | B1(5) }),
    },
    {
      id: 'max-capture-tie-p1',
      bucket: 'max-capture tie',
      pos: makePosition({ side: 1, p1Men: B1(23) | B1(26), p2Men: B1(18) | B1(10) | B1(21) | B1(14) }),
    },
    {
      id: 'max-capture-tie-king',
      bucket: 'max-capture tie',
      pos: makePosition({ side: 1, p1Kings: B1(27), p1Men: B1(30), p2Men: B1(22) | B1(13) | B1(18) }),
    },
    {
      id: 'king-long-ray-capture',
      bucket: 'king fly capture',
      pos: makePosition({ side: -1, p1Men: B1(18), p1Kings: B1(31), p2Kings: B1(4), p2Men: B1(0) }),
    },
    {
      id: 'king-multi-capture',
      bucket: 'king fly capture',
      pos: makePosition({ side: 1, p1Kings: B1(30), p2Men: B1(25) | B1(16), p2Kings: B1(7) }),
    },
    {
      id: 'blocked-promotion-lane',
      bucket: 'promotion race',
      pos: makePosition({ side: 1, p1Men: B1(8) | B1(9) | B1(13), p2Men: B1(4) | B1(5) | B1(31) }),
    },
    {
      id: 'p2-blocked-promotion-lane',
      bucket: 'promotion race',
      pos: makePosition({ side: -1, p1Men: B1(0) | B1(26) | B1(27), p2Men: B1(22) | B1(23) | B1(18) }),
    },
    {
      id: 'small-piece-king-vs-men',
      bucket: 'small-piece endgame',
      pos: makePosition({ side: 1, p1Kings: B1(21), p2Men: B1(13) | B1(6) }),
    },
    {
      id: 'small-piece-men-race',
      bucket: 'small-piece endgame',
      pos: makePosition({ side: -1, p1Men: B1(8) | B1(12), p2Men: B1(23) | B1(27) }),
    },
    {
      id: 'p1-bait-recapture-win',
      bucket: 'forced recapture trap',
      pos: makePosition({ side: 1, p1Men: B1(21) | B1(26), p2Men: B1(13) }),
    },
    {
      id: 'p2-bait-recapture-win',
      bucket: 'forced recapture trap',
      pos: makePosition({ side: -1, p1Men: B1(18), p2Men: B1(5) | B1(10) }),
    },
    {
      id: 'midgame-bait-double-recapture-p1',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: 1,
        p1Men: B1(21) | B1(26) | B1(27) | B1(30),
        p2Men: B1(13) | B1(5) | B1(6) | B1(9),
      }),
    },
    {
      id: 'midgame-bait-double-recapture-p2',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: -1,
        p1Men: B1(18) | B1(22) | B1(25) | B1(26),
        p2Men: B1(5) | B1(10) | B1(1) | B1(2),
      }),
    },
    {
      id: 'opening-bait-double-recapture-p1',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: 1,
        p1Men: B1(21) | B1(22) | B1(26) | B1(27) | B1(29) | B1(30),
        p2Men: B1(5) | B1(6) | B1(9) | B1(10) | B1(13) | B1(14),
      }),
    },
    {
      id: 'opening-bait-double-recapture-p2',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: -1,
        p1Men: B1(17) | B1(18) | B1(21) | B1(22) | B1(25) | B1(26),
        p2Men: B1(1) | B1(2) | B1(5) | B1(6) | B1(10) | B1(11),
      }),
    },
    {
      id: 'sac-two-win-three-p1',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: 1,
        p1Men: B1(6) | B1(7) | B1(10) | B1(19) | B1(30),
        p1Kings: B1(9),
        p2Men: B1(0) | B1(13) | B1(16) | B1(17) | B1(29),
        p2Kings: B1(5),
      }),
    },
    {
      id: 'sac-two-win-three-p2',
      bucket: 'forced recapture trap',
      pos: makePosition({
        side: -1,
        p1Men: B1(7) | B1(9) | B1(16) | B1(30),
        p1Kings: B1(31),
        p2Men: B1(14) | B1(19) | B1(22) | B1(28),
        p2Kings: B1(18),
      }),
    },
    {
      id: 'low-mobility-squeeze',
      bucket: 'low mobility',
      pos: makePosition({ side: 1, p1Men: B1(24) | B1(25) | B1(29), p2Men: B1(16) | B1(17) | B1(20) }),
    },
    {
      id: 'low-mobility-squeeze-p2',
      bucket: 'low mobility',
      pos: makePosition({ side: -1, p1Men: B1(11) | B1(14) | B1(15), p2Men: B1(4) | B1(5) | B1(8) }),
    },
    {
      id: 'opening-tactical-choice-p1',
      bucket: 'opening tactic',
      pos: makePosition({ side: 1, p1Men: B1(21) | B1(22) | B1(25) | B1(26) | B1(30), p2Men: B1(5) | B1(6) | B1(9) | B1(13) | B1(14) }),
    },
    {
      id: 'opening-tactical-choice-p2',
      bucket: 'opening tactic',
      pos: makePosition({ side: -1, p1Men: B1(17) | B1(18) | B1(21) | B1(25) | B1(26), p2Men: B1(1) | B1(5) | B1(6) | B1(9) | B1(10) }),
    },
  ];
}

async function runTacticalBenchmark(checkpoint: BenchmarkCheckpoint): Promise<TacticalSample[]> {
  const samples = [...checkpoint.tacticalSamples];
  const completed = new Set(samples.map(sample => sampleKey(sample.caseId, sample.level)));
  const cases = tacticalCases();
  const totalSamples = cases.length * STRICT_LEVELS.length;

  for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
    const testCase = cases[caseIndex];
    const pendingLevels = STRICT_LEVELS.filter(level => !completed.has(sampleKey(testCase.id, level)));
    if (!pendingLevels.length) {
      console.log(`[tactical ${caseIndex + 1}/${cases.length}] ${testCase.id} already done`);
      continue;
    }

    console.log(`[tactical ${caseIndex + 1}/${cases.length}] oracle ${testCase.id} (${testCase.bucket})`);
    const oracleBest = await oracleRoot(testCase.pos);

    for (const level of pendingLevels) {
      const doneCount = samples.length;
      console.log(`  [${doneCount + 1}/${totalSamples}] ${level} ${testCase.id}`);
      const sample = await runLevelOnCase(level, testCase, oracleBest);
      samples.push(sample);
      completed.add(sampleKey(testCase.id, level));
      checkpoint.tacticalSamples = samples;
      saveCheckpoint(checkpoint);
      console.log(
        `    chose ${sample.chosenMove}, oracle ${sample.oracleMove}, ` +
        `drop=${sample.scoreDrop}, ${sample.elapsedMs}ms${sample.overrideReason ? `, ${sample.overrideReason}` : ''}`,
      );
    }
  }

  return samples;
}

function summarizeTactical(samples: TacticalSample[]) {
  console.log(`\nTactical benchmark (${MODE}, time scale ${TIME_SCALE})`);
  console.log(`openingBookBypassed=${OPENING_BOOK_BYPASSED ? 'yes' : 'no'}`);
  console.log('level   solve   blunder   avgMs   p95Ms   avgDepth   avgNodes   avgQNodes   avgPV');
  for (const level of STRICT_LEVELS) {
    const rows = samples.filter(sample => sample.level === level);
    const solveRate = rows.filter(sample => sample.solved).length / rows.length;
    const blunderRate = rows.filter(sample => sample.severeBlunder).length / rows.length;
    const avgMs = rows.reduce((sum, row) => sum + row.elapsedMs, 0) / rows.length;
    const avgDepth = rows.reduce((sum, row) => sum + row.depth, 0) / rows.length;
    const avgNodes = rows.reduce((sum, row) => sum + row.nodes, 0) / rows.length;
    const avgQNodes = rows.reduce((sum, row) => sum + row.qnodes, 0) / rows.length;
    console.log(
      `${level.padEnd(7)} ${(solveRate * 100).toFixed(0).padStart(5)}%` +
      ` ${(blunderRate * 100).toFixed(0).padStart(8)}%` +
      ` ${avgMs.toFixed(0).padStart(7)}` +
      ` ${percentile(rows.map(row => row.p95BasisMs), 95).toFixed(0).padStart(7)}` +
      ` ${avgDepth.toFixed(1).padStart(9)}` +
      ` ${avgNodes.toFixed(0).padStart(10)}` +
      ` ${avgQNodes.toFixed(0).padStart(11)}` +
      ` ${(rows.reduce((sum, row) => sum + row.pvLength, 0) / rows.length).toFixed(1).padStart(7)}`,
    );
  }

  const misses = samples.filter(sample => !sample.solved || sample.severeBlunder);
  if (misses.length) {
    console.log('\nTactical misses / blunders');
    for (const miss of misses) {
      console.log(
        `${miss.level} ${miss.caseId}: chose ${miss.chosenMove}, oracle ${miss.oracleMove}, ` +
        `drop=${miss.scoreDrop}, elapsed=${miss.elapsedMs}ms${miss.overrideReason ? `, override=${miss.overrideReason}` : ''}`,
      );
    }
  }
}

function buildTacticalSummary(samples: TacticalSample[]): LevelSummary[] {
  return STRICT_LEVELS.map(level => {
    const rows = samples.filter(sample => sample.level === level);
    const solveRate = rows.filter(sample => sample.solved).length / rows.length;
    const blunderRate = rows.filter(sample => sample.severeBlunder).length / rows.length;
    return {
      level,
      samples: rows.length,
      solveRate,
      blunderRate,
      avgMs: rows.reduce((sum, row) => sum + row.elapsedMs, 0) / rows.length,
      p95Ms: percentile(rows.map(row => row.elapsedMs), 95),
      avgDepth: rows.reduce((sum, row) => sum + row.depth, 0) / rows.length,
      avgNodes: rows.reduce((sum, row) => sum + row.nodes, 0) / rows.length,
      avgQNodes: rows.reduce((sum, row) => sum + row.qnodes, 0) / rows.length,
      avgPvLength: rows.reduce((sum, row) => sum + row.pvLength, 0) / rows.length,
      timedOut: rows.filter(row => row.timedOut).length,
      overrides: rows.filter(row => !!row.overrideReason).length,
    };
  });
}

function writeReport(samples: TacticalSample[], gateOk: boolean, headToHead?: HeadToHeadCheckpoint): string {
  const cases = tacticalCases();
  const report: BenchmarkReport = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    config: {
      openingBookBypassed: OPENING_BOOK_BYPASSED,
      timeScale: TIME_SCALE,
      oracleMs: ORACLE_MS,
      oracleDepth: ORACLE_DEPTH,
      oracleTablebaseMs: ORACLE_TABLEBASE_MS,
      headToHeadGames: HEAD_TO_HEAD_GAMES,
      headToHeadMaxPlies: HEAD_TO_HEAD_MAX_PLIES,
    },
    tacticalCases: cases.map(testCase => ({
      id: testCase.id,
      bucket: testCase.bucket,
      hash: hashPosition(testCase.pos),
      side: testCase.pos.side,
      legalMoves: generateMoves(testCase.pos).length,
    })),
    tacticalSummary: buildTacticalSummary(samples),
    tacticalSamples: samples,
    releaseGatePassed: gateOk,
    headToHead,
  };

  const dir = BENCHMARK_DIR;
  mkdirSync(dir, { recursive: true });
  const latest = join(dir, `ai-benchmark-${MODE}-latest.json`);
  const timestamped = join(dir, `ai-benchmark-${MODE}-${report.generatedAt.replace(/[:.]/g, '-')}.json`);
  const content = JSON.stringify(report, null, 2);
  writeFileSync(latest, content);
  writeFileSync(timestamped, content);
  return latest;
}

function checkReleaseGate(samples: TacticalSample[]): boolean {
  if (MODE !== 'full') return true;

  const tacticalThresholds: Record<StrictDifficulty, { solve: number; blunder: number; p95Ms: number }> = {
    easy: { solve: 0.70, blunder: 0.08, p95Ms: 1800 },
    normal: { solve: 0.80, blunder: 0.05, p95Ms: 2800 },
    hard: { solve: 0.88, blunder: 0.03, p95Ms: 4500 },
    expert: { solve: 0.93, blunder: 0.015, p95Ms: 7000 },
  };

  let ok = true;
  console.log('\nRelease gate');
  for (const level of STRICT_LEVELS) {
    const rows = samples.filter(sample => sample.level === level);
    const solveRate = rows.filter(sample => sample.solved).length / rows.length;
    const blunderRate = rows.filter(sample => sample.severeBlunder).length / rows.length;
    const p95Ms = percentile(rows.map(row => row.elapsedMs), 95);
    const threshold = tacticalThresholds[level];
    const pass = solveRate >= threshold.solve && blunderRate <= threshold.blunder && p95Ms <= threshold.p95Ms;
    ok = ok && pass;
    console.log(
      `${pass ? 'PASS' : 'FAIL'} ${level}: ` +
      `solve=${(solveRate * 100).toFixed(0)}%/${(threshold.solve * 100).toFixed(0)}%, ` +
      `blunder=${(blunderRate * 100).toFixed(1)}%/${(threshold.blunder * 100).toFixed(1)}%, ` +
      `p95=${p95Ms.toFixed(0)}ms/${threshold.p95Ms}ms`,
    );
  }
  return ok;
}

function orderedPairGames(headToHead: HeadToHeadCheckpoint, a: StrictDifficulty, b: StrictDifficulty): number {
  return headToHead.completed.filter(key => key.startsWith(`${a}:${b}:`)).length;
}

function printLadderDiagnostics(headToHead: HeadToHeadCheckpoint): boolean {
  console.log('\nLadder diagnostics');
  let ok = true;

  for (let strongIdx = 1; strongIdx < STRICT_LEVELS.length; strongIdx++) {
    const strong = STRICT_LEVELS[strongIdx];
    for (let weakIdx = 0; weakIdx < strongIdx; weakIdx++) {
      const weak = STRICT_LEVELS[weakIdx];
      const strongGames = orderedPairGames(headToHead, strong, weak);
      const weakGames = orderedPairGames(headToHead, weak, strong);
      const totalGames = strongGames + weakGames;
      if (totalGames <= 0) continue;

      const strongPoints = headToHead.score[strong][weak] + (weakGames - headToHead.score[weak][strong]);
      const weakPoints = totalGames - strongPoints;
      const pass = strongPoints > weakPoints;
      ok = ok && pass;
      console.log(
        `${pass ? 'PASS' : 'FAIL'} ${strong} > ${weak}: ` +
        `${strongPoints.toFixed(1)}-${weakPoints.toFixed(1)} over ${totalGames} games`,
      );
    }
  }

  return ok;
}

type GameOutcome = 'p1' | 'p2' | 'draw';

async function pickLevelMove(level: StrictDifficulty, pos: Position, history: number[]): Promise<SearchResult> {
  const result = await iterativeDeepening(
    pos,
    scaledBudget(level, pos),
    new TT(),
    undefined,
    history,
    { cancelled: false },
    pickAdaptiveStrictDepth(level, pos),
  );
  return { ...result, best: selectStrictLevelMove(level, pos, result) };
}

async function playStrictGame(p1Level: StrictDifficulty, p2Level: StrictDifficulty): Promise<GameOutcome> {
  let pos = initialPosition();
  const history = [hashPosition(pos)];

  for (let ply = 0; ply < HEAD_TO_HEAD_MAX_PLIES; ply++) {
    const legal = generateMoves(pos);
    if (!legal.length) return pos.side === 1 ? 'p2' : 'p1';
    if (isDrawByInactivity(pos) || isThreefoldRepetition(buildRepetitionCounts(history), hashPosition(pos))) {
      return 'draw';
    }

    const level = pos.side === 1 ? p1Level : p2Level;
    const result = await pickLevelMove(level, pos, history);
    const move = result.best ?? legal[0];
    pos = applyMove(pos, move);
    history.push(hashPosition(pos));
  }
  return 'draw';
}

async function runHeadToHead(checkpoint: BenchmarkCheckpoint): Promise<HeadToHeadCheckpoint | undefined> {
  if (HEAD_TO_HEAD_GAMES <= 0) {
    console.log(`\nHead-to-head matrix skipped in ${MODE} mode. Run npm run bench:ai:full for release-scale games.`);
    return undefined;
  }

  console.log(`\nHead-to-head matrix (${HEAD_TO_HEAD_GAMES} games per ordered pair)`);
  const headToHead = checkpoint.headToHead ?? emptyHeadToHeadCheckpoint();
  const score = headToHead.score;
  const completed = new Set(headToHead.completed);
  const totalGames = STRICT_LEVELS.length * (STRICT_LEVELS.length - 1) * HEAD_TO_HEAD_GAMES;

  for (const a of STRICT_LEVELS) {
    for (const b of STRICT_LEVELS) {
      if (a === b) continue;
      for (let game = 0; game < HEAD_TO_HEAD_GAMES; game++) {
        const gameKey = `${a}:${b}:${game}`;
        if (completed.has(gameKey)) {
          console.log(`  [h2h ${completed.size}/${totalGames}] ${gameKey} already done`);
          continue;
        }

        const aIsP1 = game % 2 === 0;
        console.log(`  [h2h ${completed.size + 1}/${totalGames}] ${a} vs ${b} game ${game + 1}/${HEAD_TO_HEAD_GAMES} (${aIsP1 ? `${a}=P1` : `${a}=P2`})`);
        const outcome = await playStrictGame(aIsP1 ? a : b, aIsP1 ? b : a);
        const aWon = (outcome === 'p1' && aIsP1) || (outcome === 'p2' && !aIsP1);
        const bWon = (outcome === 'p1' && !aIsP1) || (outcome === 'p2' && aIsP1);
        if (aWon) score[a][b] += 1;
        else if (!bWon) score[a][b] += 0.5;

        completed.add(gameKey);
        headToHead.completed = [...completed];
        checkpoint.headToHead = headToHead;
        saveCheckpoint(checkpoint);
        console.log(`    outcome=${outcome}, score ${a} vs ${b} = ${score[a][b].toFixed(1)}`);
      }
    }
  }

  console.log(['vs'.padEnd(8), ...STRICT_LEVELS.map(level => level.padStart(8))].join(''));
  for (const a of STRICT_LEVELS) {
    const cells = STRICT_LEVELS.map(b => {
      if (a === b) return '-'.padStart(8);
      return score[a][b].toFixed(1).padStart(8);
    });
    console.log([a.padEnd(8), ...cells].join(''));
  }
  printLadderDiagnostics(headToHead);
  return headToHead;
}

async function main() {
  const checkpoint = loadCheckpoint();
  if (FRESH) console.log(`Fresh run requested; checkpoint reset at ${CHECKPOINT_PATH}`);
  else if (existsSync(CHECKPOINT_PATH)) console.log(`Resuming from checkpoint: ${CHECKPOINT_PATH}`);

  const tactical = await runTacticalBenchmark(checkpoint);
  summarizeTactical(tactical);
  const gateOk = checkReleaseGate(tactical);
  let headToHead: HeadToHeadCheckpoint | undefined;
  const reportPath = writeReport(tactical, gateOk, checkpoint.headToHead);
  console.log(`\nJSON report: ${reportPath}`);
  headToHead = await runHeadToHead(checkpoint);
  if (headToHead) {
    const finalReportPath = writeReport(tactical, gateOk, headToHead);
    console.log(`\nFinal JSON report: ${finalReportPath}`);
  }
  if (!gateOk) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
