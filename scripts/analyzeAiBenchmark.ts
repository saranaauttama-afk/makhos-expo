import { existsSync, readFileSync } from 'fs';

type Level = 'easy' | 'normal' | 'hard' | 'expert';

interface TacticalSummary {
  level: Level;
  samples: number;
  solveRate: number;
  blunderRate: number;
  avgMs: number;
  p95Ms: number;
  avgDepth: number;
  timedOut: number;
  overrides: number;
}

interface TacticalSample {
  caseId: string;
  level: Level;
  chosenMove: string;
  oracleMove: string;
  scoreDrop: number;
  elapsedMs: number;
  solved: boolean;
  severeBlunder: boolean;
  overrideReason?: string;
}

const KNOWN_WARNING_CASES = new Set([
  'small-piece-king-vs-men',
]);
const WARNING_CASE_REASONS: Partial<Record<string, string>> = {
  'small-piece-king-vs-men': 'probe-suspect known case',
};

interface HeadToHead {
  score: Record<Level, Record<Level, number>>;
  completed: string[];
}

interface BenchmarkReport {
  mode: string;
  generatedAt: string;
  releaseGatePassed: boolean;
  tacticalSummary: TacticalSummary[];
  tacticalSamples: TacticalSample[];
  headToHead?: HeadToHead;
}

const LEVELS: Level[] = ['easy', 'normal', 'hard', 'expert'];
const DEFAULT_PATH = '.tmp/benchmarks/ai-benchmark-full-latest.json';

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function sampleTags(sample: TacticalSample): string[] {
  const tags: string[] = [];
  if (KNOWN_WARNING_CASES.has(sample.caseId)) tags.push('probe-unstable');
  if (sample.chosenMove !== sample.oracleMove && sample.scoreDrop === 0) {
    tags.push('oracle-tied-or-equivalent');
  }
  return tags;
}

function orderedPairGames(headToHead: HeadToHead, a: Level, b: Level): number {
  return headToHead.completed.filter(key => key.startsWith(`${a}:${b}:`)).length;
}

function printTactical(report: BenchmarkReport): void {
  console.log(`Benchmark: ${report.mode} (${report.generatedAt})`);
  console.log(`Release gate: ${report.releaseGatePassed ? 'PASS' : 'FAIL'}`);
  console.log('');
  console.log('Tactical summary');
  console.log('level   solve   blunder   avgMs   p95Ms   avgDepth   timedOut   overrides');
  for (const row of report.tacticalSummary) {
    console.log(
      `${row.level.padEnd(7)} ${pct(row.solveRate).padStart(5)}` +
      ` ${pct(row.blunderRate).padStart(8)}` +
      ` ${row.avgMs.toFixed(0).padStart(7)}` +
      ` ${row.p95Ms.toFixed(0).padStart(7)}` +
      ` ${row.avgDepth.toFixed(1).padStart(9)}` +
      ` ${String(row.timedOut).padStart(10)}` +
      ` ${String(row.overrides).padStart(11)}`,
    );
  }

  const misses = report.tacticalSamples.filter(sample => !sample.solved || sample.severeBlunder);
  const knownInstability = misses.filter(sample => KNOWN_WARNING_CASES.has(sample.caseId));
  const remainingMisses = misses.filter(sample => !KNOWN_WARNING_CASES.has(sample.caseId));

  if (knownInstability.length) {
    console.log('');
    console.log('Warning-only known instability');
    for (const miss of knownInstability) {
      const tags = sampleTags(miss);
      console.log(
        `${miss.level} ${miss.caseId}: chose ${miss.chosenMove}, oracle ${miss.oracleMove}, ` +
        `drop=${miss.scoreDrop}, elapsed=${miss.elapsedMs}ms, ` +
        `note=${WARNING_CASE_REASONS[miss.caseId] ?? 'known warning case'}` +
        `${tags.length ? `, tags=${tags.join(',')}` : ''}`,
      );
    }
  }

  if (!remainingMisses.length) return;

  console.log('');
  console.log('Tactical misses / blunders');
  for (const miss of remainingMisses) {
    const tags = sampleTags(miss);
    console.log(
      `${miss.level} ${miss.caseId}: chose ${miss.chosenMove}, oracle ${miss.oracleMove}, ` +
      `drop=${miss.scoreDrop}, elapsed=${miss.elapsedMs}ms` +
      `${miss.overrideReason ? `, ${miss.overrideReason}` : ''}` +
      `${tags.length ? `, tags=${tags.join(',')}` : ''}`,
    );
  }
}

function printHeadToHead(headToHead: HeadToHead): boolean {
  console.log('');
  console.log(`Head-to-head completed: ${headToHead.completed.length}`);
  console.log(['vs'.padEnd(8), ...LEVELS.map(level => level.padStart(8))].join(''));
  for (const a of LEVELS) {
    const cells = LEVELS.map(b => (a === b ? '-' : headToHead.score[a][b].toFixed(1)).padStart(8));
    console.log([a.padEnd(8), ...cells].join(''));
  }

  console.log('');
  console.log('Ladder diagnostics');
  let monotonic = true;
  for (let strongIdx = 1; strongIdx < LEVELS.length; strongIdx++) {
    const strong = LEVELS[strongIdx];
    for (let weakIdx = 0; weakIdx < strongIdx; weakIdx++) {
      const weak = LEVELS[weakIdx];
      const strongGames = orderedPairGames(headToHead, strong, weak);
      const weakGames = orderedPairGames(headToHead, weak, strong);
      const totalGames = strongGames + weakGames;
      if (totalGames <= 0) continue;

      const strongPoints = headToHead.score[strong][weak] + (weakGames - headToHead.score[weak][strong]);
      const weakPoints = totalGames - strongPoints;
      const pass = strongPoints > weakPoints;
      monotonic = monotonic && pass;
      console.log(
        `${pass ? 'PASS' : 'FAIL'} ${strong} > ${weak}: ` +
        `${strongPoints.toFixed(1)}-${weakPoints.toFixed(1)} over ${totalGames} games`,
      );
    }
  }

  return monotonic;
}

function main(): void {
  const path = process.argv[2] ?? DEFAULT_PATH;
  if (!existsSync(path)) {
    console.error(`Benchmark report not found: ${path}`);
    process.exitCode = 1;
    return;
  }

  const report = JSON.parse(readFileSync(path, 'utf8')) as BenchmarkReport;
  printTactical(report);
  const ladderOk = report.headToHead ? printHeadToHead(report.headToHead) : true;

  console.log('');
  console.log(`Overall: ${report.releaseGatePassed && ladderOk ? 'PASS' : 'NEEDS TUNING'}`);
}

main();
