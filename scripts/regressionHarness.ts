import { existsSync, readFileSync } from 'fs';

type Level = 'easy' | 'normal' | 'hard' | 'expert';
type Classification = 'PASS' | 'WARN' | 'FAIL';

interface TacticalSummary {
  level: Level;
  samples: number;
  solveRate: number;
  blunderRate: number;
  avgMs: number;
  p95Ms: number;
  avgDepth: number;
  avgNodes: number;
  avgQNodes: number;
  timedOut: number;
  overrides: number;
}

interface TacticalSample {
  caseId: string;
  bucket: string;
  level: Level;
  elapsedMs: number;
  depth: number;
  chosenMove: string;
  oracleMove: string;
  scoreDrop: number;
  solved: boolean;
  severeBlunder: boolean;
  overrideReason?: string;
}

interface BenchmarkReport {
  mode: string;
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
  tacticalSummary: TacticalSummary[];
  tacticalSamples: TacticalSample[];
  releaseGatePassed: boolean;
}

interface CaseFailureSummary {
  caseId: string;
  levels: Level[];
  maxDrop: number;
  severeBlunders: number;
}

interface ClassificationResult {
  classification: Classification;
  fatalReasons: string[];
  warnings: string[];
}

const LEVELS: Level[] = ['easy', 'normal', 'hard', 'expert'];
const DEFAULT_PATH = '.tmp/benchmarks/ai-benchmark-quick-latest.json';
// Keep thresholds easy to edit while the harness remains a standalone local tool.
const CATASTROPHIC_DROP = 500_000;
const WARNING_DROP = 1_000;
const REPEATED_FAILURE_LEVELS = 2;
const FATAL_CATASTROPHIC_CASES = new Set([
  'sac-two-win-three-p1',
  'sac-two-win-three-p2',
  'low-mobility-squeeze',
  'low-mobility-squeeze-p2',
]);
const KNOWN_WARNING_CASES = new Set([
  'small-piece-king-vs-men',
]);

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function parseReport(path: string): BenchmarkReport {
  return JSON.parse(readFileSync(path, 'utf8')) as BenchmarkReport;
}

function summarizeRepeatedFailures(samples: TacticalSample[]): CaseFailureSummary[] {
  const grouped = new Map<string, TacticalSample[]>();
  for (const sample of samples) {
    if (sample.solved && !sample.severeBlunder) continue;
    const rows = grouped.get(sample.caseId);
    if (rows) rows.push(sample);
    else grouped.set(sample.caseId, [sample]);
  }

  return [...grouped.entries()]
    .map(([caseId, rows]) => ({
      caseId,
      levels: rows.map(row => row.level),
      maxDrop: Math.max(...rows.map(row => row.scoreDrop)),
      severeBlunders: rows.filter(row => row.severeBlunder).length,
    }))
    .filter(row => row.levels.length >= REPEATED_FAILURE_LEVELS)
    .sort((a, b) => b.maxDrop - a.maxDrop || b.levels.length - a.levels.length || a.caseId.localeCompare(b.caseId));
}

function classify(report: BenchmarkReport): ClassificationResult {
  const fatalReasons: string[] = [];
  const warnings: string[] = [];
  if (!report.config.openingBookBypassed) {
    fatalReasons.push('openingBookBypassed=no');
  }

  const catastrophic = report.tacticalSamples.filter(sample => sample.scoreDrop >= CATASTROPHIC_DROP);
  const fatalCatastrophic = catastrophic.filter(sample => FATAL_CATASTROPHIC_CASES.has(sample.caseId));
  if (fatalCatastrophic.length) {
    fatalReasons.push(
      `${fatalCatastrophic.length} fatal catastrophic drop(s) in ${[...new Set(fatalCatastrophic.map(row => row.caseId))].join(', ')}`,
    );
  }

  const repeatedCatastrophic = summarizeRepeatedFailures(report.tacticalSamples).filter(
    row => row.maxDrop >= CATASTROPHIC_DROP && row.levels.length >= REPEATED_FAILURE_LEVELS && !KNOWN_WARNING_CASES.has(row.caseId),
  );
  if (repeatedCatastrophic.length) {
    fatalReasons.push(
      `repeated catastrophic failure(s): ${repeatedCatastrophic.map(row => `${row.caseId} [${row.levels.join(',')}]`).join('; ')}`,
    );
  }

  const repeatedFailures = summarizeRepeatedFailures(report.tacticalSamples);
  const warningCatastrophic = catastrophic.filter(sample => KNOWN_WARNING_CASES.has(sample.caseId));
  if (warningCatastrophic.length) {
    warnings.push(
      `known warning case catastrophic drop(s): ${[...new Set(warningCatastrophic.map(row => row.caseId))].join(', ')}`,
    );
  }

  const repeatedWarnings = repeatedFailures.filter(row => !repeatedCatastrophic.some(fatal => fatal.caseId === row.caseId));
  if (repeatedWarnings.length) {
    warnings.push(`${repeatedWarnings.length} repeated miss case(s)`);
  }

  const expert = report.tacticalSummary.find(row => row.level === 'expert');
  const hard = report.tacticalSummary.find(row => row.level === 'hard');

  if (expert && expert.blunderRate > 0.03) {
    warnings.push(`expert blunder ${pct(expert.blunderRate)}`);
  }
  if (hard && hard.blunderRate > 0) {
    warnings.push(`hard blunder ${pct(hard.blunderRate)}`);
  }

  const nonCatMisses = report.tacticalSamples.filter(
    sample => (!sample.solved || sample.severeBlunder) && sample.scoreDrop > 0 && sample.scoreDrop < WARNING_DROP,
  );
  if (nonCatMisses.length) {
    warnings.push(`${nonCatMisses.length} non-cat miss(es) below ${WARNING_DROP} cp`);
  }

  const classification: Classification = fatalReasons.length ? 'FAIL' : warnings.length ? 'WARN' : 'PASS';
  return { classification, fatalReasons, warnings };
}

function printTacticalSummary(rows: TacticalSummary[]): void {
  console.log('level   solve   blunder   avgMs   avgDepth');
  for (const level of LEVELS) {
    const row = rows.find(entry => entry.level === level);
    if (!row) continue;
    console.log(
      `${row.level.padEnd(7)} ${pct(row.solveRate).padStart(5)}` +
      ` ${pct(row.blunderRate).padStart(8)}` +
      ` ${row.avgMs.toFixed(0).padStart(7)}` +
      ` ${row.avgDepth.toFixed(1).padStart(10)}`,
    );
  }
}

function printCatastrophic(samples: TacticalSample[]): void {
  const rows = samples
    .filter(sample => sample.scoreDrop >= CATASTROPHIC_DROP)
    .sort((a, b) => b.scoreDrop - a.scoreDrop || a.caseId.localeCompare(b.caseId));
  if (!rows.length) return;

  console.log('\nCatastrophic signatures');
  for (const row of rows) {
    console.log(
      `${row.level} ${row.caseId}: drop=${row.scoreDrop}, chose ${row.chosenMove}, oracle ${row.oracleMove}`,
    );
  }
}

function printRepeatedFailures(samples: TacticalSample[]): void {
  const repeated = summarizeRepeatedFailures(samples);
  if (!repeated.length) return;

  console.log('\nRepeated tactical failures');
  for (const row of repeated) {
    console.log(
      `${row.caseId}: levels=${row.levels.join(',')}, maxDrop=${row.maxDrop}, severeBlunders=${row.severeBlunders}`,
    );
  }
}

function printMisses(samples: TacticalSample[]): void {
  const misses = samples
    .filter(sample => !sample.solved || sample.severeBlunder)
    .sort((a, b) => b.scoreDrop - a.scoreDrop || a.caseId.localeCompare(b.caseId) || a.level.localeCompare(b.level));
  if (!misses.length) return;

  console.log('\nTactical misses');
  for (const miss of misses.slice(0, 12)) {
    console.log(
      `${miss.level} ${miss.caseId}: drop=${miss.scoreDrop}, chose ${miss.chosenMove}, oracle ${miss.oracleMove}` +
      `${miss.overrideReason ? `, override=${miss.overrideReason}` : ''}`,
    );
  }
  if (misses.length > 12) {
    console.log(`... ${misses.length - 12} more miss(es) omitted`);
  }
}

function main(): void {
  const path = process.argv[2] ?? DEFAULT_PATH;
  if (!existsSync(path)) {
    console.error(`Benchmark report not found: ${path}`);
    process.exitCode = 1;
    return;
  }

  const report = parseReport(path);
  const result = classify(report);

  console.log(`Regression harness scaffold`);
  console.log(`report=${path}`);
  console.log(`mode=${report.mode} generatedAt=${report.generatedAt}`);
  console.log(`openingBookBypassed=${report.config.openingBookBypassed ? 'yes' : 'no'}`);
  console.log(`classification=${result.classification}`);
  console.log(`fatalReasons=${result.fatalReasons.length ? result.fatalReasons.join(' | ') : '(none)'}`);
  console.log(`warnings=${result.warnings.length ? result.warnings.join(' | ') : '(none)'}`);
  console.log('');

  printTacticalSummary(report.tacticalSummary);
  printCatastrophic(report.tacticalSamples);
  printRepeatedFailures(report.tacticalSamples);
  printMisses(report.tacticalSamples);
}

main();
