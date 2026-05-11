import { copyFileSync, mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { join } from 'path';
import {
  classify,
  parseReport,
  type BenchmarkReport,
  type Level,
  type TacticalSample,
} from './regressionHarness';

const RUN_COUNT = 3;
const DEFAULT_REPORT_PATH = '.tmp/benchmarks/ai-benchmark-quick-latest.json';
const BENCHMARK_DIR = join('.tmp', 'benchmarks');
const CASE_IDS = [
  'sac-two-win-three-p1',
  'sac-two-win-three-p2',
  'low-mobility-squeeze',
  'low-mobility-squeeze-p2',
  'quiet-hanging-piece-p1',
  'small-piece-king-vs-men',
] as const;
const LEVELS: Level[] = ['easy', 'normal', 'hard', 'expert'];
const CATASTROPHIC_DROP = 500_000;
const KNOWN_WARNING_CASES = new Set(['small-piece-king-vs-men']);

interface RunRecord {
  index: number;
  report: BenchmarkReport;
  classification: string;
  fatalReasons: string[];
  artifactPath: string;
}

function clearedEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.MAKHOS_ENABLE_EVAL_EXPERIMENTS;
  delete env.MAKHOS_ENABLE_LOW_MOBILITY_RESEARCH;
  delete env.MAKHOS_LOW_MOBILITY_RESEARCH_SCALE_PCT;
  return env;
}

function levelOrder(level: Level): number {
  return LEVELS.indexOf(level);
}

function samplesForCase(report: BenchmarkReport, caseId: string): TacticalSample[] {
  return report.tacticalSamples
    .filter(sample => sample.caseId === caseId)
    .sort((a, b) => levelOrder(a.level) - levelOrder(b.level));
}

function dropVector(samples: TacticalSample[]): string {
  return LEVELS.map(level => samples.find(sample => sample.level === level)?.scoreDrop ?? 'n/a').join('/');
}

function vectorStable(records: RunRecord[], caseId: string): boolean {
  const vectors = records.map(record => dropVector(samplesForCase(record.report, caseId)));
  return new Set(vectors).size <= 1;
}

function stableWeakCase(records: RunRecord[], caseId: string): boolean {
  if (!vectorStable(records, caseId)) return false;
  const samples = samplesForCase(records[0].report, caseId);
  return samples.some(sample => sample.scoreDrop > 0 && sample.scoreDrop < CATASTROPHIC_DROP);
}

function repeatedFailCase(records: RunRecord[], caseId: string): boolean {
  if (KNOWN_WARNING_CASES.has(caseId)) return false;
  const catastrophicRuns = records.filter(record =>
    samplesForCase(record.report, caseId).some(sample => sample.scoreDrop >= CATASTROPHIC_DROP),
  );
  return catastrophicRuns.length >= 2;
}

function runFreshBenchmark(index: number): RunRecord {
  mkdirSync(BENCHMARK_DIR, { recursive: true });
  execSync('npm run bench:ai:fresh', {
    cwd: process.cwd(),
    env: clearedEnv(),
    stdio: 'ignore',
  });
  const report = parseReport(DEFAULT_REPORT_PATH);
  const result = classify(report);
  const artifactPath = join(BENCHMARK_DIR, `ai-benchmark-quick-repeat-run${index}.json`);
  copyFileSync(DEFAULT_REPORT_PATH, artifactPath);
  return {
    index,
    report,
    classification: result.classification,
    fatalReasons: result.fatalReasons,
    artifactPath,
  };
}

function printNamedCases(records: RunRecord[]): void {
  console.log('\nNamed case drops (easy/normal/hard/expert)');
  for (const caseId of CASE_IDS) {
    console.log(caseId);
    for (const record of records) {
      const samples = samplesForCase(record.report, caseId);
      console.log(`  run${record.index}: ${dropVector(samples)}`);
    }
  }
}

function printCaseBuckets(records: RunRecord[]): void {
  const unstable = CASE_IDS.filter(caseId => !vectorStable(records, caseId));
  const stableWeak = CASE_IDS.filter(caseId => stableWeakCase(records, caseId));
  const repeatedFail = CASE_IDS.filter(caseId => repeatedFailCase(records, caseId));

  console.log('\nRepeated FAIL cases');
  if (!repeatedFail.length) console.log('(none)');
  else for (const caseId of repeatedFail) console.log(caseId);

  console.log('\nUnstable cases');
  if (!unstable.length) console.log('(none)');
  else for (const caseId of unstable) console.log(caseId);

  console.log('\nStable weak cases');
  if (!stableWeak.length) console.log('(none)');
  else for (const caseId of stableWeak) console.log(caseId);
}

function main(): void {
  const records: RunRecord[] = [];
  console.log('Repeated-run quick benchmark summary');
  console.log('mode=clean OFF, runs=3');
  for (let i = 1; i <= RUN_COUNT; i++) {
    const record = runFreshBenchmark(i);
    records.push(record);
    console.log(
      `run${record.index}: generatedAt=${record.report.generatedAt} ` +
      `classification=${record.classification} ` +
      `fatalReasons=${record.fatalReasons.length ? record.fatalReasons.join(' | ') : '(none)'}`,
    );
  }
  printNamedCases(records);
  printCaseBuckets(records);
}

main();
