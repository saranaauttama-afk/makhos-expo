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
type CaseId = (typeof CASE_IDS)[number];
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

function allSamples(records: RunRecord[], caseId: CaseId): TacticalSample[] {
  return records.flatMap(record => samplesForCase(record.report, caseId));
}

function stableCleanCase(records: RunRecord[], caseId: CaseId): boolean {
  if (!vectorStable(records, caseId)) return false;
  return samplesForCase(records[0].report, caseId).every(sample => sample.scoreDrop === 0);
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

function hasOverridePattern(records: RunRecord[], caseId: CaseId): boolean {
  const samples = allSamples(records, caseId);
  return samples.some(sample => !!sample.overrideReason) && samples.some(sample => !sample.overrideReason && sample.scoreDrop > 0);
}

function classifyCase(records: RunRecord[], caseId: CaseId): { label: string; interpretation: string } {
  const unstable = !vectorStable(records, caseId);
  const stableWeak = stableWeakCase(records, caseId);
  const stableClean = stableCleanCase(records, caseId);
  const repeatedFail = repeatedFailCase(records, caseId);

  if (caseId === 'small-piece-king-vs-men') {
    return {
      label: 'unstable-oracle-probe',
      interpretation: 'Known warning-only case; repeated-run variance should be interpreted through oracle/probe instability first.',
    };
  }

  if (caseId === 'low-mobility-squeeze-p2' && stableWeak) {
    return {
      label: 'stable-weakness',
      interpretation: 'Consistent non-cat weakness across runs; useful as a stable measurement case.',
    };
  }

  if (caseId === 'sac-two-win-three-p1' && (unstable || repeatedFail)) {
    return {
      label: 'unstable-search-benchmark',
      interpretation: 'High-value tactical case with repeated-run volatility; treat single quick artifacts cautiously.',
    };
  }

  if (caseId === 'low-mobility-squeeze' && (hasOverridePattern(records, caseId) || unstable)) {
    return {
      label: 'override-sensitive / unstable-search-benchmark',
      interpretation: 'Result depends on root override behavior and can swing between clean and severe expert misses across runs.',
    };
  }

  if (caseId === 'quiet-hanging-piece-p1') {
    if (stableClean) {
      return {
        label: 'currently-clean',
        interpretation: 'Currently stable and solved in repeat output; any future movement likely needs separate oracle-noise checking.',
      };
    }
    return {
      label: 'currently-clean-or-oracle-noisy',
      interpretation: 'Case is not reliably weak in repeat output; interpret drift cautiously before treating it as a tuning target.',
    };
  }

  if (caseId === 'sac-two-win-three-p2') {
    if (stableClean) {
      return {
        label: 'clean-stable',
        interpretation: 'Stable clean tactical guardrail in repeat output.',
      };
    }
    return {
      label: 'unstable-search-benchmark',
      interpretation: 'Unexpected drift on a guardrail case; investigate benchmark/search instability before tuning.',
    };
  }

  if (stableWeak) {
    return {
      label: 'stable-weakness',
      interpretation: 'Consistent weakness across runs.',
    };
  }
  if (stableClean) {
    return {
      label: 'clean-stable',
      interpretation: 'Stable clean case across runs.',
    };
  }
  return {
    label: 'unstable-search-benchmark',
    interpretation: 'Run-to-run variance suggests benchmark/search instability rather than a settled weakness classification.',
  };
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

function printCaseInterpretation(records: RunRecord[]): void {
  console.log('\nCase classification');
  for (const caseId of CASE_IDS) {
    const runs = records
      .map(record => `run${record.index}=${dropVector(samplesForCase(record.report, caseId))}`)
      .join(' ');
    const classification = classifyCase(records, caseId);
    console.log(`${caseId}: ${runs}`);
    console.log(`  label=${classification.label}`);
    console.log(`  interpretation=${classification.interpretation}`);
  }
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
  printCaseInterpretation(records);
}

main();
