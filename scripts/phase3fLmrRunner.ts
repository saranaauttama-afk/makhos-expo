import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import {
  DEFAULT_EXTENSION_FEATURES,
  DEFAULT_MOVE_ORDERING_FEATURES,
  DEFAULT_SEARCH_FEATURES,
  ExtensionFeatureFlags,
  LmrProfile,
  SearchFeatureFlags,
} from '../src/coreClaude/search/alphabeta';
import {
  EloValue,
  EngineConfig,
  EngineTotals,
  runTournament,
  saveTournament,
  SearchControl,
  TournamentResult,
} from './tournamentHarness';
import {
  PHASE3F_LMR_START_SUITE,
  PHASE3F_LMR_V1_FINGERPRINT,
} from './tournamentStartSuite';

const CONFIG = 'config/phase3f-lmr-v1.json';
type Candidate = Exclude<LmrProfile, 'current'>;
type Config = {
  teacherIdentity: string;
  suite: string;
  seed: number;
  fingerprint: string;
  inventory: Candidate[];
  selectionRule: string;
  screening: { startOffset: number; pairedStarts: number; nodesPerMove: number; maxDepth: number; maxPlies: number };
  confirmation: { startOffset: number; pairedStarts: number; nodeBudgets: number[]; timeBudgetsMs: number[]; maxDepth: number; maxPlies: number };
  baselineSearch: SearchFeatureFlags;
  baselineExtensions: ExtensionFeatureFlags;
  baselineMoveOrdering: typeof DEFAULT_MOVE_ORDERING_FEATURES;
  baselineLmrProfile: 'current';
};

const config = JSON.parse(readFileSync(CONFIG, 'utf8')) as Config;
const arg = (name: string) => process.argv.find(x => x.startsWith(`--${name}=`))?.slice(name.length + 3);
const elo = (value: EloValue | null) => value?.kind === 'finite' ? value.value : value?.kind ?? null;
const metrics = (totals: EngineTotals) => ({
  averageDepth: totals.averageDepth,
  nodes: totals.nodes,
  qnodes: totals.qnodes,
  elapsedMs: totals.elapsedMs,
  nps: totals.nps,
  moves: totals.moves,
  lmrStats: totals.lmrStats,
});

function concise(feature: Candidate, result: TournamentResult) {
  return {
    profile: feature,
    wdl: {
      wins: result.summary.scoreEligibleCandidateWins,
      draws: result.summary.scoreEligibleDraws,
      losses: result.summary.scoreEligibleBaselineWins,
    },
    scorePercent: result.summary.candidateScorePercent,
    elo: elo(result.summary.eloDifference),
    scoreCI95: result.summary.scoreConfidenceInterval95,
    eloCI95: result.summary.eloConfidenceInterval95.map(elo),
    baselineMetrics: metrics(result.engines.baseline),
    candidateMetrics: metrics(result.engines[`profile-${feature}`]),
    unresolved: result.summary.unresolved,
    errors: result.summary.errors,
  };
}

function validate() {
  if (config.suite !== PHASE3F_LMR_START_SUITE.version ||
      config.seed !== PHASE3F_LMR_START_SUITE.seed ||
      config.fingerprint !== PHASE3F_LMR_V1_FINGERPRINT) {
    throw new Error('frozen Phase 3F corpus identity mismatch');
  }
  if (JSON.stringify(config.baselineSearch) !== JSON.stringify(DEFAULT_SEARCH_FEATURES) ||
      JSON.stringify(config.baselineExtensions) !== JSON.stringify(DEFAULT_EXTENSION_FEATURES) ||
      JSON.stringify(config.baselineMoveOrdering) !== JSON.stringify(DEFAULT_MOVE_ORDERING_FEATURES) || config.baselineLmrProfile !== 'current') {
    throw new Error('explicit Teacher v1 baseline no longer matches production semantics');
  }
}

/** Exported so tests can prove that historical measurement collection is
 * explicitly enabled for both sides without running a tournament. */
export function makePhase3fEnginePair(feature: Candidate, search: SearchControl): [EngineConfig, EngineConfig] {
  const instrument = search.mode === 'nodes';
  const baselineMeasurement = { lmrProfile:'current', ...(instrument?{collectLmrStats:true}:{}) } as const;
  const candidateMeasurement = { lmrProfile:feature, ...(instrument?{collectLmrStats:true}:{}) } as const;
  return [
    {
      id: 'baseline',
      name: 'Teacher v1 explicit baseline',
      search,
      featureOverrides: config.baselineSearch,
      extensionOverrides: config.baselineExtensions,
      moveOrderingOverrides: config.baselineMoveOrdering,
      measurementOptions: baselineMeasurement,
    },
    {
      id: `profile-${feature}`,
      name: `Teacher v1 LMR ${feature}`,
      search,
      featureOverrides: config.baselineSearch,
      extensionOverrides: config.baselineExtensions,
      moveOrderingOverrides: config.baselineMoveOrdering,
      measurementOptions: candidateMeasurement,
    },
  ];
}

async function run(feature: Candidate, control: SearchControl, offset: number, pairs: number,
  maxPlies: number, label: string, output: string) {
  const [baseline, candidate] = makePhase3fEnginePair(feature, control);
  const result = await runTournament(baseline, candidate, PHASE3F_LMR_START_SUITE, {
    startOffset: offset,
    startLimit: pairs,
    maxPlies,
    allowDescriptiveStatistics: control.mode === 'time',
  });
  saveTournament(result, `${output}/${label}`);
  return concise(feature, result);
}

async function main() {
  validate();
  const stage = arg('stage') ?? 'screening';
  const output = arg('output') ?? `.tmp/phase3f-${stage}`;
  mkdirSync(output, { recursive: true });
  const results = [];
  if (stage === 'screening') {
    const screening = config.screening;
    const requested = (arg('profiles') ?? config.inventory.join(',')).split(',') as Candidate[];
    if (requested.some(feature => !config.inventory.includes(feature))) throw new Error('unknown --profiles candidate');
    for (const feature of requested) {
      const result = await run(feature, { mode: 'nodes', budget: screening.nodesPerMove, maxDepth: screening.maxDepth },
        screening.startOffset, screening.pairedStarts, screening.maxPlies, feature, output);
      results.push(result);
      console.log(feature, result.wdl, result.scorePercent, result.eloCI95);
    }
    const favorable = results.filter(result => (result.scorePercent ?? 0) > 50)
      .sort((a,b)=>(b.scorePercent??0)-(a.scorePercent??0)||config.inventory.indexOf(a.profile)-config.inventory.indexOf(b.profile));
    const selected = favorable[0]?.profile ?? null;
    writeFileSync(`${output}/summary.json`, JSON.stringify({ schemaVersion: 'makhos-phase3f-screening-v1',
      config, selected, decision: selected ? 'confirm strongest point estimate' : 'inconclusive; stop', results }, null, 2) + '\n');
    console.log('selected', selected);
    return;
  }
  if (stage !== 'confirmation') throw new Error('stage must be screening or confirmation');
  const feature = arg('profile') as Candidate;
  if (!config.inventory.includes(feature)) throw new Error('--profile must be the mechanically selected screening candidate');
  const confirmation = config.confirmation;
  const requestedNodes = (arg('nodes') ?? confirmation.nodeBudgets.join(',')).split(',').filter(Boolean).map(Number);
  const requestedTimes = (arg('times') ?? confirmation.timeBudgetsMs.join(',')).split(',').filter(Boolean).map(Number);
  if (requestedNodes.some(n => !confirmation.nodeBudgets.includes(n)) ||
      requestedTimes.some(n => !confirmation.timeBudgetsMs.includes(n))) {
    throw new Error('confirmation budgets must be frozen values');
  }
  for (const nodes of requestedNodes) {
    results.push(await run(feature, { mode: 'nodes', budget: nodes, maxDepth: confirmation.maxDepth },
      confirmation.startOffset, confirmation.pairedStarts, confirmation.maxPlies, `nodes-${nodes}`, output));
  }
  for (const ms of requestedTimes) {
    results.push(await run(feature, { mode: 'time', budget: ms, maxDepth: confirmation.maxDepth },
      confirmation.startOffset, confirmation.pairedStarts, confirmation.maxPlies, `time-${ms}ms`, output));
  }
  writeFileSync(`${output}/summary.json`, JSON.stringify({ schemaVersion: 'makhos-phase3f-confirmation-v1',
    config, profile:feature, results }, null, 2) + '\n');
}

if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
