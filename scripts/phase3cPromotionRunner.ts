import { mkdirSync, writeFileSync } from 'node:fs';
import { runTournament, saveTournament, SearchControl } from './tournamentHarness';
import { PHASE3C_CONFIRMATION_START_SUITE, PHASE3C_CONFIRMATION_V1_FINGERPRINT } from './tournamentStartSuite';

const value = (name: string) => process.argv.find(x => x.startsWith(`--${name}=`))?.slice(name.length + 3);
const pairs = Number(value('pairs') ?? 32);
const maxPlies = Number(value('max-plies') ?? 160);
const nodeBudgets = (value('nodes') ?? '5000,20000,50000').split(',').map(Number);
const timeBudgets = (value('times') ?? '100').split(',').map(Number);
const output = value('output') ?? '.tmp/phase3c';

if (!Number.isInteger(pairs) || pairs < 1 || pairs > PHASE3C_CONFIRMATION_START_SUITE.starts.length)
  throw new Error('pairs must select 1..64 frozen confirmation starts');
if ([...nodeBudgets, ...timeBudgets].some(n => !Number.isInteger(n) || n < 1))
  throw new Error('all budgets must be positive integers');

async function run(control: SearchControl, label: string) {
  // Pin both sides explicitly so this historical experiment remains replayable
  // after the candidate becomes the production default.
  const baseline = { id: 'baseline', name: 'pre-Teacher-v1 baseline (smallEndgame enabled)', search: control,
    extensionOverrides: { smallEndgame: true } };
  const candidate = { id: 'candidate', name: 'candidate (only smallEndgame disabled)', search: control,
    extensionOverrides: { smallEndgame: false } };
  const result = await runTournament(baseline, candidate, PHASE3C_CONFIRMATION_START_SUITE,
    { startLimit: pairs, maxPlies, allowDescriptiveStatistics: control.mode === 'time' });
  saveTournament(result, `${output}/${label}`);
  console.log(label, result.summary, result.engines);
  return { label, summary: result.summary, baseline: result.engines.baseline,
    candidate: result.engines.candidate, comparability: result.metadata.comparability };
}

(async () => {
  mkdirSync(output, { recursive: true });
  const results = [];
  for (const budget of nodeBudgets) results.push(await run({ mode: 'nodes', budget, maxDepth: 64 }, `nodes-${budget}`));
  for (const budget of timeBudgets) results.push(await run({ mode: 'time', budget, maxDepth: 64 }, `time-${budget}ms`));
  writeFileSync(`${output}/summary.json`, JSON.stringify({
    schemaVersion: 'makhos-phase3c-promotion-v1', suite: PHASE3C_CONFIRMATION_START_SUITE.version,
    seed: PHASE3C_CONFIRMATION_START_SUITE.seed, fingerprint: PHASE3C_CONFIRMATION_V1_FINGERPRINT,
    pairs, maxPlies, nodeBudgets, timeBudgets, results,
  }, null, 2) + '\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
