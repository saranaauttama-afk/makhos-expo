import { mkdirSync, writeFileSync } from 'node:fs';
import { DEFAULT_EXTENSION_FEATURES } from '../src/coreClaude/search/alphabeta';
import { runTournament, saveTournament, SearchControl } from './tournamentHarness';
import { PHASE3D_CONFIRMATION_START_SUITE, PHASE3D_CONFIRMATION_V1_FINGERPRINT } from './tournamentStartSuite';

const value = (name: string) => process.argv.find(x => x.startsWith(`--${name}=`))?.slice(name.length + 3);
const pairs = Number(value('pairs') ?? 32);
const maxPlies = Number(value('max-plies') ?? 160);
const nodeBudgets = (value('nodes') ?? '20000,50000').split(',').map(Number);
const timeBudgets = (value('times') ?? '100').split(',').map(Number);
const output = value('output') ?? '.tmp/phase3d';
if (!Number.isInteger(pairs) || pairs < 1 || pairs > 64) throw new Error('pairs must select 1..64 frozen starts');
if ([...nodeBudgets, ...timeBudgets].some(n => !Number.isInteger(n) || n < 1)) throw new Error('budgets must be positive integers');

async function run(control: SearchControl, label: string) {
  // Pin every extension flag on both engines. This prevents changing defaults
  // from silently changing the historical experiment configuration.
  const baselineExtensions = { ...DEFAULT_EXTENSION_FEATURES, smallEndgame: false, soundForcedTrap: true };
  const candidateExtensions = { ...baselineExtensions, soundForcedTrap: false };
  const baseline = { id: 'baseline', name: 'Teacher v1; soundForcedTrap=true', search: control, extensionOverrides: baselineExtensions };
  const candidate = { id: 'candidate', name: 'Teacher v1; soundForcedTrap=false', search: control, extensionOverrides: candidateExtensions };
  const result = await runTournament(baseline, candidate, PHASE3D_CONFIRMATION_START_SUITE,
    { startLimit: pairs, maxPlies, allowDescriptiveStatistics: control.mode === 'time' });
  saveTournament(result, `${output}/${label}`);
  console.log(label, result.summary, result.engines);
  return { label, summary: result.summary, baseline: result.engines.baseline, candidate: result.engines.candidate,
    comparability: result.metadata.comparability };
}

(async () => {
  mkdirSync(output, { recursive: true });
  const results = [];
  for (const budget of nodeBudgets) results.push(await run({ mode: 'nodes', budget, maxDepth: 64 }, `nodes-${budget}`));
  for (const budget of timeBudgets) results.push(await run({ mode: 'time', budget, maxDepth: 64 }, `time-${budget}ms`));
  writeFileSync(`${output}/summary.json`, JSON.stringify({ schemaVersion: 'makhos-phase3d-confirmation-v1',
    teacherIdentity: 'b2e6a35db6a50ea294a10f6b76a90b4e70e0689f', suite: PHASE3D_CONFIRMATION_START_SUITE.version,
    seed: PHASE3D_CONFIRMATION_START_SUITE.seed, fingerprint: PHASE3D_CONFIRMATION_V1_FINGERPRINT,
    pairs, maxPlies, maxDepth: 64, nodeBudgets, timeBudgets, results }, null, 2) + '\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
