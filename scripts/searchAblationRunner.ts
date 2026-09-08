import { mkdirSync, writeFileSync } from 'node:fs';
import { DEFAULT_SEARCH_FEATURES, SearchFeatureFlags } from '../src/coreClaude/search/alphabeta';
import { EngineTotals, EloValue, runTournament, saveTournament, TournamentResult } from './tournamentHarness';
import { SEARCH_ABLATION_START_SUITE } from './tournamentStartSuite';

export const SEARCH_ABLATION_FEATURES: (keyof SearchFeatureFlags)[] =
  ['reverseFutility','razoring','nullMove','probCut','iid','lmr','lmp','extensions'];
export type AblationVerdict = 'likely beneficial'|'likely harmful'|'inconclusive';
export function classifyAblation(ci: [number|null, number|null]): AblationVerdict {
  if (ci[1] !== null && ci[1] < .5) return 'likely beneficial';
  if (ci[0] !== null && ci[0] > .5) return 'likely harmful';
  return 'inconclusive';
}
function arg(name:string, fallback:string) { return process.argv.find(x=>x.startsWith(`--${name}=`))?.split('=')[1] ?? fallback; }
function eloText(value:EloValue|null): number|string|null { return value?.kind === 'finite' ? value.value : value?.kind ?? null; }
function metric(t:EngineTotals) { return { averageDepth:t.averageDepth, nodes:t.nodes, qnodes:t.qnodes, nps:t.nps, moves:t.moves }; }
export function conciseResult(feature:keyof SearchFeatureFlags, result:TournamentResult) {
  return { featureDisabled:feature, verdict:classifyAblation(result.summary.scoreConfidenceInterval95),
    summary:{...result.summary,eloDifference:eloText(result.summary.eloDifference),eloConfidenceInterval95:result.summary.eloConfidenceInterval95.map(eloText)},
    baselineMetrics:metric(result.engines.baseline), candidateMetrics:metric(result.engines[`without-${feature}`]) };
}
async function main() {
  const pairs=Number(arg('pairs','32')), nodes=Number(arg('nodes','5000')), maxPlies=Number(arg('max-plies','240'));
  const depthArg=process.argv.find(x=>x.startsWith('--depth='));
  const depth=depthArg ? Number(depthArg.split('=')[1]) : undefined;
  const requested=arg('features',SEARCH_ABLATION_FEATURES.join(',')).split(',') as (keyof SearchFeatureFlags)[];
  if (!Number.isInteger(pairs)||pairs<1||pairs>64||!Number.isInteger(nodes)||nodes<1||!Number.isInteger(maxPlies)||maxPlies<1||
      (depth !== undefined && (!Number.isInteger(depth)||depth<1))) throw new Error('invalid positive integer protocol argument');
  if (requested.some(f=>!SEARCH_ABLATION_FEATURES.includes(f))) throw new Error('unknown feature in --features');
  const output=arg('output','.tmp/search-ablation'); mkdirSync(output,{recursive:true}); const results=[];
  for (const feature of requested) {
    const search = depth === undefined ? {mode:'nodes' as const,budget:nodes,maxDepth:64} : {mode:'depth' as const,budget:depth};
    const baseline={id:'baseline',name:'all default search features',search};
    const candidate={id:`without-${feature}`,name:`default except ${feature}=false`,search,featureOverrides:{[feature]:false}};
    const result=await runTournament(baseline,candidate,SEARCH_ABLATION_START_SUITE,{startLimit:pairs,maxPlies});
    saveTournament(result,`${output}/${feature}`); results.push(conciseResult(feature,result));
    console.log(`${feature}: ${result.summary.completedPairs}/${pairs} pairs, score=${result.summary.candidateScorePercent ?? 'n/a'}%, verdict=${results.at(-1)!.verdict}`);
  }
  const summary={schemaVersion:'makhos-search-ablation-summary-v1',protocol:{mode:depth===undefined?'fixed-nodes':'fixed-depth',nodesPerMove:depth===undefined?nodes:undefined,depth,pairedStarts:pairs,maxDepth:depth??64,maxPlies,startSuiteVersion:SEARCH_ABLATION_START_SUITE.version,startSuiteSeed:SEARCH_ABLATION_START_SUITE.seed,defaults:DEFAULT_SEARCH_FEATURES,singleFeatureOnly:true},results};
  writeFileSync(`${output}/summary.json`,JSON.stringify(summary,null,2)+'\n');
  const csv=['featureDisabled,pairs,completedPairs,scorePercent,elo,ciLow,ciHigh,averageDepth,nodes,qnodes,nps,unresolved,errors,verdict',...results.map(r=>[r.featureDisabled,pairs,r.summary.completedPairs,r.summary.candidateScorePercent,r.summary.eloDifference,...r.summary.eloConfidenceInterval95,r.candidateMetrics.averageDepth,r.candidateMetrics.nodes,r.candidateMetrics.qnodes,r.candidateMetrics.nps,r.summary.unresolved,r.summary.errors,r.verdict].join(','))];
  writeFileSync(`${output}/summary.csv`,csv.join('\n')+'\n');
}
if(require.main===module) main().catch(e=>{console.error(e);process.exitCode=1});
