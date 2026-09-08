import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DEFAULT_SEARCH_FEATURES, SearchFeatureFlags } from '../src/coreClaude/search/alphabeta';
import { EngineTotals, EloValue, runTournament, saveTournament, TournamentResult } from './tournamentHarness';
import { SEARCH_ABLATION_START_SUITE, SEARCH_ABLATION_START_SUITE_V1_FINGERPRINT } from './tournamentStartSuite';

export const SEARCH_ABLATION_CONFIG_PATH = 'config/search-ablation-v1.json';
export const SEARCH_ABLATION_FEATURES: (keyof SearchFeatureFlags)[] =
  ['reverseFutility','razoring','nullMove','probCut','iid','lmr','lmp','extensions'];
type Stage = 'screening'|'confirmation';
interface StageConfig { pairedStarts:number; nodesPerMove:number; maxDepth:number; maxPlies:number }
export interface SearchAblationConfig { schemaVersion:string; protocolVersion:string; startSuite:string; seed:number;
  startSuiteFingerprint:string; screening:StageConfig; confirmation:StageConfig; features:(keyof SearchFeatureFlags)[]; verdict:string }
export function loadSearchAblationConfig(path=SEARCH_ABLATION_CONFIG_PATH): SearchAblationConfig {
  return JSON.parse(readFileSync(resolve(process.cwd(),path),'utf8')) as SearchAblationConfig;
}
export function frozenStageDefaults(config:SearchAblationConfig,stage:Stage):StageConfig{return{...config[stage]};}
export type AblationVerdict = 'likely beneficial'|'likely harmful'|'inconclusive';
export function classifyAblation(ci:[number|null,number|null]):AblationVerdict {
  if(ci[1]!==null&&ci[1]<.5)return 'likely beneficial';
  if(ci[0]!==null&&ci[0]>.5)return 'likely harmful';
  return 'inconclusive';
}
function value(name:string){return process.argv.find(x=>x.startsWith(`--${name}=`))?.slice(name.length+3);}
function positive(name:string,fallback:number){const n=Number(value(name)??fallback);if(!Number.isInteger(n)||n<1)throw new Error(`${name} must be a positive integer`);return n;}
function eloText(v:EloValue|null):number|string|null{return v?.kind==='finite'?v.value:v?.kind??null;}
function metric(t:EngineTotals){return{averageDepth:t.averageDepth,nodes:t.nodes,qnodes:t.qnodes,nps:t.nps,moves:t.moves};}
export function conciseResult(feature:keyof SearchFeatureFlags,result:TournamentResult){return{featureDisabled:feature,
  verdict:classifyAblation(result.summary.scoreConfidenceInterval95),summary:{...result.summary,eloDifference:eloText(result.summary.eloDifference),
  eloConfidenceInterval95:result.summary.eloConfidenceInterval95.map(eloText)},baselineMetrics:metric(result.engines.baseline),candidateMetrics:metric(result.engines[`without-${feature}`])};}

async function main(){
  const config=loadSearchAblationConfig();
  if(config.startSuite!==SEARCH_ABLATION_START_SUITE.version||config.seed!==SEARCH_ABLATION_START_SUITE.seed||
    config.startSuiteFingerprint!==SEARCH_ABLATION_START_SUITE_V1_FINGERPRINT)throw new Error('frozen start suite does not match canonical config');
  const stage=(value('stage')??'screening') as Stage;if(stage!=='screening'&&stage!=='confirmation')throw new Error('stage must be screening or confirmation');
  const frozen=frozenStageDefaults(config,stage),pairs=positive('pairs',frozen.pairedStarts),nodes=positive('nodes',frozen.nodesPerMove),
    maxDepth=positive('max-depth',frozen.maxDepth),maxPlies=positive('max-plies',frozen.maxPlies);
  if(pairs>SEARCH_ABLATION_START_SUITE.starts.length)throw new Error('pairs exceeds frozen corpus');
  const depthValue=value('depth'),depth=depthValue===undefined?undefined:positive('depth',1);
  const requested=(value('features')??config.features.join(',')).split(',') as (keyof SearchFeatureFlags)[];
  if(requested.some(f=>!config.features.includes(f)))throw new Error('unknown feature in --features');
  const output=value('output')??'.tmp/search-ablation';mkdirSync(output,{recursive:true});const results=[];
  for(const feature of requested){
    const search=depth===undefined?{mode:'nodes' as const,budget:nodes,maxDepth}:{mode:'depth' as const,budget:depth};
    const baseline={id:'baseline',name:'all default search features',search};
    const candidate={id:`without-${feature}`,name:`default except ${feature}=false`,search,featureOverrides:{[feature]:false}};
    const result=await runTournament(baseline,candidate,SEARCH_ABLATION_START_SUITE,{startLimit:pairs,maxPlies});
    saveTournament(result,`${output}/${feature}`);results.push(conciseResult(feature,result));
    console.log(`${feature}: ${result.summary.completedPairs}/${pairs} pairs, score=${result.summary.candidateScorePercent??'n/a'}%, verdict=${results.at(-1)!.verdict}`);
  }
  const summary={schemaVersion:'makhos-search-ablation-summary-v1',protocol:{protocolVersion:config.protocolVersion,
    configSchemaVersion:config.schemaVersion,configPath:SEARCH_ABLATION_CONFIG_PATH,stage,mode:depth===undefined?'fixed-nodes':'fixed-depth',
    effectiveParameters:{nodesPerMove:depth===undefined?nodes:undefined,depth,pairedStarts:pairs,maxDepth:depth??maxDepth,maxPlies},
    startSuiteVersion:SEARCH_ABLATION_START_SUITE.version,startSuiteSeed:SEARCH_ABLATION_START_SUITE.seed,
    startSuiteFingerprint:SEARCH_ABLATION_START_SUITE_V1_FINGERPRINT,defaults:DEFAULT_SEARCH_FEATURES,singleFeatureOnly:true},results};
  writeFileSync(`${output}/summary.json`,JSON.stringify(summary,null,2)+'\n');
  const csv=['featureDisabled,pairs,completedPairs,scorePercent,elo,ciLow,ciHigh,averageDepth,nodes,qnodes,nps,unresolved,errors,verdict',...results.map(r=>[r.featureDisabled,pairs,r.summary.completedPairs,r.summary.candidateScorePercent,r.summary.eloDifference,...r.summary.eloConfidenceInterval95,r.candidateMetrics.averageDepth,r.candidateMetrics.nodes,r.candidateMetrics.qnodes,r.candidateMetrics.nps,r.summary.unresolved,r.summary.errors,r.verdict].join(','))];
  writeFileSync(`${output}/summary.csv`,csv.join('\n')+'\n');
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1});
