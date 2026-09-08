import {mkdirSync,writeFileSync} from 'node:fs';
import {DEFAULT_EXTENSION_FEATURES,ExtensionFeatureFlags} from '../src/coreClaude/search/alphabeta';
import {runTournament,saveTournament} from './tournamentHarness';
import {EXTENSION_CONFIRMATION_START_SUITE,EXTENSION_CONFIRMATION_V1_FINGERPRINT,SEARCH_ABLATION_START_SUITE} from './tournamentStartSuite';
const value=(n:string)=>process.argv.find(x=>x.startsWith(`--${n}=`))?.slice(n.length+3);
const stage=value('stage')??'screening'; const pairs=Number(value('pairs')??(stage==='confirmation'?64:32));const nodes=Number(value('nodes')??5000);const out=value('output')??'.tmp/extension-ablation';
const suite=stage==='confirmation'?EXTENSION_CONFIRMATION_START_SUITE:SEARCH_ABLATION_START_SUITE;
const requested=(value('subtypes')??Object.keys(DEFAULT_EXTENSION_FEATURES).join(',')).split(',') as (keyof ExtensionFeatureFlags)[];
if(stage==='confirmation'&&requested.length!==1)throw new Error('confirmation must test exactly one selected subtype');
(async()=>{mkdirSync(out,{recursive:true});const results=[];for(const subtype of requested){if(!(subtype in DEFAULT_EXTENSION_FEATURES))throw new Error(`unknown subtype ${subtype}`);const search={mode:'nodes' as const,budget:nodes,maxDepth:64};const result=await runTournament({id:'baseline',name:'all extensions enabled',search},{id:`without-${subtype}`,name:`only ${subtype} disabled`,search,extensionOverrides:{[subtype]:false}},suite,{startLimit:pairs,maxPlies:160});saveTournament(result,`${out}/${subtype}`);results.push({subtype,summary:result.summary,baseline:result.engines.baseline,candidate:result.engines[`without-${subtype}`]});console.log(subtype,result.summary);}
writeFileSync(`${out}/summary.json`,JSON.stringify({schemaVersion:'makhos-extension-ablation-summary-v1',stage,suite:suite.version,seed:suite.seed,fingerprint:stage==='confirmation'?EXTENSION_CONFIRMATION_V1_FINGERPRINT:undefined,pairs,nodes,maxDepth:64,maxPlies:160,results},null,2)+'\n');})();
