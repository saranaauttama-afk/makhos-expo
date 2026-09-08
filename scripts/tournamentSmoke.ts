import { join } from 'node:path';
import { runTournament, saveTournament } from './tournamentHarness';
import { TOURNAMENT_START_SUITE } from './tournamentStartSuite';

async function main(){
 const base={id:'baseline',name:'Default Engine A',search:{mode:'nodes' as const,budget:500,maxDepth:12}};
 const identical={...base,id:'candidate',name:'Identical Default Engine B'};
 const controlled={...base,id:'candidate-null-off',name:'Plumbing candidate (nullMove=false)',featureOverrides:{nullMove:false}};
 const identicalRun=await runTournament(base,identical,TOURNAMENT_START_SUITE,{startLimit:2,maxPlies:60}); saveTournament(identicalRun,join('.tmp','tournament','identical'));
 if(identicalRun.summary.baselineWins!==identicalRun.summary.candidateWins) throw new Error('identical engines were not symmetric');
 let candidateSearches=0;
 const plumbingRun=await runTournament(base,controlled,TOURNAMENT_START_SUITE,{startLimit:2,maxPlies:60,onSearch:(engine,features)=>{if(engine.id===controlled.id){candidateSearches++;if(features.nullMove!==false)throw new Error('candidate nullMove=false did not reach search dispatch');}}}); saveTournament(plumbingRun,join('.tmp','tournament','controlled-null-off'));
 if(candidateSearches===0) throw new Error('controlled candidate was never dispatched to search');
 console.log(JSON.stringify({identical:identicalRun.summary,controlledDifference:plumbingRun.summary},null,2));
 console.log(`Controlled run observed ${candidateSearches} candidate searches with effective nullMove=false; this validates config plumbing only, not strength.`);
}
main().catch(e=>{console.error(e);process.exitCode=1});
