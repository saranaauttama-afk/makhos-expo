import { join } from 'node:path';
import { runTournament, saveTournament } from './tournamentHarness';
import { TOURNAMENT_START_SUITE } from './tournamentStartSuite';

async function main(){
 const base={id:'baseline',name:'Default Engine A',search:{mode:'nodes' as const,budget:500,maxDepth:12}};
 const identical={...base,id:'candidate',name:'Identical Default Engine B'};
 const controlled={...base,id:'candidate-null-off',name:'Plumbing candidate (nullMove=false)',featureOverrides:{nullMove:false}};
 const identicalRun=await runTournament(base,identical,TOURNAMENT_START_SUITE,{startLimit:2,maxPlies:60}); saveTournament(identicalRun,join('.tmp','tournament','identical'));
 if(identicalRun.summary.baselineWins!==identicalRun.summary.candidateWins) throw new Error('identical engines were not symmetric');
 const plumbingRun=await runTournament(base,controlled,TOURNAMENT_START_SUITE,{startLimit:2,maxPlies:60}); saveTournament(plumbingRun,join('.tmp','tournament','controlled-null-off'));
 console.log(JSON.stringify({identical:identicalRun.summary,controlledDifference:plumbingRun.summary},null,2));
 console.log('Controlled run validates config plumbing only; its small sample is not strength evidence.');
}
main().catch(e=>{console.error(e);process.exitCode=1});
