import { strict as assert } from 'node:assert';
import { assessComparability, calculatePairStatistics, playTournamentGame, runTournament, SearchProvider } from './tournamentHarness';
import { generateTournamentStartSuite } from './tournamentStartSuite';
import { applyMove } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';
import { generateMoves, Move } from '../src/coreClaude/movegen';

const tinyEngine = (id:string) => ({id,name:id,search:{mode:'nodes' as const,budget:1,maxDepth:4}});
const firstLegal: SearchProvider = async (pos) => ({best:generateMoves(pos)[0],score:0,nodes:1,qnodes:0,depth:1,elapsedMs:0,timedOut:false,pv:[]});
const startAt = (id:string, position:ReturnType<typeof initialPosition>, openingMoves:Move[]=[]) => ({id,initialPosition:openingMoves.length ? undefined : position,position,openingMoves});

async function main() {
  const suite1=generateTournamentStartSuite(0x4d414b48,2), suite2=generateTournamentStartSuite(0x4d414b48,2);
  assert.deepEqual(suite1,suite2); assert.equal(new Set(suite1.starts.map(s=>JSON.stringify(s.position))).size,2);
  assert.deepEqual(generateTournamentStartSuite().starts.map(s=>s.position), [
    {side:1,p1Men:4215275520,p1Kings:0,p2Men:703,p2Kings:0,halfmoveClock:2},
    {side:-1,p1Men:4200595456,p1Kings:0,p2Men:735,p2Kings:0,halfmoveClock:3},
    {side:1,p1Men:3742367744,p1Kings:0,p2Men:510,p2Kings:0,halfmoveClock:4},
    {side:-1,p1Men:4232118272,p1Kings:0,p2Men:8415,p2Kings:0,halfmoveClock:5},
    {side:1,p1Men:4175429632,p1Kings:0,p2Men:1725,p2Kings:0,halfmoveClock:6},
    {side:-1,p1Men:4202692608,p1Kings:0,p2Men:8351,p2Kings:0,halfmoveClock:1},
    {side:1,p1Men:4179623936,p1Kings:0,p2Men:33311,p2Kings:0,halfmoveClock:0},
    {side:1,p1Men:4215275520,p1Kings:0,p2Men:2175,p2Kings:0,halfmoveClock:2},
  ], 'v1 suite changed without a version bump');
  for(const start of suite1.starts) assert.deepEqual(start.openingMoves.reduce((p,m)=>applyMove(p,m),initialPosition()),start.position);
  const a={id:'a',name:'Default A',search:{mode:'nodes' as const,budget:120,maxDepth:8}}, b={...a,id:'b',name:'Default B'};
  const first=await runTournament(a,b,suite1,{maxPlies:24,generatedAt:'stable'}), second=await runTournament(a,b,suite2,{maxPlies:24,generatedAt:'stable'});
  const deterministic = (r: typeof first) => r.pairs.map(p=>p.games.map(g=>({
    result:g.result,status:g.status,reason:g.reason,winner:g.winnerEngineId,moves:g.moves,
    metrics:g.moveMetrics.map(({elapsedMs,nps,...metric})=>metric),
  })));
  assert.deepEqual(deterministic(first),deterministic(second),'same seed/config must reproduce outcomes, moves and deterministic search metrics');
  assert.deepEqual(first.summary,second.summary,'same seed/config must reproduce aggregate strength statistics');
  for(const pair of first.pairs){ assert.equal(pair.games[0].p1EngineId,'a');assert.equal(pair.games[1].p1EngineId,'b');assert.deepEqual(pair.games[0].moves,pair.games[1].moves); }
  assert.equal(first.summary.baselineWins,first.summary.candidateWins); assert.equal(first.summary.candidateScorePercent,first.summary.completedGames?50:null);
  const secondStartOnly=await runTournament(a,b,suite1,{startOffset:1,startLimit:1,maxPlies:1,searchProvider:firstLegal});
  assert.deepEqual(secondStartOnly.starts.map(s=>s.id),[suite1.starts[1].id]);
  assert.equal(secondStartOnly.summary.pairedStarts,1);
  await assert.rejects(()=>runTournament(a,b,suite1,{startOffset:3,startLimit:1}),/startOffset/);

  assert.deepEqual(assessComparability({mode:'nodes',budget:500,maxDepth:8},{mode:'nodes',budget:500,maxDepth:8}).status,'canonical');
  assert.equal(assessComparability({mode:'nodes',budget:500},{mode:'nodes',budget:5000}).status,'nonComparable');
  assert.equal(assessComparability({mode:'depth',budget:4},{mode:'depth',budget:5}).status,'nonComparable');
  const unequal=await runTournament(a,{...b,search:{mode:'nodes',budget:5000,maxDepth:8}},suite1,{maxPlies:1,searchProvider:firstLegal});
  assert.equal(unequal.metadata.statisticsSuppressed,true); assert.equal(unequal.summary.eloDifference,null); assert.deepEqual(unequal.summary.scoreConfidenceInterval95,[null,null]);

  const allLoss=calculatePairStatistics([0,0],1), allWin=calculatePairStatistics([2,2],1), mixed=calculatePairStatistics([0,1,2],1);
  assert.deepEqual(allLoss.eloDifference,{kind:'negativeInfinity'}); assert.deepEqual(allWin.eloDifference,{kind:'positiveInfinity'});
  assert.equal(mixed.eloDifference?.kind,'finite'); assert.deepEqual(allLoss.scoreConfidenceInterval95,[0,0]); assert.deepEqual(allWin.scoreConfidenceInterval95,[1,1]);

  const terminal={side:1 as const,p1Men:0,p1Kings:0,p2Men:0,p2Kings:2,halfmoveClock:0};
  const noMove=await playTournamentGame(startAt('no-move',terminal),'p',1,tinyEngine('a'),tinyEngine('b'),2,{searchProvider:firstLegal});
  assert.equal(noMove.result,'p2-win'); assert.equal(noMove.reason,'no-legal-move');
  const inactive={side:1 as const,p1Men:0,p1Kings:1,p2Men:0,p2Kings:2,halfmoveClock:16};
  const inactivity=await playTournamentGame(startAt('inactive',inactive),'p',1,tinyEngine('a'),tinyEngine('b'),2,{searchProvider:firstLegal});
  assert.equal(inactivity.result,'draw'); assert.equal(inactivity.reason,'inactivity');
  const maxed=await playTournamentGame(startAt('max',initialPosition()),'p',1,tinyEngine('a'),tinyEngine('b'),1,{searchProvider:firstLegal});
  assert.equal(maxed.result,'unresolved'); assert.equal(maxed.status,'unresolved');
  const failed=await playTournamentGame(startAt('error',initialPosition()),'p',1,tinyEngine('a'),tinyEngine('b'),2,{searchProvider:async()=>{throw new Error('injected')}});
  assert.equal(failed.result,'error'); assert.equal(failed.status,'error');

  const cycleInitial:Position={side:1,p1Men:0,p1Kings:1,p2Men:0,p2Kings:2,halfmoveClock:0}; let cyclePos:Position={...cycleInitial}; const cycleMoves:Move[]=[];
  for(const [from,to] of [[0,4],[1,5],[4,0],[5,1],[0,4],[1,5],[4,0],[5,1]]) { const move=generateMoves(cyclePos).find(m=>m.from===from&&m.to===to)!; cycleMoves.push(move); cyclePos=applyMove(cyclePos,move); }
  const repetition=await playTournamentGame({id:'repetition',initialPosition:cycleInitial,position:cyclePos,openingMoves:cycleMoves},'p',1,tinyEngine('a'),tinyEngine('b'),2,{searchProvider:firstLegal});
  assert.equal(repetition.result,'draw'); assert.equal(repetition.reason,'threefold-repetition');
  const incompleteSuite={version:'fixture',seed:1,generator:'fixture',starts:[startAt('max',initialPosition())]};
  const incomplete=await runTournament(tinyEngine('a'),tinyEngine('b'),incompleteSuite,{maxPlies:1,searchProvider:firstLegal});
  assert.equal(incomplete.summary.completedPairs,0); assert.equal(incomplete.summary.completedGames,0); assert.equal(incomplete.summary.unresolved,2); assert.equal(incomplete.summary.draws,0); assert.equal(incomplete.summary.scoreEligibleBaselineWins,0); assert.equal(incomplete.summary.scoreEligibleCandidateWins,0); assert.equal(incomplete.summary.eloDifference,null);
  console.log(`PASS tournament harness: 2 paired starts, ${first.summary.games} games, symmetric and reproducible`);
}
main().catch(e=>{console.error(e);process.exitCode=1});
