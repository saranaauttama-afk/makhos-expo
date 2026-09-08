import { strict as assert } from 'node:assert';
import { runTournament } from './tournamentHarness';
import { generateTournamentStartSuite } from './tournamentStartSuite';
import { applyMove } from '../src/coreClaude/movegen';
import { initialPosition } from '../src/coreClaude/position';

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
  console.log(`PASS tournament harness: 2 paired starts, ${first.summary.games} games, symmetric and reproducible`);
}
main().catch(e=>{console.error(e);process.exitCode=1});
