import { execFileSync } from 'node:child_process';
import { cpus, platform, release } from 'node:os';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position, Side } from '../src/coreClaude/position';
import { DEFAULT_SEARCH_FEATURES, fixedDepthSearch, fixedNodeSearch, iterativeDeepening,
  resetSearchHeuristicsForMeasurement, SearchFeatureFlags, SearchResult } from '../src/coreClaude/search/alphabeta';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { TournamentStart, TournamentStartSuite } from './tournamentStartSuite';

export type SearchControl = { mode: 'nodes'; budget: number; maxDepth?: number }
  | { mode: 'depth'; budget: number }
  | { mode: 'time'; budget: number; maxDepth?: number };
export interface EngineConfig { id: string; name: string; search: SearchControl; featureOverrides?: Partial<SearchFeatureFlags> }
export interface MoveMetric { ply: number; engineId: string; side: Side; move: Move; nodes: number; qnodes: number; depth: number; elapsedMs: number; nps: number }
export type GameStatus = 'normal' | 'unresolved' | 'error';
export interface GameRecord { pairId: string; startId: string; gameInPair: 1 | 2; p1EngineId: string; p2EngineId: string; winnerEngineId?: string; status: GameStatus; result: 'p1-win'|'p2-win'|'draw'|'unresolved'|'error'; reason: string; plies: number; moves: Move[]; moveMetrics: MoveMetric[] }
export interface PairRecord { pairId: string; startId: string; games: GameRecord[]; candidateScore?: number }
export interface EngineTotals { wins: number; draws: number; nodes: number; qnodes: number; elapsedMs: number; moves: number; averageDepth: number; nps: number }
export interface TournamentSummary { baselineWins: number; candidateWins: number; draws: number; unresolved: number; errors: number; games: number; completedGames: number; pairedStarts: number; candidateScorePercent: number|null; eloDifference: number|null; confidenceInterval95: [number|null, number|null]; confidenceMethod: string }
export interface TournamentResult { schemaVersion: 'makhos-tournament-result-v1'; metadata: Record<string, unknown>; baseline: EngineConfig; candidate: EngineConfig; starts: TournamentStartSuite['starts']; summary: TournamentSummary; engines: Record<string, EngineTotals>; pairs: PairRecord[] }

function sameMove(a: Move, b: Move): boolean { return a.from === b.from && a.to === b.to && a.promote === b.promote && a.captured.join(',') === b.captured.join(',') && (a.path ?? []).join(',') === (b.path ?? []).join(','); }
async function search(pos: Position, history: number[], config: EngineConfig, tt: TT): Promise<SearchResult> {
  // Search currently uses module-level ordering arrays. Clearing them makes
  // each measured move isolated; only the owning player's TT survives moves.
  resetSearchHeuristicsForMeasurement();
  const f = config.featureOverrides ?? {};
  if (config.search.mode === 'nodes') return fixedNodeSearch(pos, config.search.budget, tt, history, config.search.maxDepth ?? 64, undefined, f);
  if (config.search.mode === 'depth') return fixedDepthSearch(pos, config.search.budget, tt, history, undefined, f);
  return iterativeDeepening(pos, config.search.budget, tt, undefined, history, undefined, config.search.maxDepth ?? 64, undefined, false, {}, f);
}

async function play(start: TournamentStart, pairId: string, gameInPair: 1|2,
  p1: EngineConfig, p2: EngineConfig, maxPlies: number): Promise<GameRecord> {
  let replay = initialPosition(); const history = [hashPosition(replay)];
  for (const openingMove of start.openingMoves) { const legal = generateMoves(replay).find(m => sameMove(m, openingMove)); if (!legal) throw new Error(`${start.id}: illegal recorded opening move`); replay=applyMove(replay,legal); history.push(hashPosition(replay)); }
  if (JSON.stringify(replay)!==JSON.stringify(start.position)) throw new Error(`${start.id}: recorded position does not match opening sequence`);
  let pos = { ...start.position }; const moves: Move[] = []; const moveMetrics: MoveMetric[] = [];
  const runtimes = new Map([[p1.id, new TT()], [p2.id, new TT()]]);
  const finish = (status: GameStatus, result: GameRecord['result'], reason: string, winnerEngineId?: string): GameRecord =>
    ({ pairId, startId:start.id, gameInPair, p1EngineId: p1.id, p2EngineId: p2.id, winnerEngineId, status, result, reason, plies: moves.length, moves, moveMetrics });
  for (let ply = 0; ply < maxPlies; ply++) {
    const legal = generateMoves(pos);
    if (!legal.length) return finish('normal', pos.side === 1 ? 'p2-win' : 'p1-win', 'no-legal-move', pos.side === 1 ? p2.id : p1.id);
    if (isDrawByInactivity(pos)) return finish('normal', 'draw', 'inactivity');
    if (isThreefoldRepetition(buildRepetitionCounts(history), hashPosition(pos))) return finish('normal', 'draw', 'threefold-repetition');
    const engine = pos.side === 1 ? p1 : p2;
    let found: SearchResult;
    try { found = await search(pos, history, engine, runtimes.get(engine.id)!); }
    catch (error) { return finish('error', 'error', `search-exception:${error instanceof Error ? error.message : String(error)}`); }
    if (!found.best) return finish('error', 'error', 'search-returned-no-move');
    const legalMove = legal.find(m => sameMove(m, found.best!));
    if (!legalMove) return finish('error', 'error', 'search-returned-illegal-move');
    const totalNodes = found.nodes + found.qnodes;
    moveMetrics.push({ ply, engineId: engine.id, side: pos.side, move: legalMove, nodes: found.nodes, qnodes: found.qnodes,
      depth: found.depth, elapsedMs: found.elapsedMs, nps: found.elapsedMs > 0 ? totalNodes * 1000 / found.elapsedMs : 0 });
    moves.push(legalMove); pos = applyMove(pos, legalMove); history.push(hashPosition(pos));
  }
  return finish('unresolved', 'unresolved', 'maxPlies');
}

function score(game: GameRecord, candidateId: string): number|undefined { if (game.status !== 'normal') return undefined; if (game.result === 'draw') return .5; return game.winnerEngineId === candidateId ? 1 : 0; }
function elo(p: number): number|null { return p <= 0 || p >= 1 ? null : 400 * Math.log10(p / (1 - p)); }
function bootstrap(pairScores: number[], seed: number, samples = 20000): [number|null, number|null] {
  if (!pairScores.length) return [null, null]; let s = seed >>> 0; const values: number[] = [];
  for (let n=0;n<samples;n++) { let total=0; for(let i=0;i<pairScores.length;i++){ s^=s<<13;s^=s>>>17;s^=s<<5; total += pairScores[(s>>>0)%pairScores.length]; } values.push(total / (2*pairScores.length)); }
  values.sort((a,b)=>a-b); return [elo(values[Math.floor(samples*.025)]), elo(values[Math.floor(samples*.975)])];
}
function totals(id: string, games: GameRecord[]): EngineTotals { const ms = games.flatMap(g=>g.moveMetrics).filter(m=>m.engineId===id); const nodes=ms.reduce((s,m)=>s+m.nodes,0), qnodes=ms.reduce((s,m)=>s+m.qnodes,0), elapsedMs=ms.reduce((s,m)=>s+m.elapsedMs,0); return { wins: games.filter(g=>g.winnerEngineId===id).length, draws: games.filter(g=>g.result==='draw').length, nodes,qnodes,elapsedMs,moves:ms.length,averageDepth:ms.length?ms.reduce((s,m)=>s+m.depth,0)/ms.length:0,nps:elapsedMs?(nodes+qnodes)*1000/elapsedMs:0 }; }

export async function runTournament(baseline: EngineConfig, candidate: EngineConfig, suite: TournamentStartSuite,
  options: { maxPlies?: number; startLimit?: number; generatedAt?: string } = {}): Promise<TournamentResult> {
  if (baseline.id === candidate.id) throw new Error('engine ids must be distinct');
  const starts=suite.starts.slice(0, options.startLimit); const pairs: PairRecord[]=[];
  for (const [i,start] of starts.entries()) { const pairId=`pair-${i+1}`; const games=[await play(start,pairId,1,baseline,candidate,options.maxPlies??240), await play(start,pairId,2,candidate,baseline,options.maxPlies??240)]; const ss=games.map(g=>score(g,candidate.id)); pairs.push({pairId,startId:start.id,games,candidateScore:ss.every(x=>x!==undefined)?(ss[0]!+ss[1]!):undefined}); }
  const games=pairs.flatMap(p=>p.games), pairScores=pairs.map(p=>p.candidateScore).filter((x):x is number=>x!==undefined), completedGames=pairScores.length*2, p=completedGames?pairScores.reduce((a,b)=>a+b,0)/completedGames:null;
  let gitCommit='unknown'; try { gitCommit=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(); } catch {}
  const summary: TournamentSummary={baselineWins:games.filter(g=>g.winnerEngineId===baseline.id).length,candidateWins:games.filter(g=>g.winnerEngineId===candidate.id).length,draws:games.filter(g=>g.result==='draw').length,unresolved:games.filter(g=>g.status==='unresolved').length,errors:games.filter(g=>g.status==='error').length,games:games.length,completedGames,pairedStarts:starts.length,candidateScorePercent:p===null?null:p*100,eloDifference:p===null?null:elo(p),confidenceInterval95:bootstrap(pairScores,suite.seed),confidenceMethod:'deterministic pair-level bootstrap, 20,000 resamples; incomplete pairs excluded'};
  return {schemaVersion:'makhos-tournament-result-v1',metadata:{generatedAt:options.generatedAt??new Date().toISOString(),gitCommit,rulesSpec:'docs/THAI_RULES_SPEC.md@Phase-1C',startSuiteVersion:suite.version,startSuiteSeed:suite.seed,startSuiteGenerator:suite.generator,nodeVersion:process.version,os:`${platform()} ${release()}`,cpu:cpus()[0]?.model??'unknown',searchProtocol:baseline.search.mode==='nodes'&&candidate.search.mode==='nodes'?'fixed-nodes canonical':'non-canonical/comparability requires matching controls',defaultSearchFeatures:DEFAULT_SEARCH_FEATURES},baseline,candidate,starts,summary,engines:{[baseline.id]:totals(baseline.id,games),[candidate.id]:totals(candidate.id,games)},pairs};
}

export function saveTournament(result: TournamentResult, directory: string): void { mkdirSync(directory,{recursive:true}); writeFileSync(join(directory,'result.json'),JSON.stringify(result,null,2)+'\n'); const esc=(v:unknown)=>`"${String(v??'').replace(/"/g,'""')}"`; const rows=[['pairId','startId','game','p1Engine','p2Engine','result','winnerEngine','status','reason','plies'],...result.pairs.flatMap(p=>p.games.map(g=>[g.pairId,g.startId,g.gameInPair,g.p1EngineId,g.p2EngineId,g.result,g.winnerEngineId??'',g.status,g.reason,g.plies]))]; writeFileSync(join(directory,'games.csv'),rows.map(r=>r.map(esc).join(',')).join('\n')+'\n'); const pairRows=[['pairId','startId','candidatePairScore','game1Result','game2Result'],...result.pairs.map(p=>[p.pairId,p.startId,p.candidateScore??'',p.games[0].result,p.games[1].result])]; writeFileSync(join(directory,'pairs.csv'),pairRows.map(r=>r.map(esc).join(',')).join('\n')+'\n'); }
