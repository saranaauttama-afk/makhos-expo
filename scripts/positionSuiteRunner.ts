import { mkdirSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';
import { fixedDepthSearch, fixedNodeSearch, moveKey } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { acceptedMoves, CANONICAL_POSITION_NODE_BUDGET, DIAGNOSTIC_POSITION_DEPTH,
  isVerified, moveMatches, POSITION_SUITE, POSITION_SUITE_VERSION, PositionCase } from './positionSuite';

export interface PositionRunRow {
  id: string; split: string; validation: string; motifs: string[]; chosenMove: string;
  score: number; nodes: number; qnodes: number; depth: number; elapsedMs: number; nps: number;
  pass?: boolean;
}
function outcomePass(c: PositionCase, score: number): boolean | undefined {
  if (c.expected.type !== 'wdl') return undefined;
  return c.expected.outcome === 'draw' ? score === 0 : c.expected.outcome === 'win' ? score > 0 : score < 0;
}
export async function runPositionSuite(mode: 'nodes' | 'depth' = 'nodes', budget = CANONICAL_POSITION_NODE_BUDGET) {
  const rows: PositionRunRow[] = [];
  for (const c of POSITION_SUITE) {
    const result = mode === 'nodes'
      ? await fixedNodeSearch(c.position, budget, new TT(), c.historyHashes)
      : await fixedDepthSearch(c.position, budget, new TT(), c.historyHashes);
    const acceptable = acceptedMoves(c.expected);
    const pass = !isVerified(c) ? undefined : outcomePass(c, result.score) ??
      (result.best !== undefined && acceptable.some(move => moveMatches(result.best!, move)));
    const totalNodes = result.nodes + result.qnodes;
    rows.push({ id: c.id, split: c.split, validation: c.validation, motifs: c.motifs,
      chosenMove: result.best ? moveKey(result.best).toString() : '', score: result.score,
      nodes: result.nodes, qnodes: result.qnodes, depth: result.depth, elapsedMs: result.elapsedMs,
      nps: result.elapsedMs ? Math.round(totalNodes * 1000 / result.elapsedMs) : 0, pass });
  }
  return rows;
}
function ratio(rows: PositionRunRow[]) {
  const scored = rows.filter(row => row.pass !== undefined);
  return { correct: scored.filter(row => row.pass).length, total: scored.length };
}
function csv(rows: PositionRunRow[]): string {
  const head = 'id,split,validation,motifs,chosenMove,score,nodes,qnodes,depth,elapsedMs,nps,pass';
  return [head, ...rows.map(r => [r.id, r.split, r.validation, r.motifs.join('|'), r.chosenMove,
    r.score, r.nodes, r.qnodes, r.depth, r.elapsedMs, r.nps, r.pass ?? ''].join(','))].join('\n') + '\n';
}
async function main() {
  const depthArg = process.argv.find(a => a.startsWith('--depth='));
  const nodesArg = process.argv.find(a => a.startsWith('--nodes='));
  const mode = depthArg ? 'depth' : 'nodes';
  const budget = Number((depthArg ?? nodesArg)?.split('=')[1] ??
    (mode === 'nodes' ? CANONICAL_POSITION_NODE_BUDGET : DIAGNOSTIC_POSITION_DEPTH));
  if (!Number.isInteger(budget) || budget < 1) throw new Error('budget must be a positive integer');
  const rows = await runPositionSuite(mode, budget);
  const dev = ratio(rows.filter(r => r.split === 'development'));
  const holdout = ratio(rows.filter(r => r.split === 'holdout'));
  const motifs = Object.fromEntries([...new Set(rows.flatMap(r => r.motifs))].sort().map(motif =>
    [motif, ratio(rows.filter(r => r.motifs.includes(motif) && r.pass !== undefined))]));
  const report = { generatedAt: new Date().toISOString(), commit: execSync('git rev-parse HEAD').toString().trim(),
    suiteVersion: POSITION_SUITE_VERSION, search: { mode, budget }, developmentVerified: dev,
    holdoutVerified: holdout, diagnostic: { legacyUnverified: rows.filter(r => r.validation === 'legacy-unverified').length,
      provisional: rows.filter(r => r.validation === 'provisional-engine').length }, byMotif: motifs, rows };
  mkdirSync('.tmp/position-suite', { recursive: true });
  writeFileSync('.tmp/position-suite/result.json', JSON.stringify(report, null, 2));
  writeFileSync('.tmp/position-suite/results.csv', csv(rows));
  console.log(`Position suite ${POSITION_SUITE_VERSION}; ${mode}=${budget}`);
  console.log(`development verified accuracy: ${dev.correct}/${dev.total}`);
  console.log(`holdout verified accuracy: ${holdout.correct}/${holdout.total}`);
  console.log(`legacy/unverified diagnostics: ${report.diagnostic.legacyUnverified}; provisional: ${report.diagnostic.provisional}`);
  console.log('by motif:', JSON.stringify(motifs));
  console.log('Artifacts: .tmp/position-suite/result.json, .tmp/position-suite/results.csv');
}
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1; });
