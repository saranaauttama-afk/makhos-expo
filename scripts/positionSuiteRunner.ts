import { execSync } from 'child_process';
import { mkdirSync, writeFileSync } from 'fs';
import { Move, generateMoves } from '../src/coreClaude/movegen';
import { isDrawByInactivity } from '../src/coreClaude/position';
import { fixedDepthSearch, fixedNodeSearch } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgameDeterministic } from '../src/coreClaude/search/endgameTablebase';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { acceptedMoves, CANONICAL_POSITION_NODE_BUDGET, DIAGNOSTIC_POSITION_DEPTH,
  isVerified, moveMatches, POSITION_SUITE, POSITION_SUITE_V1_FINGERPRINT,
  POSITION_SUITE_VERSION, PositionCase } from './positionSuite';

export type WdlEvidence = { outcome: 'win' | 'draw' | 'loss'; source:
  'rule-repetition' | 'rule-inactivity' | 'rule-elimination' | 'rule-no-legal-move' | 'exact-endgame-oracle' };
export interface PositionRunRow {
  id: string; split: string; validation: string; motifs: string[];
  chosenMove?: Move; chosenMoveSignature?: string;
  score: number; nodes: number; qnodes: number; depth: number; elapsedMs: number; nps: number;
  pass?: boolean; wdlEvidence?: WdlEvidence;
}
export interface PositionSuiteReport {
  generatedAt: string; commit: string; suiteVersion: string; suiteFingerprint: string;
  search: { mode: 'nodes' | 'depth'; budget: number }; revealHoldout: boolean;
  developmentVerified: { correct: number; total: number };
  holdoutVerified: { correct: number; total: number };
  diagnostic: { legacyUnverified: number; provisional: number };
  byMotif: Record<string, { correct: number; total: number }>;
  rows: PositionRunRow[];
}

/** Full artifact identity: from/to alone is insufficient for Thai capture paths. */
export function fullMoveSignature(move: Move): string {
  return JSON.stringify({ from: move.from, to: move.to, captured: [...move.captured],
    path: [...(move.path ?? [])], promote: move.promote });
}

/** Ordinary searchScore is intentionally ignored: its sign is not W/D/L proof. */
export function resolveWdlEvidence(c: PositionCase, _ordinarySearchScore?: number): WdlEvidence | undefined {
  if (c.expected.type !== 'wdl') return undefined;
  const hash = hashPosition(c.position);
  const repetitions = buildRepetitionCounts(c.historyHashes.length ? c.historyHashes : [hash]);
  if (isThreefoldRepetition(repetitions, hash)) return { outcome: 'draw', source: 'rule-repetition' };
  if (isDrawByInactivity(c.position)) return { outcome: 'draw', source: 'rule-inactivity' };
  const own = c.position.side === 1 ? c.position.p1Men | c.position.p1Kings : c.position.p2Men | c.position.p2Kings;
  const opponent = c.position.side === 1 ? c.position.p2Men | c.position.p2Kings : c.position.p1Men | c.position.p1Kings;
  if (own === 0) return { outcome: 'loss', source: 'rule-elimination' };
  if (opponent === 0) return { outcome: 'win', source: 'rule-elimination' };
  if (generateMoves(c.position).length === 0) return { outcome: 'loss', source: 'rule-no-legal-move' };
  const exact = probeSmallEndgameDeterministic(c.position, c.historyHashes, 100_000);
  if (!exact.limitReached && exact.probe?.exact) return {
    outcome: exact.probe.score > 0 ? 'win' : exact.probe.score < 0 ? 'loss' : 'draw',
    source: 'exact-endgame-oracle',
  };
  return undefined;
}

export async function runPositionSuite(mode: 'nodes' | 'depth' = 'nodes', budget = CANONICAL_POSITION_NODE_BUDGET) {
  const rows: PositionRunRow[] = [];
  for (const c of POSITION_SUITE) {
    const result = mode === 'nodes'
      ? await fixedNodeSearch(c.position, budget, new TT(), c.historyHashes)
      : await fixedDepthSearch(c.position, budget, new TT(), c.historyHashes);
    const acceptable = acceptedMoves(c.expected);
    const wdlEvidence = resolveWdlEvidence(c, result.score);
    const pass = !isVerified(c) ? undefined : c.expected.type === 'wdl'
      ? (wdlEvidence ? wdlEvidence.outcome === c.expected.outcome : undefined)
      : (result.best !== undefined && acceptable.some(move => moveMatches(result.best!, move)));
    const totalNodes = result.nodes + result.qnodes;
    rows.push({ id: c.id, split: c.split, validation: c.validation, motifs: c.motifs,
      chosenMove: result.best ? { ...result.best, captured: [...result.best.captured],
        path: result.best.path ? [...result.best.path] : undefined } : undefined,
      chosenMoveSignature: result.best ? fullMoveSignature(result.best) : undefined,
      score: result.score, nodes: result.nodes, qnodes: result.qnodes, depth: result.depth,
      elapsedMs: result.elapsedMs, nps: result.elapsedMs ? Math.round(totalNodes * 1000 / result.elapsedMs) : 0,
      pass, wdlEvidence });
  }
  return rows;
}
function ratio(rows: PositionRunRow[]) {
  const scored = rows.filter(row => row.pass !== undefined);
  return { correct: scored.filter(row => row.pass).length, total: scored.length };
}
export function createPositionSuiteReport(rows: PositionRunRow[], mode: 'nodes' | 'depth', budget: number,
  revealHoldout = false, commit = 'test'): PositionSuiteReport {
  const dev = ratio(rows.filter(r => r.split === 'development'));
  const holdout = ratio(rows.filter(r => r.split === 'holdout'));
  // In tuning-blind mode even aggregate motif buckets must exclude holdout:
  // a singleton motif would otherwise disclose its case verdict indirectly.
  const motifRows = revealHoldout ? rows : rows.filter(r => r.split !== 'holdout');
  const motifs = Object.fromEntries([...new Set(motifRows.flatMap(r => r.motifs))].sort().map(motif =>
    [motif, ratio(motifRows.filter(r => r.motifs.includes(motif) && r.pass !== undefined))]));
  return { generatedAt: new Date().toISOString(), commit, suiteVersion: POSITION_SUITE_VERSION,
    suiteFingerprint: POSITION_SUITE_V1_FINGERPRINT, search: { mode, budget }, revealHoldout,
    developmentVerified: dev, holdoutVerified: holdout,
    diagnostic: { legacyUnverified: rows.filter(r => r.validation === 'legacy-unverified').length,
      provisional: rows.filter(r => r.validation === 'provisional-engine').length }, byMotif: motifs,
    // Default tuning artifacts contain no holdout case rows at all. Aggregate accuracy remains visible.
    rows: revealHoldout ? rows : rows.filter(r => r.split !== 'holdout') };
}
function csvEscape(value: unknown): string {
  const text = typeof value === 'string' ? value : String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
function csv(rows: PositionRunRow[]): string {
  const head = 'id,split,validation,motifs,chosenMove,score,nodes,qnodes,depth,elapsedMs,nps,pass,wdlEvidence';
  return [head, ...rows.map(r => [r.id, r.split, r.validation, r.motifs.join('|'), r.chosenMoveSignature,
    r.score, r.nodes, r.qnodes, r.depth, r.elapsedMs, r.nps, r.pass ?? '', r.wdlEvidence?.source ?? '']
    .map(csvEscape).join(','))].join('\n') + '\n';
}
async function main() {
  const depthArg = process.argv.find(a => a.startsWith('--depth='));
  const nodesArg = process.argv.find(a => a.startsWith('--nodes='));
  const mode = depthArg ? 'depth' : 'nodes';
  const budget = Number((depthArg ?? nodesArg)?.split('=')[1] ??
    (mode === 'nodes' ? CANONICAL_POSITION_NODE_BUDGET : DIAGNOSTIC_POSITION_DEPTH));
  if (!Number.isInteger(budget) || budget < 1) throw new Error('budget must be a positive integer');
  const revealHoldout = process.argv.includes('--reveal-holdout');
  const rows = await runPositionSuite(mode, budget);
  const report = createPositionSuiteReport(rows, mode, budget, revealHoldout,
    execSync('git rev-parse HEAD').toString().trim());
  mkdirSync('.tmp/position-suite', { recursive: true });
  writeFileSync('.tmp/position-suite/result.json', JSON.stringify(report, null, 2));
  writeFileSync('.tmp/position-suite/results.csv', csv(report.rows));
  console.log(`Position suite ${POSITION_SUITE_VERSION}; ${mode}=${budget}; revealHoldout=${revealHoldout}`);
  console.log(`development verified accuracy: ${report.developmentVerified.correct}/${report.developmentVerified.total}`);
  console.log(`holdout verified accuracy: ${report.holdoutVerified.correct}/${report.holdoutVerified.total}`);
  console.log(`legacy/unverified diagnostics: ${report.diagnostic.legacyUnverified}; provisional: ${report.diagnostic.provisional}`);
  console.log('by motif:', JSON.stringify(report.byMotif));
  console.log('Artifacts: .tmp/position-suite/result.json, .tmp/position-suite/results.csv');
}
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1; });
