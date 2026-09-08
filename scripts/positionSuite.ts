import { B1 } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { TACTICAL_PUZZLES } from './puzzleFixtures';
import { TOURNAMENT_START_SUITE } from './tournamentStartSuite';

export const POSITION_CASE_SCHEMA_VERSION = 'makhos-position-case-schema-v1' as const;
export const POSITION_SUITE_VERSION = 'makhos-position-suite-v1' as const;
export const CANONICAL_POSITION_NODE_BUDGET = 5_000;
export const DIAGNOSTIC_POSITION_DEPTH = 4;

export type PositionSplit = 'development' | 'holdout' | 'diagnostic';
export type ValidationStatus = 'verified-exact' | 'verified-provenance' | 'provisional-engine' | 'legacy-unverified';
export type Motif = 'forced capture' | 'multi-capture' | 'sacrifice' | 'promotion race' |
  'king technique' | 'tempo' | 'defense' | 'trap' | 'draw/repetition' |
  'endgame conversion' | 'low mobility';
export type ExpectedResult =
  | { type: 'acceptable-moves'; moves: ExpectedMove[]; unique: boolean; expectedPv?: ExpectedMove[] }
  | { type: 'wdl'; outcome: 'win' | 'draw' | 'loss'; expectedPv?: ExpectedMove[] }
  | { type: 'forced-legal-only'; move: ExpectedMove; expectedPv?: ExpectedMove[] }
  | { type: 'unlabeled' };
export interface ExpectedMove { from: number; to: number; captured?: number[] }
export interface Provenance {
  kind: 'rule-derived' | 'deterministic-trace' | 'legacy-handcrafted' | 'external';
  reference: string;
  evidence: string;
}
export interface PositionCase {
  id: string;
  schemaVersion: typeof POSITION_CASE_SCHEMA_VERSION;
  suiteVersion: typeof POSITION_SUITE_VERSION;
  position: Position;
  /** Complete board hashes up to and including the root. */
  historyHashes: number[];
  openingMoves?: Move[];
  motifs: Motif[];
  difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
  split: PositionSplit;
  provenance: Provenance;
  validation: ValidationStatus;
  expected: ExpectedResult;
  audit?: { positionLegal: boolean; expectedMoveLegal: boolean; uniqueProven: false; bestMoveEvidence: false; oldMaxCaptureImpact: string };
}

const position = (fields: Partial<Position> & Pick<Position, 'side'>): Position => ({
  p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0, ...fields,
});
const withRootHistory = (p: Position): number[] => [hashPosition(p)];
const ruleEvidence = (description: string): Provenance => ({
  kind: 'rule-derived', reference: 'docs/THAI_RULES_SPEC.md', evidence: description,
});

const exactCases: PositionCase[] = [
  (() => { const p = position({ side: 1, p1Kings: B1(18), p2Men: B1(14) }); return {
    id: 'exact-dev-forced-king-capture-p1', schemaVersion: POSITION_CASE_SCHEMA_VERSION,
    suiteVersion: POSITION_SUITE_VERSION, position: p, historyHashes: withRootHistory(p),
    motifs: ['forced capture', 'king technique', 'endgame conversion'], split: 'development' as const,
    provenance: ruleEvidence('The sole legal move captures the last opposing piece.'), validation: 'verified-exact' as const,
    expected: { type: 'forced-legal-only' as const, move: { from: 18, to: 9, captured: [14] } },
  }; })(),
  (() => { const p = position({ side: 1, p1Kings: B1(0) | B1(1), p2Men: B1(5) }); return {
    id: 'exact-dev-two-equivalent-captures', schemaVersion: POSITION_CASE_SCHEMA_VERSION,
    suiteVersion: POSITION_SUITE_VERSION, position: p, historyHashes: withRootHistory(p),
    motifs: ['forced capture', 'king technique'], split: 'development' as const,
    provenance: ruleEvidence('Both legal captures immediately remove the last opposing piece.'), validation: 'verified-exact' as const,
    expected: { type: 'acceptable-moves' as const, unique: false, moves: [
      { from: 0, to: 9, captured: [5] }, { from: 1, to: 8, captured: [5] },
    ] },
  }; })(),
  (() => { const p = position({ side: -1, p2Kings: B1(13), p1Men: B1(17) }); return {
    id: 'exact-holdout-forced-king-capture-p2', schemaVersion: POSITION_CASE_SCHEMA_VERSION,
    suiteVersion: POSITION_SUITE_VERSION, position: p, historyHashes: withRootHistory(p),
    motifs: ['forced capture', 'king technique', 'endgame conversion'], split: 'holdout' as const,
    provenance: ruleEvidence('The sole legal move captures the last opposing piece.'), validation: 'verified-exact' as const,
    expected: { type: 'forced-legal-only' as const, move: { from: 13, to: 22, captured: [17] } },
  }; })(),
  (() => { const p = position({ side: 1, p1Kings: B1(0), p2Men: B1(5) }); return {
    id: 'exact-holdout-forced-edge-capture', schemaVersion: POSITION_CASE_SCHEMA_VERSION,
    suiteVersion: POSITION_SUITE_VERSION, position: p, historyHashes: withRootHistory(p),
    motifs: ['forced capture', 'low mobility', 'endgame conversion'], split: 'holdout' as const,
    provenance: ruleEvidence('The sole legal move captures the last opposing piece.'), validation: 'verified-exact' as const,
    expected: { type: 'forced-legal-only' as const, move: { from: 0, to: 9, captured: [5] } },
  }; })(),
  (() => { const p = position({ side: 1, p1Kings: B1(18), p2Men: B1(14) }); const h = hashPosition(p); return {
    id: 'exact-diagnostic-threefold-root', schemaVersion: POSITION_CASE_SCHEMA_VERSION,
    suiteVersion: POSITION_SUITE_VERSION, position: p, historyHashes: [h, h, h],
    motifs: ['draw/repetition'], split: 'diagnostic' as const,
    provenance: ruleEvidence('Three occurrences in explicit root history are a draw under project policy.'), validation: 'verified-exact' as const,
    expected: { type: 'wdl' as const, outcome: 'draw' as const },
  }; })(),
];

function positionStructurallyLegal(p: Position): boolean {
  const boards = [p.p1Men, p.p1Kings, p.p2Men, p.p2Kings].map(value => value >>> 0);
  return (p.side === 1 || p.side === -1) && Number.isInteger(p.halfmoveClock) && p.halfmoveClock >= 0 &&
    boards.every((value, i) => boards.every((other, j) => i === j || (value & other) === 0));
}
function moveMatches(move: Move, expected: ExpectedMove): boolean {
  return move.from === expected.from && move.to === expected.to &&
    (expected.captured === undefined || expected.captured.join(',') === move.captured.join(','));
}
function legacyMotifs(type: string): Motif[] {
  if (type.includes('promotion')) return ['promotion race'];
  if (type.includes('sacrifice')) return ['sacrifice'];
  if (type.includes('trap')) return ['trap'];
  if (type.includes('escape')) return ['defense', 'low mobility'];
  if (type.includes('tempo')) return ['tempo'];
  if (type.includes('king') || type.includes('endgame')) return ['king technique'];
  return [];
}

export const LEGACY_PUZZLE_AUDIT: PositionCase[] = TACTICAL_PUZZLES.map(puzzle => {
  const legal = generateMoves(puzzle.pos);
  const expectedMoveLegal = puzzle.expectedMove !== undefined && legal.some(move => moveMatches(move, puzzle.expectedMove!));
  const captureLengths = legal.map(move => move.captured.length);
  return {
    id: `legacy-${puzzle.id}`, schemaVersion: POSITION_CASE_SCHEMA_VERSION, suiteVersion: POSITION_SUITE_VERSION,
    position: puzzle.pos, historyHashes: withRootHistory(puzzle.pos), motifs: legacyMotifs(puzzle.type),
    difficulty: puzzle.difficulty, split: 'diagnostic',
    provenance: { kind: 'legacy-handcrafted', reference: 'scripts/puzzleFixtures.ts',
      evidence: 'No authoritative position-and-solution provenance is recorded.' },
    validation: 'legacy-unverified', expected: puzzle.expectedMove
      ? { type: 'acceptable-moves', unique: false, moves: [puzzle.expectedMove] }
      : { type: 'unlabeled' },
    audit: { positionLegal: positionStructurallyLegal(puzzle.pos), expectedMoveLegal,
      uniqueProven: false, bestMoveEvidence: false,
      oldMaxCaptureImpact: captureLengths.length && Math.max(...captureLengths) !== Math.min(...captureLengths)
        ? 'multiple capture lengths exist; the removed maximum-capture filter could change the old legal set'
        : 'no differing capture lengths at the audited root; no uniqueness/bestness inference made' },
  };
});

function strategicCases(): PositionCase[] {
  return TOURNAMENT_START_SUITE.starts.slice(0, 4).map(start => {
    let p = start.initialPosition ?? initialPosition();
    const history = [hashPosition(p)];
    for (const move of start.openingMoves) { p = applyMove(p, move); history.push(hashPosition(p)); }
    return {
      id: `strategic-${start.id}`, schemaVersion: POSITION_CASE_SCHEMA_VERSION, suiteVersion: POSITION_SUITE_VERSION,
      position: start.position, historyHashes: history, openingMoves: start.openingMoves,
      motifs: [], split: 'diagnostic', validation: 'provisional-engine', expected: { type: 'unlabeled' },
      provenance: { kind: 'deterministic-trace', reference: TOURNAMENT_START_SUITE.version,
        evidence: `${TOURNAMENT_START_SUITE.generator}; legal trace only, with no objective move label.` },
    };
  });
}

export const POSITION_SUITE: readonly PositionCase[] = Object.freeze([
  ...exactCases, ...LEGACY_PUZZLE_AUDIT, ...strategicCases(),
]);

export function canonicalStateKey(c: PositionCase): string {
  const p = c.position;
  return [p.side, p.p1Men >>> 0, p.p1Kings >>> 0, p.p2Men >>> 0, p.p2Kings >>> 0,
    p.halfmoveClock, c.historyHashes.join('.')].join(':');
}
export function isVerified(c: PositionCase): boolean {
  return c.validation === 'verified-exact' || c.validation === 'verified-provenance';
}
export function acceptedMoves(expected: ExpectedResult): ExpectedMove[] {
  if (expected.type === 'forced-legal-only') return [expected.move];
  if (expected.type === 'acceptable-moves') return expected.moves;
  return [];
}
export { moveMatches, positionStructurallyLegal };
