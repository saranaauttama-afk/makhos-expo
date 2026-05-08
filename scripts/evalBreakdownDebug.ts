import { B1, bitCount, toIndex, toRC } from '../src/coreClaude/bitboards';
import { evaluate, evaluateWithBreakdown, type EvalBreakdown } from '../src/coreClaude/eval';
import { generateMoves, Move, applyMove } from '../src/coreClaude/movegen';
import { Position } from '../src/coreClaude/position';
import { iterativeDeepening, moveKey } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { getEndgameWeaknessFixtures, type EndgameWeaknessFixtureId } from './endgameWeaknessFixtures';

function sumBreakdownTerms(b: EvalBreakdown): number {
  return (
    b.material +
    b.psqt +
    b.mobility +
    b.promotionThreat +
    b.hangingPieces +
    b.backRankGuard +
    b.simplification +
    b.kingEndgame +
    b.allKingsEndgame
  ) | 0;
}

function mapBits(bb: number, mapper: (sq: number) => number): number {
  let out = 0;
  for (let sq = 0; sq < 32; sq++) {
    if (bb & B1(sq)) out |= B1(mapper(sq));
  }
  return out >>> 0;
}

function rotateSquare180(sq: number): number {
  const { r, c } = toRC(sq);
  const rotated = toIndex(7 - r, 7 - c);
  if (rotated < 0) throw new Error(`invalid rotated square for ${sq}`);
  return rotated;
}

function mirroredEquivalent(pos: Position): Position {
  return {
    side: (pos.side === 1 ? -1 : 1) as 1 | -1,
    p1Men: mapBits(pos.p2Men, rotateSquare180),
    p1Kings: mapBits(pos.p2Kings, rotateSquare180),
    p2Men: mapBits(pos.p1Men, rotateSquare180),
    p2Kings: mapBits(pos.p1Kings, rotateSquare180),
    halfmoveClock: pos.halfmoveClock,
  };
}

function compactBreakdown(b: EvalBreakdown): string {
  return [
    `mat=${b.material}`,
    `psqt=${b.psqt}`,
    `mob=${b.mobility}`,
    `promo=${b.promotionThreat}`,
    `hang=${b.hangingPieces}`,
    `back=${Math.round(b.backRankGuard)}`,
    `simp=${b.simplification}`,
    `kEg=${b.kingEndgame}`,
    `kkEg=${b.allKingsEndgame}`,
    `eg=${b.endgameFactor.toFixed(2)}`,
    `kv=${b.kingValue}`,
    `final=${b.finalScore}`,
  ].join(' ');
}

const CASE_IDS: EndgameWeaknessFixtureId[] = [
  'quiet-hanging-piece-p1',
  'low-mobility-squeeze',
  'small-endgame',
  'small-piece-king-vs-men',
];
const ORACLE_MS = 1500;
const ORACLE_DEPTH = 11;
const ORACLE_TABLEBASE_MS = 1500;

interface MoveScore {
  move: Move;
  score: number;
}

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
}

function rotateMove180(move: Move): Move {
  return {
    from: rotateSquare180(move.from),
    to: rotateSquare180(move.to),
    captured: move.captured.map(rotateSquare180),
    promote: move.promote,
    path: move.path?.map(rotateSquare180),
  };
}

async function scoreMoveWithOracle(pos: Position, move: Move): Promise<number> {
  const child = applyMove(pos, move);
  const exactEndgame = probeSmallEndgame(child, [hashPosition(pos), hashPosition(child)], ORACLE_TABLEBASE_MS);
  if (exactEndgame) return -exactEndgame.score;
  const result = await iterativeDeepening(
    child,
    ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos), hashPosition(child)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
  return -result.score;
}

async function oracleRoot(pos: Position): Promise<MoveScore> {
  const legal = generateMoves(pos);
  const fallback = legal[0];
  if (!fallback) throw new Error('oracleRoot called on terminal position');
  const exactEndgame = probeSmallEndgame(pos, [hashPosition(pos)], ORACLE_TABLEBASE_MS);
  if (exactEndgame?.best) return { move: exactEndgame.best, score: exactEndgame.score };

  const total = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const quietLowMobility = legal.length <= 3 && legal[0].captured.length === 0 && total <= 8;
  if (quietLowMobility) {
    let best: MoveScore | undefined;
    for (const move of legal) {
      const score = await scoreMoveWithOracle(pos, move);
      if (!best || score > best.score) best = { move, score };
    }
    if (best) return best;
  }

  const result = await iterativeDeepening(
    pos,
    ORACLE_MS,
    new TT(),
    undefined,
    [hashPosition(pos)],
    { cancelled: false },
    ORACLE_DEPTH,
  );
  return { move: result.best ?? fallback, score: result.score };
}

function classifySymmetry(
  score: number,
  mirrorScore: number,
  legalMoves: number,
  mirrorLegalMoves: number,
  kingValue: number,
  mirrorKingValue: number,
): string {
  const scoreDelta = Math.abs(score - mirrorScore);
  if (legalMoves !== mirrorLegalMoves) return 'SUSPICIOUS legal-move mismatch';
  if (kingValue !== mirrorKingValue) return 'SUSPICIOUS king-value mismatch';
  if (scoreDelta === 0) return 'PASS exact';
  if (scoreDelta <= 8) return 'WARN small expected-ish delta';
  if (scoreDelta <= 24) return 'WARN moderate geometric delta';
  return 'WARN large asymmetry gap';
}

async function inspectCase(id: string, pos: Position): Promise<void> {
  const score = evaluate(pos);
  const breakdown = evaluateWithBreakdown(pos);
  const breakdownSum = sumBreakdownTerms(breakdown);
  const legal = generateMoves(pos);
  const mirrored = mirroredEquivalent(pos);
  const mirrorLegal = generateMoves(mirrored);
  const mirrorScore = evaluate(mirrored);
  const mirrorBreakdown = evaluateWithBreakdown(mirrored);
  const sameScore = score === breakdown.finalScore && score === breakdownSum;
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const symmetryLabel = classifySymmetry(
    score,
    mirrorScore,
    legal.length,
    mirrorLegal.length,
    breakdown.kingValue,
    mirrorBreakdown.kingValue,
  );
  const oracle = await oracleRoot(pos);
  const mirrorOracle = await oracleRoot(mirrored);
  const mappedOracleMove = rotateMove180(oracle.move);
  const mappedOracleKey = moveKey(mappedOracleMove);
  const mirrorOracleMatches = mappedOracleKey === moveKey(mirrorOracle.move);
  const oracleScoreDelta = Math.abs(oracle.score - mirrorOracle.score);

  console.log(`\n[${id}] side=${pos.side} pieces=${totalPieces}`);
  console.log(`score=${score} sum=${breakdownSum} consistency=${sameScore ? 'PASS' : 'FAIL'}`);
  console.log(compactBreakdown(breakdown));
  console.log(
    `mirrorScore=${mirrorScore} mirrorFinal=${mirrorBreakdown.finalScore} symmetry=${symmetryLabel}`,
  );
  console.log(
    `legalMoves=${legal.length} mirrorLegalMoves=${mirrorLegal.length} sideToMove=${pos.side}->${mirrored.side} ` +
    `kingValue=${breakdown.kingValue}/${mirrorBreakdown.kingValue}`,
  );
  console.log(
    `oracle=${fmtMove(oracle.move)} score=${oracle.score} | mirrorOracle=${fmtMove(mirrorOracle.move)} score=${mirrorOracle.score}`,
  );
  console.log(
    `mappedOracle=${fmtMove(mappedOracleMove)} oracleMirrorMatch=${mirrorOracleMatches ? 'PASS' : 'WARN'} ` +
    `oracleScoreDelta=${oracleScoreDelta}`,
  );
}

async function main(): Promise<void> {
  console.log('Eval breakdown debug');
  for (const entry of getEndgameWeaknessFixtures(CASE_IDS)) {
    await inspectCase(entry.id, entry.pos);
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
