import { B1, bitCount, toIndex, toRC } from '../src/coreClaude/bitboards';
import { evaluate, evaluateWithBreakdown, type EvalBreakdown } from '../src/coreClaude/eval';
import { Position } from '../src/coreClaude/position';
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

function inspectCase(id: string, pos: Position): void {
  const score = evaluate(pos);
  const breakdown = evaluateWithBreakdown(pos);
  const breakdownSum = sumBreakdownTerms(breakdown);
  const mirrored = mirroredEquivalent(pos);
  const mirrorScore = evaluate(mirrored);
  const mirrorBreakdown = evaluateWithBreakdown(mirrored);
  const sameScore = score === breakdown.finalScore && score === breakdownSum;
  const sameMirror = score === mirrorScore && breakdown.finalScore === mirrorBreakdown.finalScore;
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);

  console.log(`\n[${id}] side=${pos.side} pieces=${totalPieces}`);
  console.log(`score=${score} sum=${breakdownSum} consistency=${sameScore ? 'PASS' : 'FAIL'}`);
  console.log(compactBreakdown(breakdown));
  console.log(`mirrorScore=${mirrorScore} mirrorFinal=${mirrorBreakdown.finalScore} symmetry=${sameMirror ? 'PASS' : 'WARN'}`);
}

function main(): void {
  console.log('Eval breakdown debug');
  for (const entry of getEndgameWeaknessFixtures(CASE_IDS)) inspectCase(entry.id, entry.pos);
}

main();
