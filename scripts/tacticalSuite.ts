// scripts/tacticalSuite.ts
//
// Focused smoke/regression checks for the pure minimax engine. Run with:
//   npm run test:tactical

import { B1 } from '../src/coreClaude/bitboards';
import { generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';
import { iterativeDeepening, moveKey } from '../src/coreClaude/search/alphabeta';
import { probeSmallEndgame } from '../src/coreClaude/search/endgameTablebase';
import { TT } from '../src/coreClaude/search/tt';

function makePosition(fields: Partial<Position> & Pick<Position, 'side'>): Position {
  return {
    p1Men: 0,
    p1Kings: 0,
    p2Men: 0,
    p2Kings: 0,
    halfmoveClock: 0,
    ...fields,
  };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function moveLabel(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? ` caps=${move.captured.join(',')}` : '';
  const promote = move.promote ? ' promote' : '';
  return `${move.from}->${move.to}${caps}${promote}`;
}

function sameMove(a: Move | undefined, b: Move): boolean {
  return !!a && a.from === b.from && a.to === b.to && a.captured.length === b.captured.length;
}

async function search(pos: Position, maxDepth = 7) {
  return iterativeDeepening(pos, 700, new TT(), undefined, [], { cancelled: false }, maxDepth);
}

async function main() {
  let checks = 0;

  const doubleCapture = makePosition({
    side: 1,
    p1Men: B1(22) | B1(30),
    p2Men: B1(17) | B1(9),
  });
  const doubleCaptureMoves = generateMoves(doubleCapture);
  assert(doubleCaptureMoves.length === 1, `expected one forced double capture, got ${doubleCaptureMoves.map(moveLabel).join(' | ')}`);
  assert(doubleCaptureMoves[0].from === 22 && doubleCaptureMoves[0].to === 6 && doubleCaptureMoves[0].captured.length === 2, 'movegen should keep the longest capture only');
  checks += 2;

  const doubleCaptureSearch = await search(doubleCapture);
  assert(sameMove(doubleCaptureSearch.best, doubleCaptureMoves[0]), `search missed forced double capture: ${moveLabel(doubleCaptureSearch.best)}`);
  checks++;

  const promotion = makePosition({
    side: 1,
    p1Men: B1(5) | B1(24),
    p2Men: B1(31) | B1(27),
  });
  const promotionSearch = await search(promotion);
  assert(promotionSearch.best?.promote, `search should prefer immediate promotion, got ${moveLabel(promotionSearch.best)}`);
  checks++;

  const opening = await iterativeDeepening(initialPosition(), 500, new TT(), undefined, [], { cancelled: false }, 5);
  assert(opening.best, 'opening search should return a legal move');
  assert((opening.rootCandidates?.length ?? 0) >= 2, 'opening search should expose root candidates for book/multi-PV verification');
  assert(opening.rootCandidates?.some(candidate => moveKey(candidate.move) === moveKey(opening.best!)), 'root candidates should include the selected best move');
  checks += 3;

  const endgame = makePosition({
    side: 1,
    p1Kings: B1(18),
    p2Men: B1(14),
  });
  const endgameHit = probeSmallEndgame(endgame, [], 150);
  assert(endgameHit?.best?.captured.length === 1, `tablebase probe should find the king capture, got ${moveLabel(endgameHit?.best)}`);
  checks++;

  console.log(`tacticalSuite: ${checks} checks passed`);
}

main().catch((error: unknown) => {
  console.error(error);
  throw error;
});
