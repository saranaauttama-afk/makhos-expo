// Debug script to analyze forced recapture trap cases

import { B1, bitCount } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move } from '../src/coreClaude/movegen';
import { evaluate } from '../src/coreClaude/eval';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';

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

function formatSquare(sq: number): string {
  const row = Math.floor(sq / 4);
  const col = (sq % 4) * 2 + (row % 2);
  return `${row + 1}${col + 1}`;
}

function formatMove(m: Move): string {
  return `${formatSquare(m.from)}->${formatSquare(m.to)}`;
}

async function analyzePosition(name: string, pos: Position, oracleMove: string) {
  console.log(`\n=== ${name} ===`);
  console.log(`Pieces: ${bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings)}`);
  console.log(`Side: ${pos.side === 1 ? 'P1' : 'P2'}`);

  const legal = generateMoves(pos);
  console.log(`\nLegal moves: ${legal.length}`);

  console.log(`\nStatic eval: ${evaluate(pos)}`);

  // Search at easy depth (1-2)
  console.log(`\n--- Easy level (depth=1) ---`);
  const easy = await iterativeDeepening(pos, 10000, undefined, undefined, [], undefined, 1);
  console.log(`Best move: ${easy.best ? formatMove(easy.best) : 'none'}`);
  console.log(`Score: ${easy.score}`);
  console.log(`Override: ${easy.overrideReason || 'none'}`);

  // Search at normal depth (2-3)
  console.log(`\n--- Normal level (depth=2) ---`);
  const normal = await iterativeDeepening(pos, 10000, undefined, undefined, [], undefined, 2);
  console.log(`Best move: ${normal.best ? formatMove(normal.best) : 'none'}`);
  console.log(`Score: ${normal.score}`);
  console.log(`Override: ${normal.overrideReason || 'none'}`);

  // Search at hard depth (3-4)
  console.log(`\n--- Hard level (depth=3) ---`);
  const hard = await iterativeDeepening(pos, 10000, undefined, undefined, [], undefined, 3);
  console.log(`Best move: ${hard.best ? formatMove(hard.best) : 'none'}`);
  console.log(`Score: ${hard.score}`);
  console.log(`Override: ${hard.overrideReason || 'none'}`);

  console.log(`\nOracle expects: ${oracleMove}`);

  // Analyze each legal move
  console.log(`\n--- Move analysis ---`);
  for (const move of legal.slice(0, 5)) {
    const child = applyMove(pos, move);
    const childEval = evaluate(child);
    const opponentReplies = generateMoves(child);
    const hasCapture = opponentReplies.length > 0 && opponentReplies[0].captured.length > 0;
    console.log(`${formatMove(move)}: childEval=${-childEval}, opponent replies=${opponentReplies.length}, forced capture=${hasCapture}`);
  }
}

// midgame-bait-double-recapture-p1
const pos1 = makePosition({
  side: 1,
  p1Men: B1(21) | B1(26) | B1(27) | B1(30),
  p2Men: B1(13) | B1(5) | B1(6) | B1(9),
});

// opening-bait-double-recapture-p1
const pos2 = makePosition({
  side: 1,
  p1Men: B1(21) | B1(22) | B1(26) | B1(27) | B1(29) | B1(30),
  p2Men: B1(5) | B1(6) | B1(9) | B1(10) | B1(13) | B1(14),
});

(async () => {
  await analyzePosition('midgame-bait-double-recapture-p1', pos1, '27->23');
  await analyzePosition('opening-bait-double-recapture-p1', pos2, '23->19');
})();
