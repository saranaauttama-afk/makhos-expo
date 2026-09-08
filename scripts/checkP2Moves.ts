// Check P2 legal moves after P1 opens

import { initialPosition } from '../src/coreClaude/position';
import { generateMoves, applyMove } from '../src/coreClaude/movegen';

const pos = initialPosition();
console.log('Initial position - P1 moves:');
const p1Moves = generateMoves(pos);
console.log(p1Moves.slice(0, 7).map(m => `${m.from}->${m.to}`).join(', '));

// Apply P1: 25->22
const move25_22 = p1Moves.find(m => m.from === 25 && m.to === 22);
if (!move25_22) {
  throw new Error('25->22 not legal!');
}

const afterP1 = applyMove(pos, move25_22);
console.log('\nAfter P1 plays 25->22, P2 legal moves:');
const p2Moves = generateMoves(afterP1);
p2Moves.forEach((m, i) => {
  console.log(`${i+1}. ${m.from}->${m.to}`);
});
