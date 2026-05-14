// Check which opening moves are legal

import { initialPosition } from '../src/coreClaude/position';
import { generateMoves } from '../src/coreClaude/movegen';

const pos = initialPosition();
const legal = generateMoves(pos);

console.log(`Legal moves from initial position (${legal.length} total):`);
legal.forEach((m, i) => {
  console.log(`${i + 1}. ${m.from}->${m.to}`);
});

console.log('\nProposed opening book moves:');
const proposals = [
  { from: 25, to: 22, note: 'Standard opening' },
  { from: 26, to: 23, note: 'Three-in-line' },
  { from: 27, to: 23, note: 'Dragon head' },
  { from: 28, to: 24, note: 'Double corner' },
  { from: 24, to: 20, note: 'Five points' },
  { from: 22, to: 18, note: 'Alternative opening' },
];

proposals.forEach((prop, i) => {
  const isLegal = legal.some(m => m.from === prop.from && m.to === prop.to);
  console.log(`${i + 1}. ${prop.from}->${prop.to} (${prop.note}): ${isLegal ? '✓ LEGAL' : '✗ ILLEGAL'}`);
});
