#!/usr/bin/env tsx
/**
 * Analyze small-endgame test case to understand why Expert chooses wrong move
 */

import { B1 } from '../src/coreClaude/bitboards';
import { evaluateWithBreakdown } from '../src/coreClaude/eval';
import { generateMoves, applyMove } from '../src/coreClaude/movegen';
import { Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';

// Small-endgame position from benchmark
const position: Position = {
  side: -1,
  p1Men: B1(25),
  p1Kings: B1(18),
  p2Men: B1(6),
  p2Kings: B1(10),
  halfmoveClock: 0,
};

function formatMove(from: number, to: number): string {
  return `${from}->${to}`;
}

async function main() {
  console.log('='.repeat(80));
  console.log('SMALL-ENDGAME ANALYSIS');
  console.log('='.repeat(80));
  console.log('');
  console.log('Position:');
  console.log('  P1: King@18, Man@25');
  console.log('  P2: King@10, Man@6 (to move)');
  console.log('');
  console.log('Oracle expects: 11->8 (square 10->7 in 0-based)');
  console.log('Expert chooses: 7->10 (square 6->9 in 0-based)');
  console.log('Drop: 121cp');
  console.log('');

  // Generate all legal moves
  const moves = generateMoves(position);
  console.log(`Legal moves: ${moves.length}`);
  console.log('');

  // Evaluate each move
  console.log('Move Analysis:');
  console.log('-'.repeat(80));

  const moveScores: Array<{ move: string; score: number; depth: number; nodes: number }> = [];

  for (const move of moves) {
    const newPos = applyMove(position, move);
    const tt = new TT();

    // Search this move
    const result = await iterativeDeepening(
      newPos,
      2000, // 2 second search
      tt,
      undefined,
      [],
      undefined,
      8 // depth limit
    );

    const moveStr = formatMove(move.from, move.to);
    const score = -result.score; // Negate because it's from opponent's view

    moveScores.push({
      move: moveStr,
      score,
      depth: result.depth,
      nodes: result.nodes || 0,
    });

    const marker = moveStr === '10->7' ? ' ← ORACLE' : moveStr === '6->9' ? ' ← EXPERT CHOSE' : '';
    console.log(`  ${moveStr.padEnd(10)} score=${score.toString().padStart(6)}  depth=${result.depth}  nodes=${result.nodes || 0}${marker}`);
  }

  console.log('');
  console.log('-'.repeat(80));

  // Sort by score
  moveScores.sort((a, b) => b.score - a.score);

  console.log('');
  console.log('Best moves (sorted):');
  for (let i = 0; i < Math.min(5, moveScores.length); i++) {
    const m = moveScores[i];
    const marker = m.move === '10->7' ? ' ← ORACLE' : m.move === '6->9' ? ' ← EXPERT' : '';
    console.log(`  ${(i + 1)}. ${m.move.padEnd(10)} score=${m.score}${marker}`);
  }

  console.log('');
  console.log('='.repeat(80));

  // Evaluate position statically
  console.log('');
  console.log('Static Evaluation:');
  const breakdown = evaluateWithBreakdown(position);
  console.log(`  Total: ${breakdown.score}`);
  console.log(`  Material: ${breakdown.material}`);
  console.log(`  PSQT: ${breakdown.psqt}`);
  console.log(`  Mobility: ${breakdown.mobility}`);
  console.log(`  King Endgame: ${breakdown.kingEndgame}`);
  console.log(`  Endgame Factor: ${breakdown.endgameFactor}`);

  console.log('');
  console.log('Analysis:');

  const oracleMove = moveScores.find(m => m.move === '10->7');
  const expertMove = moveScores.find(m => m.move === '6->9');

  if (oracleMove && expertMove) {
    const diff = oracleMove.score - expertMove.score;
    console.log(`  Oracle move (10->7) scores: ${oracleMove.score}`);
    console.log(`  Expert move (6->9) scores: ${expertMove.score}`);
    console.log(`  Difference: ${diff}cp`);
    console.log('');

    if (Math.abs(diff) < 30) {
      console.log('  → Moves are very close in evaluation!');
      console.log('  → This suggests search depth or tie-breaking issue');
    } else {
      console.log('  → Significant evaluation difference');
      console.log('  → Oracle move is clearly better in search');
    }
  }

  console.log('');
}

main().catch(console.error);
