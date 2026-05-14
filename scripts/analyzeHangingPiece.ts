#!/usr/bin/env tsx
/**
 * Analyze quiet-hanging-piece-p1 to understand why Expert hangs a piece
 */

import { B1 } from '../src/coreClaude/bitboards';
import { evaluateWithBreakdown } from '../src/coreClaude/eval';
import { generateMoves, applyMove } from '../src/coreClaude/movegen';
import { Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';

// quiet-hanging-piece-p1 position
const position: Position = {
  side: 1,
  p1Men: B1(21) | B1(25) | B1(30),
  p1Kings: 0,
  p2Men: B1(13) | B1(14) | B1(17),
  p2Kings: 0,
  halfmoveClock: 0,
};

function formatMove(from: number, to: number): string {
  return `${from + 1}->${to + 1}`;
}

async function main() {
  console.log('='.repeat(80));
  console.log('QUIET-HANGING-PIECE-P1 ANALYSIS');
  console.log('='.repeat(80));
  console.log('');
  console.log('Position:');
  console.log('  P1: Men@21, Men@25, Men@30 (to move)');
  console.log('  P2: Men@13, Men@14, Men@17');
  console.log('');
  console.log('Oracle expects: 26->23 (25->22 in 0-based)');
  console.log('Expert chooses: 22->17 (21->16 in 0-based)');
  console.log('Drop: 425cp');
  console.log('');

  // Generate all legal moves
  const moves = generateMoves(position);
  console.log(`Legal moves: ${moves.length}`);
  console.log('');

  // Evaluate each move
  console.log('Move Analysis:');
  console.log('-'.repeat(80));

  const moveScores: Array<{ move: string; score: number; depth: number; nodes: number; hangingPenalty: number }> = [];

  for (const move of moves) {
    const newPos = applyMove(position, move);
    const breakdown = evaluateWithBreakdown(newPos);
    const tt = new TT();

    // Search this move (use 2 seconds for proper analysis)
    const result = await iterativeDeepening(
      newPos,
      2000, // 2 seconds for deeper analysis
      tt,
      undefined,
      [],
      undefined,
      14 // expert depth cap
    );

    const moveStr = formatMove(move.from, move.to);
    const score = -result.score; // Negate because it's from opponent's view

    moveScores.push({
      move: moveStr,
      score,
      depth: result.depth,
      nodes: result.nodes || 0,
      hangingPenalty: breakdown.hangingPieces,
    });

    const marker = moveStr === '26->23' ? ' ← ORACLE' : moveStr === '22->17' ? ' ← EXPERT CHOSE' : '';
    console.log(`  ${moveStr.padEnd(10)} score=${score.toString().padStart(6)}  depth=${result.depth}  hanging=${breakdown.hangingPieces}${marker}`);
  }

  console.log('');
  console.log('-'.repeat(80));

  // Sort by score
  moveScores.sort((a, b) => b.score - a.score);

  console.log('');
  console.log('Best moves (sorted):');
  for (let i = 0; i < Math.min(5, moveScores.length); i++) {
    const m = moveScores[i];
    const marker = m.move === '26->23' ? ' ← ORACLE' : m.move === '22->17' ? ' ← EXPERT' : '';
    console.log(`  ${(i + 1)}. ${m.move.padEnd(10)} score=${m.score} hanging=${m.hangingPenalty}${marker}`);
  }

  console.log('');
  console.log('='.repeat(80));

  // Evaluate position statically
  console.log('');
  console.log('Static Evaluation of Initial Position:');
  const breakdown = evaluateWithBreakdown(position);
  console.log(`  Total: ${breakdown.finalScore}`);
  console.log(`  Material: ${breakdown.material}`);
  console.log(`  PSQT: ${breakdown.psqt}`);
  console.log(`  Mobility: ${breakdown.mobility}`);
  console.log(`  Hanging Pieces: ${breakdown.hangingPieces}`);

  console.log('');
  console.log('Analysis:');

  const oracleMove = moveScores.find(m => m.move === '26->23');
  const expertMove = moveScores.find(m => m.move === '22->17');

  if (oracleMove && expertMove) {
    const diff = oracleMove.score - expertMove.score;
    console.log(`  Oracle move (26->23) scores: ${oracleMove.score} (hanging=${oracleMove.hangingPenalty})`);
    console.log(`  Expert move (22->17) scores: ${expertMove.score} (hanging=${expertMove.hangingPenalty})`);
    console.log(`  Difference: ${diff}cp`);
    console.log('');

    if (Math.abs(diff) < 50) {
      console.log('  → Moves are close in evaluation!');
      console.log('  → Hanging piece penalty may be insufficient');
    } else {
      console.log('  → Significant evaluation difference');
      console.log('  → Oracle move is clearly better');
    }
  }

  console.log('');
}

main().catch(console.error);
