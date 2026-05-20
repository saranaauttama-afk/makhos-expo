#!/usr/bin/env tsx
/**
 * Test NN vs Minimax on puzzle solving
 */

import { initNNInference, evaluateNN, selectBestNNMove, closeNNInference } from '../src/coreClaude/nnInference.node';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { generateMoves } from '../src/coreClaude/movegen';
import { TACTICAL_PUZZLES, PuzzleFixture } from './puzzleFixtures';

// Take first 3 easy puzzles to test
const testPuzzles = TACTICAL_PUZZLES.filter(p => p.difficulty === 'easy').slice(0, 3);

interface TestResult {
  puzzleName: string;
  nnMove?: { from: number; to: number };
  nnValue: number;
  nnTimeMs: number;
  minimaxMove?: { from: number; to: number };
  minimaxValue: number;
  minimaxTimeMs: number;
  movesMatch: boolean;
}

async function testPuzzle(puzzle: PuzzleFixture): Promise<TestResult> {
  const legalMoves = generateMoves(puzzle.pos);

  if (legalMoves.length === 0) {
    return {
      puzzleName: puzzle.name,
      nnMove: undefined,
      nnValue: 0,
      nnTimeMs: 0,
      minimaxMove: undefined,
      minimaxValue: 0,
      minimaxTimeMs: 0,
      movesMatch: false,
    };
  }

  // Test NN
  const nnStart = Date.now();
  const { value: nnValue, policyLogits } = await evaluateNN(puzzle.pos);
  const nnMove = selectBestNNMove(policyLogits, legalMoves, puzzle.pos);
  const nnTimeMs = Date.now() - nnStart;

  // Test Minimax
  const minimaxStart = Date.now();
  const tt = new TT();
  const minimaxResult = await iterativeDeepening(puzzle.pos, 2000, tt, undefined, []); // 2s limit
  const minimaxTimeMs = Date.now() - minimaxStart;

  const movesMatch = minimaxResult.bestMove
    ? (nnMove.from === minimaxResult.bestMove.from && nnMove.to === minimaxResult.bestMove.to)
    : false;

  const expectedMatch = puzzle.expectedMove
    ? (nnMove.from === puzzle.expectedMove.from && nnMove.to === puzzle.expectedMove.to)
    : false;

  return {
    puzzleName: puzzle.name,
    nnMove: { from: nnMove.from, to: nnMove.to },
    nnValue,
    nnTimeMs,
    minimaxMove: minimaxResult.bestMove ? { from: minimaxResult.bestMove.from, to: minimaxResult.bestMove.to } : undefined,
    minimaxValue: minimaxResult.score,
    minimaxTimeMs,
    movesMatch: expectedMatch || movesMatch, // Match either expected or minimax
  };
}

async function main() {
  console.log('='.repeat(80));
  console.log('NN PUZZLE SOLVING TEST');
  console.log('='.repeat(80));
  console.log('');

  // Initialize NN
  await initNNInference();

  const results: TestResult[] = [];

  for (const puzzle of testPuzzles) {
    console.log(`Testing: ${puzzle.name}`);
    const result = await testPuzzle(puzzle);
    results.push(result);

    console.log(`  NN:      ${result.nnMove ? `${result.nnMove.from}→${result.nnMove.to}` : 'N/A'} (value: ${result.nnValue.toFixed(3)}, ${result.nnTimeMs}ms)`);
    console.log(`  Minimax: ${result.minimaxMove ? `${result.minimaxMove.from}→${result.minimaxMove.to}` : 'N/A'} (value: ${result.minimaxValue}, ${result.minimaxTimeMs}ms)`);
    console.log(`  Match:   ${result.movesMatch ? '✅ YES' : '❌ NO'}`);
    console.log('');
  }

  console.log('='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  const matchCount = results.filter(r => r.movesMatch).length;
  const avgNNTime = results.reduce((sum, r) => sum + r.nnTimeMs, 0) / results.length;
  const avgMinimaxTime = results.reduce((sum, r) => sum + r.minimaxTimeMs, 0) / results.length;

  console.log(`Total puzzles:    ${results.length}`);
  console.log(`Moves matched:    ${matchCount} (${(matchCount / results.length * 100).toFixed(1)}%)`);
  console.log(`Avg NN time:      ${avgNNTime.toFixed(1)}ms`);
  console.log(`Avg Minimax time: ${avgMinimaxTime.toFixed(1)}ms`);
  console.log(`Speed advantage:  ${(avgMinimaxTime / avgNNTime).toFixed(1)}x faster`);

  await closeNNInference();
}

main().catch(console.error);
