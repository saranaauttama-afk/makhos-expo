#!/usr/bin/env tsx
/**
 * benchmarkNNvsMinimax.ts
 *
 * Benchmark NN model vs Minimax search
 * Tests head-to-head games and tactical suite
 */

import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { initNNInference, evaluateNN, selectBestNNMove, closeNNInference } from '../src/coreClaude/nnInference.node';
import { TT } from '../src/coreClaude/search/tt';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { hashPosition } from '../src/coreClaude/search/zobrist';

interface GameResult {
  winner: 'NN' | 'Minimax' | 'Draw';
  moves: number;
  reason: string;
}

/**
 * Play one game: NN vs Minimax
 */
async function playGame(nnFirst: boolean, maxPlies: number = 200): Promise<GameResult> {
  let pos = initialPosition();
  const history: Position[] = [pos];
  let moves = 0;

  while (moves < maxPlies) {
    // Check draw conditions
    if (isDrawByInactivity(pos)) {
      return { winner: 'Draw', moves, reason: 'inactivity' };
    }

    const hash = hashPosition(pos);
    if (isThreefoldRepetition(buildRepetitionCounts(history.map(hashPosition)), hash)) {
      return { winner: 'Draw', moves, reason: 'repetition' };
    }

    // Generate moves
    const legalMoves = generateMoves(pos);
    if (legalMoves.length === 0) {
      // No moves = lose
      const winner = pos.side === 1 ? 'Minimax' : 'NN';
      return { winner: nnFirst ? winner : (winner === 'NN' ? 'Minimax' : 'NN'), moves, reason: 'no_moves' };
    }

    // Choose move based on player
    let chosenMove: Move;
    const isNNTurn = (pos.side === 1 && nnFirst) || (pos.side === -1 && !nnFirst);

    if (isNNTurn) {
      // NN's turn
      const { policyLogits } = await evaluateNN(pos);
      chosenMove = selectBestNNMove(policyLogits, legalMoves, pos);
    } else {
      // Minimax's turn
      const tt = new TT(); // Fixed SIZE (1M slots)
      const hashes = history.map(hashPosition);
      // iterativeDeepening(root, timeMs, tt, onInfo, historyHashes, cancel, maxDepth)
      const result = await iterativeDeepening(pos, 5000, tt, undefined, hashes); // 5s time limit

      if (!result.bestMove) {
        // Fallback: pick random legal move if search failed
        chosenMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      } else {
        chosenMove = result.bestMove;
      }
    }

    // Apply move
    pos = applyMove(pos, chosenMove);
    history.push(pos);
    moves++;
  }

  return { winner: 'Draw', moves, reason: 'max_plies' };
}

/**
 * Run head-to-head benchmark
 */
async function runHeadToHead(games: number = 10): Promise<void> {
  console.log('='.repeat(80));
  console.log('HEAD-TO-HEAD: NN vs MINIMAX');
  console.log('='.repeat(80));
  console.log(`Playing ${games} games (${games / 2} as each color)`);
  console.log('');

  let nnWins = 0;
  let minimaxWins = 0;
  let draws = 0;

  for (let i = 0; i < games; i++) {
    const nnFirst = i % 2 === 0;
    console.log(`Game ${i + 1}/${games}: ${nnFirst ? 'NN (P1) vs Minimax (P2)' : 'Minimax (P1) vs NN (P2)'}`);

    const result = await playGame(nnFirst);
    console.log(`  Result: ${result.winner} (${result.moves} moves, ${result.reason})`);

    if (result.winner === 'NN') nnWins++;
    else if (result.winner === 'Minimax') minimaxWins++;
    else draws++;

    console.log('');
  }

  console.log('='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));
  console.log(`NN wins:      ${nnWins} (${((nnWins / games) * 100).toFixed(1)}%)`);
  console.log(`Minimax wins: ${minimaxWins} (${((minimaxWins / games) * 100).toFixed(1)}%)`);
  console.log(`Draws:        ${draws} (${((draws / games) * 100).toFixed(1)}%)`);
  console.log('');
}

/**
 * Test NN move selection speed
 */
async function benchmarkSpeed(): Promise<void> {
  console.log('='.repeat(80));
  console.log('SPEED BENCHMARK');
  console.log('='.repeat(80));

  const pos = initialPosition();
  const moves = generateMoves(pos);
  const iterations = 1000;

  // Warm up
  for (let i = 0; i < 10; i++) {
    await evaluateNN(pos);
  }

  // Benchmark
  const start = Date.now();
  for (let i = 0; i < iterations; i++) {
    await evaluateNN(pos);
  }
  const elapsed = Date.now() - start;

  console.log(`Iterations: ${iterations}`);
  console.log(`Total time: ${elapsed}ms`);
  console.log(`Avg per eval: ${(elapsed / iterations).toFixed(2)}ms`);
  console.log(`Evals/sec: ${((iterations / elapsed) * 1000).toFixed(0)}`);
  console.log('');
}

/**
 * Main
 */
async function main() {
  const args = process.argv.slice(2);
  const speedOnly = args.includes('--speed');
  const games = parseInt(args.find((a) => a.startsWith('--games='))?.split('=')[1] || '10');
  const modelPath = args.find((a) => a.startsWith('--model='))?.split('=')[1];

  try {
    // Initialize NN
    await initNNInference(modelPath);

    // Run benchmarks
    await benchmarkSpeed();

    if (!speedOnly) {
      await runHeadToHead(games);
    }

    // Cleanup
    closeNNInference();
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
