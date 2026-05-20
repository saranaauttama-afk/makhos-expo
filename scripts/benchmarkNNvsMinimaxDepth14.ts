#!/usr/bin/env tsx
/**
 * benchmarkNNvsMinimaxDepth14.ts
 *
 * Benchmark NN model vs Minimax search with FIXED depth 14
 * Test to see if deeper Minimax can compete with NN
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
  totalMinimaxTimeMs: number;
}

const MINIMAX_DEPTH = 14;          // Fixed depth 14 (deeper than default 12)
const MINIMAX_TIME_LIMIT = 30000;  // 30 sec timeout per move (safety)

/**
 * Play one game: NN vs Minimax (depth 14)
 */
async function playGame(nnFirst: boolean, maxPlies: number = 200): Promise<GameResult> {
  let pos = initialPosition();
  const history: Position[] = [pos];
  let moves = 0;
  let totalMinimaxTimeMs = 0;

  while (moves < maxPlies) {
    // Check draw conditions
    if (isDrawByInactivity(pos)) {
      return { winner: 'Draw', moves, reason: 'inactivity', totalMinimaxTimeMs };
    }

    const hash = hashPosition(pos);
    if (isThreefoldRepetition(buildRepetitionCounts(history.map(hashPosition)), hash)) {
      return { winner: 'Draw', moves, reason: 'repetition', totalMinimaxTimeMs };
    }

    // Generate moves
    const legalMoves = generateMoves(pos);
    if (legalMoves.length === 0) {
      // No moves = lose
      const winner = pos.side === 1 ? 'Minimax' : 'NN';
      return {
        winner: nnFirst ? winner : (winner === 'NN' ? 'Minimax' : 'NN'),
        moves,
        reason: 'no_moves',
        totalMinimaxTimeMs
      };
    }

    // Choose move based on player
    let chosenMove: Move;
    const isNNTurn = (pos.side === 1 && nnFirst) || (pos.side === -1 && !nnFirst);

    if (isNNTurn) {
      // NN's turn
      const { policyLogits } = await evaluateNN(pos);
      chosenMove = selectBestNNMove(policyLogits, legalMoves, pos);
    } else {
      // Minimax's turn - FIXED DEPTH 14
      const tt = new TT();
      const hashes = history.map(hashPosition);
      const startTime = Date.now();

      // Force depth 14 by setting maxDepth parameter
      const result = await iterativeDeepening(
        pos,
        MINIMAX_TIME_LIMIT,  // 30s timeout (safety)
        tt,
        undefined,
        hashes,
        undefined,
        MINIMAX_DEPTH        // maxDepth = 14
      );

      const elapsed = Date.now() - startTime;
      totalMinimaxTimeMs += elapsed;

      if (!result.bestMove) {
        // Fallback: pick random legal move if search failed
        chosenMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      } else {
        chosenMove = result.bestMove;
      }

      // Print timing info for this move
      console.log(`    Minimax depth ${result.depth}: ${elapsed}ms (${result.nodes.toLocaleString()} nodes)`);
    }

    // Apply move
    pos = applyMove(pos, chosenMove);
    history.push(pos);
    moves++;
  }

  return { winner: 'Draw', moves, reason: 'max_plies', totalMinimaxTimeMs };
}

/**
 * Run head-to-head benchmark
 */
async function runHeadToHead(games: number = 3): Promise<void> {
  console.log('='.repeat(80));
  console.log('HEAD-TO-HEAD: NN vs MINIMAX DEPTH 14');
  console.log('='.repeat(80));
  console.log(`Playing ${games} test games`);
  console.log(`Minimax: Fixed depth ${MINIMAX_DEPTH}, timeout ${MINIMAX_TIME_LIMIT}ms per move`);
  console.log('');

  let nnWins = 0;
  let minimaxWins = 0;
  let draws = 0;
  let totalGameTimeMs = 0;

  for (let i = 0; i < games; i++) {
    const nnFirst = i % 2 === 0;
    console.log(`Game ${i + 1}/${games}: ${nnFirst ? 'NN (P1) vs Minimax (P2)' : 'Minimax (P1) vs NN (P2)'}`);

    const gameStart = Date.now();
    const result = await playGame(nnFirst);
    const gameTime = Date.now() - gameStart;
    totalGameTimeMs += gameTime;

    console.log(`  Result: ${result.winner} (${result.moves} moves, ${result.reason})`);
    console.log(`  Game time: ${(gameTime / 1000).toFixed(1)}s (Minimax total: ${(result.totalMinimaxTimeMs / 1000).toFixed(1)}s)`);

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
  console.log(`Total time:   ${(totalGameTimeMs / 1000 / 60).toFixed(1)} minutes`);
  console.log('');
}

/**
 * Main
 */
async function main() {
  const args = process.argv.slice(2);
  const games = parseInt(args.find((a) => a.startsWith('--games='))?.split('=')[1] || '3');

  try {
    // Initialize NN
    await initNNInference();

    // Run benchmark
    await runHeadToHead(games);

    // Cleanup
    closeNNInference();
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
