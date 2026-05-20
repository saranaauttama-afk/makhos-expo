#!/usr/bin/env tsx
/**
 * benchmarkNNvsMinimaxSlow.ts
 *
 * Benchmark NN vs Minimax with longer time limit (10 seconds per move)
 * Test to see if giving Minimax more time helps it compete with NN
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
  avgMinimaxDepth: number;
}

const MINIMAX_TIME_LIMIT = 10000;  // 10 seconds per move (2x longer than default)

/**
 * Play one game: NN vs Minimax (10s per move)
 */
async function playGame(nnFirst: boolean, maxPlies: number = 200): Promise<GameResult> {
  let pos = initialPosition();
  const history: Position[] = [pos];
  let moves = 0;
  let totalMinimaxTimeMs = 0;
  let totalMinimaxDepth = 0;
  let minimaxMoves = 0;

  while (moves < maxPlies) {
    // Check draw conditions
    if (isDrawByInactivity(pos)) {
      return {
        winner: 'Draw',
        moves,
        reason: 'inactivity',
        totalMinimaxTimeMs,
        avgMinimaxDepth: minimaxMoves > 0 ? totalMinimaxDepth / minimaxMoves : 0
      };
    }

    const hash = hashPosition(pos);
    if (isThreefoldRepetition(buildRepetitionCounts(history.map(hashPosition)), hash)) {
      return {
        winner: 'Draw',
        moves,
        reason: 'repetition',
        totalMinimaxTimeMs,
        avgMinimaxDepth: minimaxMoves > 0 ? totalMinimaxDepth / minimaxMoves : 0
      };
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
        totalMinimaxTimeMs,
        avgMinimaxDepth: minimaxMoves > 0 ? totalMinimaxDepth / minimaxMoves : 0
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
      // Minimax's turn - 10 second time limit
      const tt = new TT();
      const hashes = history.map(hashPosition);
      const startTime = Date.now();

      const result = await iterativeDeepening(
        pos,
        MINIMAX_TIME_LIMIT,  // 10s per move
        tt,
        undefined,
        hashes
      );

      const elapsed = Date.now() - startTime;
      totalMinimaxTimeMs += elapsed;
      totalMinimaxDepth += result.depth;
      minimaxMoves++;

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

  return {
    winner: 'Draw',
    moves,
    reason: 'max_plies',
    totalMinimaxTimeMs,
    avgMinimaxDepth: minimaxMoves > 0 ? totalMinimaxDepth / minimaxMoves : 0
  };
}

/**
 * Run head-to-head benchmark
 */
async function runHeadToHead(games: number = 4): Promise<void> {
  console.log('='.repeat(80));
  console.log('HEAD-TO-HEAD: NN vs MINIMAX (SLOW)');
  console.log('='.repeat(80));
  console.log(`Playing ${games} test games`);
  console.log(`Minimax: ${MINIMAX_TIME_LIMIT}ms per move (2x default)`);
  console.log('');

  let nnWins = 0;
  let minimaxWins = 0;
  let draws = 0;
  let totalGameTimeMs = 0;
  let totalAvgDepth = 0;

  for (let i = 0; i < games; i++) {
    const nnFirst = i % 2 === 0;
    console.log(`Game ${i + 1}/${games}: ${nnFirst ? 'NN (P1) vs Minimax (P2)' : 'Minimax (P1) vs NN (P2)'}`);

    const gameStart = Date.now();
    const result = await playGame(nnFirst);
    const gameTime = Date.now() - gameStart;
    totalGameTimeMs += gameTime;
    totalAvgDepth += result.avgMinimaxDepth;

    console.log(`  Result: ${result.winner} (${result.moves} moves, ${result.reason})`);
    console.log(`  Game time: ${(gameTime / 1000).toFixed(1)}s (Minimax avg depth: ${result.avgMinimaxDepth.toFixed(1)})`);

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
  console.log(`Avg Minimax depth: ${(totalAvgDepth / games).toFixed(1)}`);
  console.log(`Total time:   ${(totalGameTimeMs / 1000 / 60).toFixed(1)} minutes`);
  console.log('');
  console.log('Comparison with default (5s per move):');
  console.log('  Default: NN 90%, Minimax 10%, avg depth ~12');
  console.log('  Slow:    See results above');
  console.log('');
}

/**
 * Main
 */
async function main() {
  const args = process.argv.slice(2);
  const games = parseInt(args.find((a) => a.startsWith('--games='))?.split('=')[1] || '4');

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
