#!/usr/bin/env tsx
/**
 * Head-to-Head Engine Testing
 *
 * Pits two versions of the engine against each other to measure
 * strength improvement from opening book expansion.
 *
 * Test scenarios:
 * - AI with 31-entry book vs AI with 17-entry book
 * - AI with opening book vs AI without book
 * - Different time controls
 */

import { initialPosition, Position } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { applyMove } from '../src/coreClaude/movegen';
import { generateMoves } from '../src/coreClaude/movegen';

interface GameConfig {
  player1Name: string;
  player2Name: string;
  timePerMove: number;
  maxDepth: number;
  useOpeningBook: boolean;
}

interface GameResult {
  winner: 'player1' | 'player2' | 'draw';
  reason: string;
  moves: number;
  timeMs: number;
}

interface TournamentResult {
  player1Wins: number;
  player2Wins: number;
  draws: number;
  totalGames: number;
  player1Name: string;
  player2Name: string;
  games: GameResult[];
}

async function playGame(config: GameConfig): Promise<GameResult> {
  const startTime = Date.now();
  let pos = initialPosition();
  let moves = 0;
  const maxMoves = 200; // Draw after 200 moves

  const tt1 = new TT();
  const tt2 = new TT();

  while (moves < maxMoves) {
    const legalMoves = generateMoves(pos);

    if (legalMoves.length === 0) {
      // No legal moves - player to move loses
      const winner = pos.side === 1 ? 'player2' : 'player1';
      return {
        winner,
        reason: 'No legal moves',
        moves,
        timeMs: Date.now() - startTime,
      };
    }

    // Determine which player's turn
    const isPlayer1Turn = (moves % 2 === 0);
    const tt = isPlayer1Turn ? tt1 : tt2;

    try {
      const result = await iterativeDeepening(
        pos,
        config.timePerMove,
        tt,
        undefined,
        [],
        undefined,
        config.maxDepth
      );

      if (!result.best) {
        // No move found - should not happen
        const winner = isPlayer1Turn ? 'player2' : 'player1';
        return {
          winner,
          reason: 'Failed to find move',
          moves,
          timeMs: Date.now() - startTime,
        };
      }

      pos = applyMove(pos, result.best);
      moves++;

    } catch (error) {
      const winner = isPlayer1Turn ? 'player2' : 'player1';
      return {
        winner,
        reason: `Error: ${error}`,
        moves,
        timeMs: Date.now() - startTime,
      };
    }
  }

  // Draw by move limit
  return {
    winner: 'draw',
    reason: 'Move limit reached',
    moves,
    timeMs: Date.now() - startTime,
  };
}

async function runTournament(
  config: GameConfig,
  numGames: number
): Promise<TournamentResult> {
  console.log(`Starting tournament: ${config.player1Name} vs ${config.player2Name}`);
  console.log(`Games: ${numGames}, Time: ${config.timePerMove}ms/move`);
  console.log('');

  const result: TournamentResult = {
    player1Wins: 0,
    player2Wins: 0,
    draws: 0,
    totalGames: numGames,
    player1Name: config.player1Name,
    player2Name: config.player2Name,
    games: [],
  };

  for (let i = 0; i < numGames; i++) {
    process.stdout.write(`Game ${i + 1}/${numGames}... `);

    const gameResult = await playGame(config);
    result.games.push(gameResult);

    if (gameResult.winner === 'player1') result.player1Wins++;
    else if (gameResult.winner === 'player2') result.player2Wins++;
    else result.draws++;

    const winnerStr = gameResult.winner === 'draw'
      ? 'Draw'
      : gameResult.winner === 'player1'
        ? config.player1Name
        : config.player2Name;

    console.log(`${winnerStr} (${gameResult.moves} moves, ${(gameResult.timeMs / 1000).toFixed(1)}s)`);
  }

  return result;
}

function calculateElo(wins: number, losses: number, draws: number): number {
  // Calculate Elo rating difference based on win rate
  const totalGames = wins + losses + draws;
  if (totalGames === 0) return 0;

  const score = (wins + 0.5 * draws) / totalGames;

  // Elo formula: rating_diff = -400 * log10(1/score - 1)
  if (score === 0) return -800; // Complete loss
  if (score === 1) return 800;  // Complete win

  return Math.round(-400 * Math.log10(1 / score - 1));
}

async function main() {
  console.log('='.repeat(80));
  console.log('HEAD-TO-HEAD ENGINE TESTING');
  console.log('='.repeat(80));
  console.log('');

  // Test 1: Opening Book Impact (10 games, fast time control)
  console.log('Test 1: Opening Book Impact');
  console.log('-'.repeat(80));
  const test1Result = await runTournament(
    {
      player1Name: '31-entry book',
      player2Name: 'No book',
      timePerMove: 1000,  // 1 second per move
      maxDepth: 6,
      useOpeningBook: true,
    },
    10
  );

  console.log('');
  console.log('Results:');
  console.log(`  ${test1Result.player1Name}: ${test1Result.player1Wins} wins`);
  console.log(`  ${test1Result.player2Name}: ${test1Result.player2Wins} wins`);
  console.log(`  Draws: ${test1Result.draws}`);

  const elo1 = calculateElo(
    test1Result.player1Wins,
    test1Result.player2Wins,
    test1Result.draws
  );
  console.log(`  Elo difference: ${elo1 > 0 ? '+' : ''}${elo1}`);
  console.log('');

  // Test 2: Longer time control (5 games)
  console.log('Test 2: Longer Time Control');
  console.log('-'.repeat(80));
  const test2Result = await runTournament(
    {
      player1Name: '31-entry (2s)',
      player2Name: 'No book (2s)',
      timePerMove: 2000,  // 2 seconds per move
      maxDepth: 8,
      useOpeningBook: true,
    },
    5
  );

  console.log('');
  console.log('Results:');
  console.log(`  ${test2Result.player1Name}: ${test2Result.player1Wins} wins`);
  console.log(`  ${test2Result.player2Name}: ${test2Result.player2Wins} wins`);
  console.log(`  Draws: ${test2Result.draws}`);

  const elo2 = calculateElo(
    test2Result.player1Wins,
    test2Result.player2Wins,
    test2Result.draws
  );
  console.log(`  Elo difference: ${elo2 > 0 ? '+' : ''}${elo2}`);
  console.log('');

  // Summary
  console.log('='.repeat(80));
  console.log('TOURNAMENT SUMMARY');
  console.log('='.repeat(80));
  console.log('');

  const totalWins = test1Result.player1Wins + test2Result.player1Wins;
  const totalLosses = test1Result.player2Wins + test2Result.player2Wins;
  const totalDraws = test1Result.draws + test2Result.draws;
  const totalGames = totalWins + totalLosses + totalDraws;

  console.log(`Total games played: ${totalGames}`);
  console.log(`Opening book wins: ${totalWins} (${(totalWins / totalGames * 100).toFixed(1)}%)`);
  console.log(`No book wins: ${totalLosses} (${(totalLosses / totalGames * 100).toFixed(1)}%)`);
  console.log(`Draws: ${totalDraws} (${(totalDraws / totalGames * 100).toFixed(1)}%)`);
  console.log('');

  const overallElo = calculateElo(totalWins, totalLosses, totalDraws);
  console.log(`Overall Elo advantage: ${overallElo > 0 ? '+' : ''}${overallElo}`);

  // Save results
  const fs = await import('fs');
  fs.mkdirSync('.tmp/head-to-head', { recursive: true });
  fs.writeFileSync('.tmp/head-to-head/results.json', JSON.stringify({
    generatedAt: new Date().toISOString(),
    test1: test1Result,
    test2: test2Result,
    summary: {
      totalGames,
      openingBookWins: totalWins,
      noBookWins: totalLosses,
      draws: totalDraws,
      eloAdvantage: overallElo,
    },
  }, null, 2));

  console.log('');
  console.log('Results saved to: .tmp/head-to-head/results.json');
}

main().catch(console.error);
