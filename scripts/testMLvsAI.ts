/**
 * Test ML Model vs Traditional AI
 *
 * Run multiple games between ML player and alpha-beta search to evaluate performance
 *
 * Usage:
 *   npx tsx scripts/testMLvsAI.ts
 */

import { Position, initialPosition, isTerminal, Side } from '../src/core/position';
import { Move, generateMoves, applyMove } from '../src/core/movegen';
import { iterativeDeepening } from '../src/core/search/alphabeta';
import { TT } from '../src/core/search/tt';
import { MLPlayer } from '../src/core/ml/mlPlayer';
import * as path from 'path';

interface GameResult {
  winner: 'ML' | 'AI' | 'Draw';
  moves: number;
  reason: string;
}

async function playGame(
  mlPlayer: MLPlayer,
  mlSide: Side,
  aiTimeMs: number = 1000,
  maxMoves: number = 200
): Promise<GameResult> {
  let pos = initialPosition();
  let moves = 0;
  const tt = new TT();

  console.log(`\nGame: ML plays as ${mlSide === 1 ? 'P1 (bottom)' : 'P2 (top)'}`);

  while (moves < maxMoves) {
    // Check terminal
    if (isTerminal(pos)) {
      const legalMoves = generateMoves(pos);
      if (legalMoves.length === 0) {
        const winner = pos.side === mlSide ? 'AI' : 'ML';
        return { winner, moves, reason: 'No legal moves' };
      }
    }

    // Check draw
    if (pos.halfmoveClock >= 40) {
      return { winner: 'Draw', moves, reason: 'Halfmove clock >= 40' };
    }

    // Select move
    let move: Move | undefined;

    if (pos.side === mlSide) {
      // ML player's turn
      console.log(`  Move ${moves + 1}: ML thinking...`);
      move = await mlPlayer.selectMove(pos);
      if (!move) {
        return { winner: 'AI', moves, reason: 'ML has no moves' };
      }
      console.log(`    ML plays: ${move.from} → ${move.to} ${move.captured.length > 0 ? '(capture)' : ''}`);
    } else {
      // AI's turn
      console.log(`  Move ${moves + 1}: AI thinking (${aiTimeMs}ms)...`);
      const result = iterativeDeepening(pos, aiTimeMs, tt);
      move = result.best;
      if (!move) {
        return { winner: 'ML', moves, reason: 'AI has no moves' };
      }
      console.log(`    AI plays: ${move.from} → ${move.to} (depth: ${result.depth}, score: ${result.score})`);
    }

    // Apply move
    pos = applyMove(pos, move);
    moves++;
  }

  return { winner: 'Draw', moves, reason: 'Max moves reached' };
}

async function runTournament(
  mlModelPath: string,
  numGames: number = 10,
  aiTimeMs: number = 1000
): Promise<void> {
  console.log('='.repeat(60));
  console.log('ML vs AI Tournament');
  console.log('='.repeat(60));
  console.log(`Model: ${mlModelPath}`);
  console.log(`Games: ${numGames}`);
  console.log(`AI time: ${aiTimeMs}ms per move`);
  console.log('='.repeat(60));

  // Load ML model
  console.log('\nLoading ML model...');
  const mlPlayer = new MLPlayer({ modelPath: mlModelPath, temperature: 0.1 });
  await mlPlayer.load();
  console.log('✓ ML model loaded\n');

  // Results
  const results: GameResult[] = [];
  let mlWins = 0;
  let aiWins = 0;
  let draws = 0;

  // Play games
  for (let i = 0; i < numGames; i++) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Game ${i + 1}/${numGames}`);
    console.log('='.repeat(60));

    // Alternate sides
    const mlSide: Side = i % 2 === 0 ? 1 : -1;

    const result = await playGame(mlPlayer, mlSide, aiTimeMs);
    results.push(result);

    console.log(`\nResult: ${result.winner} wins (${result.reason}) after ${result.moves} moves`);

    if (result.winner === 'ML') mlWins++;
    else if (result.winner === 'AI') aiWins++;
    else draws++;
  }

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('TOURNAMENT SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total games: ${numGames}`);
  console.log(`ML wins: ${mlWins} (${(mlWins / numGames * 100).toFixed(1)}%)`);
  console.log(`AI wins: ${aiWins} (${(aiWins / numGames * 100).toFixed(1)}%)`);
  console.log(`Draws: ${draws} (${(draws / numGames * 100).toFixed(1)}%)`);
  console.log('='.repeat(60));

  // Print individual results
  console.log('\nDetailed Results:');
  results.forEach((r, i) => {
    console.log(`  Game ${i + 1}: ${r.winner.padEnd(5)} (${r.moves} moves, ${r.reason})`);
  });

  // Conclusion
  console.log('\n' + '='.repeat(60));
  if (mlWins > aiWins) {
    console.log('✓ ML model is STRONGER than traditional AI!');
  } else if (mlWins < aiWins) {
    console.log('✗ ML model is WEAKER than traditional AI');
  } else {
    console.log('= ML model and AI are EQUAL');
  }
  console.log('='.repeat(60));
}

// Main
async function main() {
  const mlModelPath = path.join(__dirname, '..', 'ml', 'best_model.onnx');

  const args = process.argv.slice(2);
  const numGames = args[0] ? parseInt(args[0]) : 10;
  const aiTimeMs = args[1] ? parseInt(args[1]) : 1000;

  await runTournament(mlModelPath, numGames, aiTimeMs);
}

main().catch(console.error);