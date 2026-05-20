#!/usr/bin/env npx tsx
/**
 * Benchmark v3 (99.32% Val Acc) vs Minimax
 */

import { Position, initialPosition } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move } from '../src/coreClaude/movegen';
import { isDrawByInactivity } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/iterativeDeepening';
import { TT } from '../src/coreClaude/search/tt';
import * as ort from 'onnxruntime-node';
import { extractNNFeatures } from '../src/coreClaude/nnFeatures';

let session: ort.InferenceSession;

async function initNN() {
  console.log('Loading v3 model (99.32% Val Acc)...');
  session = await ort.InferenceSession.create('models/thai_checkers_v3.onnx');
  console.log('✅ v3 loaded');
  console.log('');
}

async function evaluateNN(pos: Position): Promise<{ value: number; policyLogits: Float32Array }> {
  const features = extractNNFeatures(pos);
  const inputTensor = new ort.Tensor('float32', features, [1, 320]);
  const outputs = await session.run({ features: inputTensor });

  const policyLogits = outputs['policy_logits'].data as Float32Array;
  const valueOutput = outputs['value'].data as Float32Array;
  const value = valueOutput[0];

  return { value, policyLogits };
}

function selectBestNNMove(policyLogits: Float32Array, legalMoves: Move[]): Move | undefined {
  if (legalMoves.length === 0) return undefined;

  let bestMove = legalMoves[0];
  let bestLogit = -Infinity;

  for (const move of legalMoves) {
    const moveIdx = move.from * 32 + move.to;
    const logit = policyLogits[moveIdx];
    if (logit > bestLogit) {
      bestLogit = logit;
      bestMove = move;
    }
  }

  return bestMove;
}

async function minimaxMove(pos: Position, timeMs: number, maxDepth: number): Promise<Move | undefined> {
  const tt = new TT();
  const result = await iterativeDeepening(pos, timeMs, tt, undefined, [], undefined, maxDepth);
  return result.move;
}

interface GameResult {
  winner: 'nn' | 'minimax' | 'draw';
  plies: number;
  reason: string;
}

async function playGame(nnPlaysP1: boolean, minimaxTimeMs: number, minimaxMaxDepth: number, maxPlies = 200): Promise<GameResult> {
  let pos = initialPosition();
  let plies = 0;

  while (plies < maxPlies) {
    const moves = generateMoves(pos);

    if (moves.length === 0) {
      const winner = pos.side === 1 ? 'minimax' : 'nn';
      return {
        winner: nnPlaysP1 ? (winner === 'nn' ? 'nn' : 'minimax') : (winner === 'minimax' ? 'nn' : 'minimax'),
        plies,
        reason: 'checkmate'
      };
    }

    if (isDrawByInactivity(pos)) {
      return { winner: 'draw', plies, reason: 'inactivity' };
    }

    const isNNTurn = (pos.side === 1 && nnPlaysP1) || (pos.side === -1 && !nnPlaysP1);

    let move: Move | undefined;
    if (isNNTurn) {
      const { policyLogits } = await evaluateNN(pos);
      move = selectBestNNMove(policyLogits, moves);
    } else {
      move = await minimaxMove(pos, minimaxTimeMs, minimaxMaxDepth);
    }

    if (!move) {
      return { winner: 'draw', plies, reason: 'no_move' };
    }

    pos = applyMove(pos, move);
    plies++;
  }

  return { winner: 'draw', plies, reason: 'max_plies' };
}

async function runBenchmark(games = 20, minimaxTimeMs = 5000, minimaxMaxDepth = 12) {
  await initNN();

  console.log('='.repeat(80));
  console.log('BENCHMARK: v3 (99.32% Val Acc, no augment) vs Minimax');
  console.log('='.repeat(80));
  console.log(`Games: ${games} (${games/2} as P1, ${games/2} as P2)`);
  console.log(`Minimax: ${minimaxTimeMs}ms, max depth ${minimaxMaxDepth}`);
  console.log('');

  let nnWins = 0;
  let minimaxWins = 0;
  let draws = 0;

  const startTime = Date.now();

  for (let i = 0; i < games; i++) {
    const nnPlaysP1 = i % 2 === 0;
    const gameNum = i + 1;

    process.stdout.write(`Game ${gameNum}/${games}: v3=${nnPlaysP1 ? 'P1' : 'P2'}, Minimax=${nnPlaysP1 ? 'P2' : 'P1'} ... `);

    const result = await playGame(nnPlaysP1, minimaxTimeMs, minimaxMaxDepth);

    if (result.winner === 'nn') nnWins++;
    else if (result.winner === 'minimax') minimaxWins++;
    else draws++;

    const resultStr = result.winner === 'draw'
      ? `Draw (${result.reason})`
      : `${result.winner.toUpperCase()} wins (${result.reason})`;

    console.log(`${resultStr} in ${result.plies} plies`);
  }

  const totalTime = (Date.now() - startTime) / 1000;

  console.log('');
  console.log('='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));
  console.log(`v3 (NN) wins: ${nnWins} (${(nnWins/games*100).toFixed(1)}%)`);
  console.log(`Minimax wins: ${minimaxWins} (${(minimaxWins/games*100).toFixed(1)}%)`);
  console.log(`Draws: ${draws} (${(draws/games*100).toFixed(1)}%)`);
  console.log(`Total time: ${totalTime.toFixed(1)}s (${(totalTime/games).toFixed(1)}s per game)`);
  console.log('');

  // Calculate win rate and Elo
  const nnScore = (nnWins + draws * 0.5) / games;
  console.log(`v3 score: ${nnScore.toFixed(3)}`);

  if (nnScore > 0 && nnScore < 1) {
    const eloDiff = Math.round(400 * Math.log10(nnScore / (1 - nnScore)));
    console.log(`Estimated Elo advantage: ${eloDiff > 0 ? '+' : ''}${eloDiff}`);
  }
  console.log('');

  if (nnWins > minimaxWins) {
    console.log('🏆 v3 (99.32% Val Acc) BEATS Minimax! 🏆');
  } else if (minimaxWins > nnWins) {
    console.log('❌ Minimax beats v3');
  } else {
    console.log('🤝 TIE! Equal strength');
  }
}

// Parse args
const args = process.argv.slice(2);
const gamesArg = args.find(arg => arg.startsWith('--games='));
const timeArg = args.find(arg => arg.startsWith('--time='));
const depthArg = args.find(arg => arg.startsWith('--depth='));

const games = gamesArg ? parseInt(gamesArg.split('=')[1]) : 20;
const timeMs = timeArg ? parseInt(timeArg.split('=')[1]) : 5000;
const maxDepth = depthArg ? parseInt(depthArg.split('=')[1]) : 12;

runBenchmark(games, timeMs, maxDepth).catch(console.error);
