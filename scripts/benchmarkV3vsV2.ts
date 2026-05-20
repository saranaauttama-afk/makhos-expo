#!/usr/bin/env npx tsx
/**
 * Benchmark v3 (99.32% Val Acc, no augment) vs v2 (95% Val Acc)
 * Head-to-head NN battle!
 */

import { Position, initialPosition, clone } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move } from '../src/coreClaude/movegen';
import { isDrawByInactivity } from '../src/coreClaude/position';
import * as ort from 'onnxruntime-node';
import { extractNNFeatures } from '../src/coreClaude/nnFeatures';

// Load both models
let sessionV2: ort.InferenceSession;
let sessionV3: ort.InferenceSession;

async function initModels() {
  console.log('Loading models...');
  sessionV2 = await ort.InferenceSession.create('models/thai_checkers_v2_merged.onnx');
  console.log('✅ v2 loaded (95% Val Acc)');
  sessionV3 = await ort.InferenceSession.create('models/thai_checkers_v3.onnx');
  console.log('✅ v3 loaded (99.32% Val Acc, no augment)');
  console.log('');
}

async function evaluateNN(pos: Position, session: ort.InferenceSession): Promise<{ value: number; policyLogits: Float32Array }> {
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

interface GameResult {
  winner: 'v3' | 'v2' | 'draw';
  plies: number;
  reason: string;
}

async function playGame(v3PlaysP1: boolean, maxPlies = 200): Promise<GameResult> {
  let pos = initialPosition();
  let plies = 0;

  while (plies < maxPlies) {
    const moves = generateMoves(pos);

    // Checkmate/stalemate
    if (moves.length === 0) {
      const winner = pos.side === 1 ? 'v2' : 'v3';
      return {
        winner: v3PlaysP1 ? (winner === 'v3' ? 'v3' : 'v2') : (winner === 'v2' ? 'v3' : 'v2'),
        plies,
        reason: 'checkmate'
      };
    }

    // Draw by inactivity
    if (isDrawByInactivity(pos)) {
      return { winner: 'draw', plies, reason: 'inactivity' };
    }

    // Select move based on whose turn
    const isV3Turn = (pos.side === 1 && v3PlaysP1) || (pos.side === -1 && !v3PlaysP1);
    const session = isV3Turn ? sessionV3 : sessionV2;

    const { policyLogits } = await evaluateNN(pos, session);
    const move = selectBestNNMove(policyLogits, moves);

    if (!move) {
      return { winner: 'draw', plies, reason: 'no_move' };
    }

    pos = applyMove(pos, move);
    plies++;
  }

  return { winner: 'draw', plies, reason: 'max_plies' };
}

async function runBenchmark(games = 20) {
  await initModels();

  console.log('='.repeat(80));
  console.log('BENCHMARK: v3 (99.32% Val Acc) vs v2 (95% Val Acc)');
  console.log('='.repeat(80));
  console.log(`Games: ${games} (${games/2} as P1, ${games/2} as P2)`);
  console.log('');

  let v3Wins = 0;
  let v2Wins = 0;
  let draws = 0;

  for (let i = 0; i < games; i++) {
    const v3PlaysP1 = i % 2 === 0;
    const gameNum = i + 1;

    process.stdout.write(`Game ${gameNum}/${games}: v3=${v3PlaysP1 ? 'P1' : 'P2'}, v2=${v3PlaysP1 ? 'P2' : 'P1'} ... `);

    const result = await playGame(v3PlaysP1);

    if (result.winner === 'v3') v3Wins++;
    else if (result.winner === 'v2') v2Wins++;
    else draws++;

    const resultStr = result.winner === 'draw'
      ? `Draw (${result.reason})`
      : `${result.winner.toUpperCase()} wins (${result.reason})`;

    console.log(`${resultStr} in ${result.plies} plies`);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));
  console.log(`v3 wins: ${v3Wins} (${(v3Wins/games*100).toFixed(1)}%)`);
  console.log(`v2 wins: ${v2Wins} (${(v2Wins/games*100).toFixed(1)}%)`);
  console.log(`Draws: ${draws} (${(draws/games*100).toFixed(1)}%)`);
  console.log('');

  // Calculate Elo difference
  const v3Score = (v3Wins + draws * 0.5) / games;
  if (v3Score > 0 && v3Score < 1) {
    const eloDiff = Math.round(400 * Math.log10(v3Score / (1 - v3Score)));
    console.log(`v3 score: ${v3Score.toFixed(3)}`);
    console.log(`Estimated Elo advantage: ${eloDiff > 0 ? '+' : ''}${eloDiff}`);
  }
  console.log('');

  if (v3Wins > v2Wins) {
    console.log('🏆 v3 (99.32% Val Acc, no augment) is STRONGER! 🏆');
  } else if (v2Wins > v3Wins) {
    console.log('🏆 v2 (95% Val Acc) is STRONGER! 🏆');
  } else {
    console.log('🤝 TIE! Both models are equal strength');
  }
}

// Parse args
const args = process.argv.slice(2);
const gamesArg = args.find(arg => arg.startsWith('--games='));
const games = gamesArg ? parseInt(gamesArg.split('=')[1]) : 20;

runBenchmark(games).catch(console.error);
