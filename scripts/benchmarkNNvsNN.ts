#!/usr/bin/env tsx
/**
 * benchmarkNNvsNN.ts
 *
 * Benchmark two NN models against each other
 * Usage: npx tsx scripts/benchmarkNNvsNN.ts --model1=v2 --model2=v3 --games=20
 */

import { applyMove, generateMoves, Move } from '../src/coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { initNNInference, evaluateNN, selectBestNNMove, closeNNInference } from '../src/coreClaude/nnInference.node';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import * as ort from 'onnxruntime-node';

interface GameResult {
  winner: 'Model1' | 'Model2' | 'Draw';
  moves: number;
  reason: string;
}

let session1: ort.InferenceSession | null = null;
let session2: ort.InferenceSession | null = null;

async function initModels(model1Path: string, model2Path: string) {
  console.log(`Loading Model 1 from: ${model1Path}`);
  session1 = await ort.InferenceSession.create(model1Path);
  console.log('Model 1 loaded successfully');

  console.log(`Loading Model 2 from: ${model2Path}`);
  session2 = await ort.InferenceSession.create(model2Path);
  console.log('Model 2 loaded successfully');
}

async function evaluateWithSession(session: ort.InferenceSession, pos: Position) {
  const { extractNNFeatures } = await import('../src/coreClaude/nnFeatures');
  const features = extractNNFeatures(pos);
  const inputTensor = new ort.Tensor('float32', features, [1, 320]);
  const outputs = await session.run({ features: inputTensor });
  const policyLogits = outputs['policy_logits'].data as Float32Array;
  const valueOutput = outputs['value'].data as Float32Array;
  return { policyLogits, value: valueOutput[0] };
}

/**
 * Play one game: Model1 vs Model2
 */
async function playGame(model1First: boolean, maxPlies: number = 200): Promise<GameResult> {
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
      const winner = pos.side === 1 ? 'Model2' : 'Model1';
      return {
        winner: model1First ? winner : (winner === 'Model1' ? 'Model2' : 'Model1'),
        moves,
        reason: 'no_moves'
      };
    }

    // Choose move based on which model's turn
    let chosenMove: Move;
    const isModel1Turn = (pos.side === 1 && model1First) || (pos.side === -1 && !model1First);

    if (isModel1Turn && session1) {
      // Model 1's turn
      const { policyLogits } = await evaluateWithSession(session1, pos);
      chosenMove = selectBestNNMove(policyLogits, legalMoves, pos);
    } else if (!isModel1Turn && session2) {
      // Model 2's turn
      const { policyLogits } = await evaluateWithSession(session2, pos);
      chosenMove = selectBestNNMove(policyLogits, legalMoves, pos);
    } else {
      // Fallback: random move
      chosenMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
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
async function runHeadToHead(games: number = 20): Promise<void> {
  console.log('='.repeat(80));
  console.log('HEAD-TO-HEAD: Model 1 vs Model 2');
  console.log('='.repeat(80));
  console.log(`Playing ${games} games (${games / 2} as each color)`);
  console.log('');

  let model1Wins = 0;
  let model2Wins = 0;
  let draws = 0;

  for (let i = 0; i < games; i++) {
    const model1First = i % 2 === 0;
    console.log(`Game ${i + 1}/${games}: ${model1First ? 'Model1 (P1) vs Model2 (P2)' : 'Model2 (P1) vs Model1 (P2)'}`);

    const result = await playGame(model1First);
    console.log(`  Result: ${result.winner} (${result.moves} moves, ${result.reason})`);

    if (result.winner === 'Model1') model1Wins++;
    else if (result.winner === 'Model2') model2Wins++;
    else draws++;

    console.log('');
  }

  console.log('='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));
  console.log(`Model 1 wins: ${model1Wins} (${((model1Wins / games) * 100).toFixed(1)}%)`);
  console.log(`Model 2 wins: ${model2Wins} (${((model2Wins / games) * 100).toFixed(1)}%)`);
  console.log(`Draws:        ${draws} (${((draws / games) * 100).toFixed(1)}%)`);
  console.log('');

  // Calculate Elo difference (approximate)
  if (model1Wins + model2Wins > 0) {
    const score1 = (model1Wins + draws * 0.5) / games;
    const eloDiff = Math.round(400 * Math.log10(score1 / (1 - score1)));
    console.log(`Model 1 Elo advantage: ${eloDiff > 0 ? '+' : ''}${eloDiff}`);
  }
}

/**
 * Main
 */
async function main() {
  const args = process.argv.slice(2);
  const model1 = args.find((a) => a.startsWith('--model1='))?.split('=')[1] || 'v2_selfplay_only';
  const model2 = args.find((a) => a.startsWith('--model2='))?.split('=')[1] || 'v3_with_opening';
  const games = parseInt(args.find((a) => a.startsWith('--games='))?.split('=')[1] || '20');

  const model1Path = `models/thai_checkers_${model1}.onnx`;
  const model2Path = `models/thai_checkers_${model2}.onnx`;

  try {
    await initModels(model1Path, model2Path);
    await runHeadToHead(games);
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
