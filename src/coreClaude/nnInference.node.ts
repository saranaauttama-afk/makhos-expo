/**
 * nnInference.ts - ONNX Neural Network inference for Thai Checkers
 *
 * Loads trained ONNX model and provides evaluation for positions
 */

import * as ort from 'onnxruntime-node';
import * as path from 'path';
import { Position } from './position';
import { extractNNFeatures } from './nnFeatures';
import { Move } from './movegen';

let session: ort.InferenceSession | null = null;

/**
 * Initialize ONNX inference session
 */
export async function initNNInference(modelPath?: string): Promise<void> {
  if (session) return; // Already initialized

  const defaultModelPath = path.join(__dirname, '../../models/thai_checkers_v3.onnx');
  const actualPath = modelPath || defaultModelPath;

  console.log(`Loading NN model from: ${actualPath}`);
  session = await ort.InferenceSession.create(actualPath);
  console.log(`NN model loaded successfully`);
}

/**
 * Evaluate position using NN
 * Returns { value, policyLogits }
 * - value: position evaluation from perspective of side-to-move (-1 to 1)
 * - policyLogits: 1024-dim array of move probabilities (from*32 + to)
 */
export async function evaluateNN(pos: Position): Promise<{ value: number; policyLogits: Float32Array }> {
  if (!session) {
    throw new Error('NN inference not initialized. Call initNNInference() first.');
  }

  // Extract features (320-dim)
  const features = extractNNFeatures(pos);

  // Create input tensor
  const inputTensor = new ort.Tensor('float32', features, [1, 320]);

  // Run inference
  const outputs = await session.run({ features: inputTensor });

  // Extract outputs
  const policyLogits = outputs['policy_logits'].data as Float32Array;
  const value = outputs['value'].data[0] as number;

  return { value, policyLogits };
}

/**
 * Get move probabilities from policy logits
 * Returns map: moveKey -> probability
 */
export function getPolicyProbabilities(policyLogits: Float32Array, legalMoves: Move[]): Map<string, number> {
  const moveProbs = new Map<string, number>();

  // Softmax over legal moves only
  const legalLogits: number[] = [];
  const moveKeys: string[] = [];

  for (const move of legalMoves) {
    const idx = move.from * 32 + move.to;
    legalLogits.push(policyLogits[idx]);
    moveKeys.push(`${move.from}_${move.to}`);
  }

  // Compute softmax
  const maxLogit = Math.max(...legalLogits);
  const expLogits = legalLogits.map((x) => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);

  for (let i = 0; i < legalLogits.length; i++) {
    moveProbs.set(moveKeys[i], expLogits[i] / sumExp);
  }

  return moveProbs;
}

/**
 * Select best move according to NN policy
 * IMPORTANT: Must flip move indices for P2 to match flipped feature extraction
 */
export function selectBestNNMove(policyLogits: Float32Array, legalMoves: Move[], pos: Position): Move {
  let bestMove = legalMoves[0];
  let bestLogit = -Infinity;

  // Features are flipped for P2, so policy indices must also be flipped
  const flipBoard = pos.side === -1;
  const mapSquare = (sq: number) => (flipBoard ? 31 - sq : sq);

  for (const move of legalMoves) {
    // Map move squares to match the flipped feature space
    const from = mapSquare(move.from);
    const to = mapSquare(move.to);
    const idx = from * 32 + to;
    const logit = policyLogits[idx];

    if (logit > bestLogit) {
      bestLogit = logit;
      bestMove = move;
    }
  }

  return bestMove;
}

/**
 * Cleanup
 */
export function closeNNInference(): void {
  session = null;
}
