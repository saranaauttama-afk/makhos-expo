/**
 * ML Model Inference Wrapper
 *
 * Loads ONNX model and provides inference for Makhos position evaluation
 */

import * as ort from 'onnxruntime-react-native';
import { Position } from '../position';

export interface MLPrediction {
  policy: Float32Array;  // (32*32=1024) move probabilities
  value: number;         // [-1, 1] position evaluation
}

export class MLModel {
  private session: ort.InferenceSession | null = null;
  private modelPath: string;

  constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  /**
   * Load the ONNX model from disk or bundled asset
   */
  async load(): Promise<void> {
    console.log(`Loading ML model from ${this.modelPath}...`);

    // For React Native: use bundled asset
    // For Node.js: use file path directly
    this.session = await ort.InferenceSession.create(this.modelPath);
    console.log('✓ ML model loaded successfully');
  }

  /**
   * Convert Position to neural network input format
   *
   * Input shape: (6, 32) representing:
   * - Plane 0: P1 men
   * - Plane 1: P1 kings
   * - Plane 2: P2 men
   * - Plane 3: P2 kings
   * - Plane 4: Side to move (1.0 = P1, 0.0 = P2)
   * - Plane 5: Halfmove clock (normalized by 20)
   */
  private positionToInput(pos: Position): Float32Array {
    const input = new Float32Array(6 * 32);

    // Plane 0: P1 men
    for (let sq = 0; sq < 32; sq++) {
      input[0 * 32 + sq] = (pos.p1Men >>> sq) & 1;
    }

    // Plane 1: P1 kings
    for (let sq = 0; sq < 32; sq++) {
      input[1 * 32 + sq] = (pos.p1Kings >>> sq) & 1;
    }

    // Plane 2: P2 men
    for (let sq = 0; sq < 32; sq++) {
      input[2 * 32 + sq] = (pos.p2Men >>> sq) & 1;
    }

    // Plane 3: P2 kings
    for (let sq = 0; sq < 32; sq++) {
      input[3 * 32 + sq] = (pos.p2Kings >>> sq) & 1;
    }

    // Plane 4: Side to move (1.0 if P1, 0.0 if P2)
    const sideValue = pos.side === 1 ? 1.0 : 0.0;
    for (let sq = 0; sq < 32; sq++) {
      input[4 * 32 + sq] = sideValue;
    }

    // Plane 5: Halfmove clock (normalized)
    const normalizedClock = Math.min(pos.halfmoveClock / 20.0, 1.0);
    for (let sq = 0; sq < 32; sq++) {
      input[5 * 32 + sq] = normalizedClock;
    }

    return input;
  }

  /**
   * Run inference on a position
   *
   * @param pos - The position to evaluate
   * @returns Policy (move probabilities) and value (position evaluation)
   */
  async predict(pos: Position): Promise<MLPrediction> {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Convert position to input tensor
    const inputData = this.positionToInput(pos);
    const tensor = new ort.Tensor('float32', inputData, [1, 6, 32]);

    // Run inference
    const feeds = { state: tensor };
    const outputs = await this.session.run(feeds);

    // Extract outputs
    const policyOutput = outputs.policy.data as Float32Array;
    const valueOutput = outputs.value.data as Float32Array;

    return {
      policy: policyOutput,
      value: valueOutput[0]
    };
  }

  /**
   * Get top-K moves from policy output
   *
   * @param policy - Policy output from model (32*32 array)
   * @param legalMask - Boolean mask of legal moves (32*32 array)
   * @param k - Number of top moves to return
   * @returns Array of {from, to, probability} sorted by probability
   */
  getTopMoves(
    policy: Float32Array,
    legalMask: boolean[],
    k: number = 5
  ): Array<{ from: number; to: number; prob: number }> {
    // Apply softmax to legal moves only
    const legalIndices: number[] = [];
    const legalLogits: number[] = [];

    for (let i = 0; i < 1024; i++) {
      if (legalMask[i]) {
        legalIndices.push(i);
        legalLogits.push(policy[i]);
      }
    }

    // Softmax
    const maxLogit = Math.max(...legalLogits);
    const expValues = legalLogits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    const probs = expValues.map(x => x / sumExp);

    // Create (index, prob) pairs and sort
    const indexedProbs = legalIndices.map((idx, i) => ({
      index: idx,
      prob: probs[i]
    }));

    indexedProbs.sort((a, b) => b.prob - a.prob);

    // Take top K
    return indexedProbs.slice(0, k).map(({ index, prob }) => ({
      from: Math.floor(index / 32),
      to: index % 32,
      prob
    }));
  }

  /**
   * Check if model is loaded
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
}