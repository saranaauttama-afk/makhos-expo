/**
 * ML-powered AI Player
 *
 * Uses neural network to select moves instead of traditional alpha-beta search
 */

import { Position } from '../position';
import { Move, generateMoves } from '../movegen';
import { MLModel } from './inference';

export interface MLPlayerConfig {
  modelPath: string;
  temperature?: number;  // Sampling temperature (0 = greedy, higher = more random)
  topK?: number;         // Consider only top K moves
}

export class MLPlayer {
  private model: MLModel;
  private temperature: number;
  private topK: number;

  constructor(config: MLPlayerConfig) {
    this.model = new MLModel(config.modelPath);
    this.temperature = config.temperature ?? 0.1;  // Low temp = more deterministic
    this.topK = config.topK ?? 10;
  }

  async load(): Promise<void> {
    await this.model.load();
  }

  /**
   * Select best move using ML model
   *
   * @param pos - Current position
   * @param timeMs - Time budget (not used for ML, but kept for API compatibility)
   * @returns Selected move
   */
  async selectMove(pos: Position, timeMs?: number): Promise<Move | undefined> {
    if (!this.model.isLoaded()) {
      throw new Error('ML model not loaded');
    }

    // Generate legal moves
    const legalMoves = generateMoves(pos);
    if (legalMoves.length === 0) return undefined;
    if (legalMoves.length === 1) return legalMoves[0];

    // Create legal move mask (32*32 array)
    const legalMask = new Array(32 * 32).fill(false);
    for (const move of legalMoves) {
      const idx = move.from * 32 + move.to;
      legalMask[idx] = true;
    }

    // Run inference
    const prediction = await this.model.predict(pos);

    // Get top moves
    const topMoves = this.model.getTopMoves(prediction.policy, legalMask, this.topK);

    if (topMoves.length === 0) {
      // Fallback: return first legal move
      console.warn('No moves from ML model, using fallback');
      return legalMoves[0];
    }

    // Select move (with temperature sampling)
    const selectedMove = this.sampleMove(topMoves, legalMoves);

    return selectedMove;
  }

  /**
   * Sample move from probability distribution with temperature
   */
  private sampleMove(
    topMoves: Array<{ from: number; to: number; prob: number }>,
    legalMoves: Move[]
  ): Move {
    if (this.temperature === 0) {
      // Greedy: pick highest probability
      const best = topMoves[0];
      return this.findMove(legalMoves, best.from, best.to)!;
    }

    // Apply temperature
    const adjustedProbs = topMoves.map(m => Math.pow(m.prob, 1.0 / this.temperature));
    const sum = adjustedProbs.reduce((a, b) => a + b, 0);
    const normalized = adjustedProbs.map(p => p / sum);

    // Sample
    const rand = Math.random();
    let cumulative = 0;
    for (let i = 0; i < topMoves.length; i++) {
      cumulative += normalized[i];
      if (rand <= cumulative) {
        const selected = topMoves[i];
        return this.findMove(legalMoves, selected.from, selected.to)!;
      }
    }

    // Fallback (shouldn't happen)
    const fallback = topMoves[0];
    return this.findMove(legalMoves, fallback.from, fallback.to)!;
  }

  /**
   * Find Move object from legal moves by from/to squares
   */
  private findMove(legalMoves: Move[], from: number, to: number): Move | undefined {
    return legalMoves.find(m => m.from === from && m.to === to);
  }

  /**
   * Get position evaluation from ML model
   */
  async evaluate(pos: Position): Promise<number> {
    const prediction = await this.model.predict(pos);
    return prediction.value;
  }
}