#!/usr/bin/env tsx
/**
 * Generate training data for NN supervised learning
 *
 * Generates 100K positions labeled with minimax depth 12:
 * - 40K from self-play (diverse positions)
 * - 30K from opening book variations
 * - 20K from tactical positions
 * - 10K from endgame positions
 *
 * Each position includes:
 * - Position state
 * - Best move (policy target)
 * - Position value (value target)
 * - Move scores (for top-k moves)
 */

import * as fs from 'fs';
import * as path from 'path';
import { Position } from '../src/coreClaude/position';
import { generateMoves, Move, applyMove } from '../src/coreClaude/movegen';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { B1, bitCount } from '../src/coreClaude/bitboards';

interface TrainingExample {
  // Position state (for feature extraction later)
  position: Position;

  // Labels from minimax depth 12
  bestMove: { from: number; to: number };
  positionValue: number;  // -1 to +1

  // Additional info
  depth: number;
  nodes: number;
  moveScores?: Array<{ from: number; to: number; score: number }>;
}

const OUTPUT_DIR = '.tmp/training_data';
const MINIMAX_DEPTH = 12;
const MINIMAX_TIME_MS = 2000; // 2 seconds per position

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

let totalGenerated = 0;

/**
 * Initial position for Thai Checkers
 */
function getInitialPosition(): Position {
  return {
    side: 1,
    p1Men: B1(24) | B1(25) | B1(26) | B1(27) | B1(28) | B1(29) | B1(30) | B1(31),
    p1Kings: 0,
    p2Men: B1(0) | B1(1) | B1(2) | B1(3) | B1(4) | B1(5) | B1(6) | B1(7),
    p2Kings: 0,
    halfmoveClock: 0,
  };
}

/**
 * Run minimax depth 12 to label a position
 */
async function labelPosition(pos: Position): Promise<TrainingExample | null> {
  const tt = new TT();
  const result = await iterativeDeepening(
    pos,
    MINIMAX_TIME_MS,
    tt,
    undefined,
    [],
    undefined,
    MINIMAX_DEPTH
  );

  if (!result.best) {
    return null; // No legal moves or draw
  }

  // Normalize value to [-1, 1]
  const normalizedValue = Math.max(-1, Math.min(1, result.score / 500));

  return {
    position: pos,
    bestMove: { from: result.best.from, to: result.best.to },
    positionValue: normalizedValue,
    depth: result.depth,
    nodes: result.nodes || 0,
  };
}

/**
 * Generate positions from self-play
 */
async function generateSelfPlayPositions(count: number): Promise<TrainingExample[]> {
  console.log(`\nGenerating ${count} self-play positions...`);
  const examples: TrainingExample[] = [];

  for (let game = 0; game < count / 20; game++) { // ~20 positions per game
    let pos = getInitialPosition();
    const gamePositions: Position[] = [];

    // Play a game with random moves (for diversity)
    for (let ply = 0; ply < 40; ply++) {
      const moves = generateMoves(pos);
      if (moves.length === 0) break;

      // Random move with some strategy (prefer captures)
      const hasCap = moves[0].captured.length > 0;
      let move: Move;
      if (hasCap) {
        move = moves[Math.floor(Math.random() * Math.min(3, moves.length))];
      } else {
        move = moves[Math.floor(Math.random() * moves.length)];
      }

      // Save position every 2 plies
      if (ply % 2 === 0) {
        gamePositions.push({ ...pos });
      }

      pos = applyMove(pos, move);
    }

    // Label random positions from this game
    const sampled = gamePositions
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.min(20, gamePositions.length));

    for (const samplePos of sampled) {
      const example = await labelPosition(samplePos);
      if (example) {
        examples.push(example);
        totalGenerated++;
        if (totalGenerated % 100 === 0) {
          console.log(`  Progress: ${totalGenerated} positions generated`);
        }
      }
      if (examples.length >= count) break;
    }
    if (examples.length >= count) break;
  }

  return examples.slice(0, count);
}

/**
 * Generate positions from opening book variations
 */
async function generateOpeningPositions(count: number): Promise<TrainingExample[]> {
  console.log(`\nGenerating ${count} opening positions...`);
  const examples: TrainingExample[] = [];

  // Start from initial position and explore variations
  const queue: Position[] = [getInitialPosition()];
  const visited = new Set<number>();

  while (queue.length > 0 && examples.length < count) {
    const pos = queue.shift()!;
    const hash = hashPosition(pos);

    if (visited.has(hash)) continue;
    visited.add(hash);

    // Label this position
    const example = await labelPosition(pos);
    if (example) {
      examples.push(example);
      totalGenerated++;
      if (totalGenerated % 100 === 0) {
        console.log(`  Progress: ${totalGenerated} positions generated`);
      }
    }

    // Add children (top 3 moves)
    const moves = generateMoves(pos);
    for (const move of moves.slice(0, 3)) {
      const child = applyMove(pos, move);
      const totalPieces = bitCount(
        child.p1Men | child.p1Kings | child.p2Men | child.p2Kings
      );
      // Only opening positions (≥12 pieces)
      if (totalPieces >= 12) {
        queue.push(child);
      }
    }
  }

  return examples.slice(0, count);
}

/**
 * Generate tactical positions (positions with forced captures)
 */
async function generateTacticalPositions(count: number): Promise<TrainingExample[]> {
  console.log(`\nGenerating ${count} tactical positions...`);
  const examples: TrainingExample[] = [];

  // Generate from self-play but filter for tactical
  for (let attempt = 0; attempt < count * 3; attempt++) {
    let pos = getInitialPosition();

    // Play random moves to reach middlegame
    for (let i = 0; i < 10 + Math.floor(Math.random() * 10); i++) {
      const moves = generateMoves(pos);
      if (moves.length === 0) break;
      const move = moves[Math.floor(Math.random() * moves.length)];
      pos = applyMove(pos, move);
    }

    // Check if tactical (has forced captures)
    const moves = generateMoves(pos);
    if (moves.length > 0 && moves[0].captured.length > 0) {
      const example = await labelPosition(pos);
      if (example) {
        examples.push(example);
        totalGenerated++;
        if (totalGenerated % 100 === 0) {
          console.log(`  Progress: ${totalGenerated} positions generated`);
        }
      }
    }

    if (examples.length >= count) break;
  }

  return examples.slice(0, count);
}

/**
 * Generate endgame positions (≤8 pieces)
 */
async function generateEndgamePositions(count: number): Promise<TrainingExample[]> {
  console.log(`\nGenerating ${count} endgame positions...`);
  const examples: TrainingExample[] = [];

  // Generate from self-play but filter for endgame
  for (let attempt = 0; attempt < count * 3; attempt++) {
    let pos = getInitialPosition();

    // Play random moves to reach endgame
    for (let i = 0; i < 30 + Math.floor(Math.random() * 20); i++) {
      const moves = generateMoves(pos);
      if (moves.length === 0) break;
      const move = moves[Math.floor(Math.random() * moves.length)];
      pos = applyMove(pos, move);
    }

    // Check if endgame (≤8 pieces)
    const totalPieces = bitCount(
      pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings
    );
    if (totalPieces <= 8 && totalPieces >= 4) {
      const example = await labelPosition(pos);
      if (example) {
        examples.push(example);
        totalGenerated++;
        if (totalGenerated % 100 === 0) {
          console.log(`  Progress: ${totalGenerated} positions generated`);
        }
      }
    }

    if (examples.length >= count) break;
  }

  return examples.slice(0, count);
}

/**
 * Save training data to JSON
 */
function saveTrainingData(examples: TrainingExample[], filename: string) {
  const filepath = path.join(OUTPUT_DIR, filename);
  fs.writeFileSync(filepath, JSON.stringify(examples, null, 2));
  console.log(`\nSaved ${examples.length} examples to ${filepath}`);
  console.log(`File size: ${(fs.statSync(filepath).size / 1024 / 1024).toFixed(2)} MB`);
}

/**
 * Main function
 */
async function main() {
  console.log('='.repeat(80));
  console.log('TRAINING DATA GENERATOR FOR NN');
  console.log('='.repeat(80));
  console.log('');
  console.log('Target: 100K positions labeled with minimax depth 12');
  console.log('');
  console.log('Breakdown:');
  console.log('  - 40K self-play positions (diverse)');
  console.log('  - 30K opening positions (book variations)');
  console.log('  - 20K tactical positions (forced captures)');
  console.log('  - 10K endgame positions (≤8 pieces)');
  console.log('');
  console.log(`Minimax depth: ${MINIMAX_DEPTH}`);
  console.log(`Time per position: ${MINIMAX_TIME_MS}ms`);
  console.log('');
  console.log('Estimated time: ~55 hours (100K × 2s)');
  console.log('For quick test, reduce counts in code');
  console.log('');
  console.log('='.repeat(80));

  const startTime = Date.now();

  // Generate each category
  const selfPlay = await generateSelfPlayPositions(40000);
  saveTrainingData(selfPlay, 'selfplay.json');

  const opening = await generateOpeningPositions(30000);
  saveTrainingData(opening, 'opening.json');

  const tactical = await generateTacticalPositions(20000);
  saveTrainingData(tactical, 'tactical.json');

  const endgame = await generateEndgamePositions(10000);
  saveTrainingData(endgame, 'endgame.json');

  // Combine all
  const allExamples = [...selfPlay, ...opening, ...tactical, ...endgame];
  saveTrainingData(allExamples, 'training_data_100k.json');

  const elapsedHours = (Date.now() - startTime) / 1000 / 60 / 60;

  console.log('');
  console.log('='.repeat(80));
  console.log('GENERATION COMPLETE!');
  console.log('='.repeat(80));
  console.log('');
  console.log(`Total positions: ${allExamples.length}`);
  console.log(`Time elapsed: ${elapsedHours.toFixed(2)} hours`);
  console.log(`Output directory: ${OUTPUT_DIR}`);
  console.log('');
  console.log('Next steps:');
  console.log('  1. Implement feature extraction (320 dims)');
  console.log('  2. Create PyTorch training script');
  console.log('  3. Upload to Colab for GPU training');
  console.log('');
}

main().catch(console.error);
