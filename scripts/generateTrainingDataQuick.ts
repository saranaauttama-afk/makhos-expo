#!/usr/bin/env tsx
/**
 * Quick test version - Generate 100 positions for testing
 * (Instead of 100K which takes 55 hours)
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
  position: Position;
  bestMove: { from: number; to: number };
  positionValue: number;
  depth: number;
  nodes: number;
}

const OUTPUT_DIR = '.tmp/training_data';
const MINIMAX_DEPTH = 10; // Reduced from 12 for speed
const MINIMAX_TIME_MS = 1000; // 1 second

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

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

async function labelPosition(pos: Position): Promise<TrainingExample | null> {
  const tt = new TT();
  const result = await iterativeDeepening(pos, MINIMAX_TIME_MS, tt, undefined, [], undefined, MINIMAX_DEPTH);

  if (!result.best) return null;

  const normalizedValue = Math.max(-1, Math.min(1, result.score / 500));

  return {
    position: pos,
    bestMove: { from: result.best.from, to: result.best.to },
    positionValue: normalizedValue,
    depth: result.depth,
    nodes: result.nodes || 0,
  };
}

async function generateQuickDataset(count: number): Promise<TrainingExample[]> {
  console.log(`Generating ${count} test positions...`);
  const examples: TrainingExample[] = [];

  for (let i = 0; i < count; i++) {
    let pos = getInitialPosition();

    // Random walk to diverse position
    const steps = 5 + Math.floor(Math.random() * 15);
    for (let j = 0; j < steps; j++) {
      const moves = generateMoves(pos);
      if (moves.length === 0) break;
      const move = moves[Math.floor(Math.random() * moves.length)];
      pos = applyMove(pos, move);
    }

    const example = await labelPosition(pos);
    if (example) {
      examples.push(example);
      console.log(`  ${i + 1}/${count} - depth ${example.depth}, value ${example.positionValue.toFixed(2)}`);
    }
  }

  return examples;
}

async function main() {
  console.log('='.repeat(80));
  console.log('QUICK TRAINING DATA TEST (100 positions)');
  console.log('='.repeat(80));
  console.log('');

  const startTime = Date.now();
  const examples = await generateQuickDataset(100);

  const filepath = path.join(OUTPUT_DIR, 'training_data_test_100.json');
  fs.writeFileSync(filepath, JSON.stringify(examples, null, 2));

  const elapsedMin = (Date.now() - startTime) / 1000 / 60;

  console.log('');
  console.log('='.repeat(80));
  console.log('COMPLETE!');
  console.log('='.repeat(80));
  console.log(`Generated: ${examples.length} positions`);
  console.log(`Time: ${elapsedMin.toFixed(1)} minutes`);
  console.log(`File: ${filepath}`);
  console.log(`Size: ${(fs.statSync(filepath).size / 1024).toFixed(1)} KB`);
  console.log('');
  console.log('Next: Implement feature extraction + PyTorch training');
}

main().catch(console.error);
