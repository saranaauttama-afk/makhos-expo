#!/usr/bin/env tsx
/**
 * Incremental training data generator
 *
 * Generates data in chunks of 1000 positions at a time.
 * Saves progress after each chunk.
 * Can resume from last checkpoint if interrupted.
 *
 * Usage:
 *   npx tsx scripts/generateTrainingDataIncremental.ts
 *
 * Run multiple times until reaching 100K positions.
 */

import * as fs from 'fs';
import * as path from 'path';
import { Position } from '../src/coreClaude/position';
import { generateMoves, Move, applyMove } from '../src/coreClaude/movegen';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { B1, bitCount } from '../src/coreClaude/bitboards';

interface TrainingExample {
  position: Position;
  bestMove: { from: number; to: number };
  positionValue: number;
  depth: number;
  nodes: number;
  category: 'selfplay' | 'opening' | 'tactical' | 'endgame';
}

interface Progress {
  totalGenerated: number;
  selfplayCount: number;
  openingCount: number;
  tacticalCount: number;
  endgameCount: number;
  lastChunkFile: string;
  startTime: number;
}

const OUTPUT_DIR = '.tmp/training_data';
const PROGRESS_FILE = path.join(OUTPUT_DIR, 'progress.json');
const CHUNK_SIZE = 10000; // Generate 10000 at a time

// Search config for each category
const DEPTH_CONFIG = {
  selfplay: 12,
  opening: 3,     // Low depth (already done)
  tactical: 8,    // Higher depth for tactical accuracy
  endgame: 14,    // Very high depth for endgame precision
};
const TIME_CONFIG = {
  selfplay: 2000,
  opening: 1000,
  tactical: 5000,  // More time for tactical positions
  endgame: 10000,  // Most time for endgame
};

// Generate unique ID for parallel processing
function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 9);
}

// Target distribution - Gen selfplay only for consistent training
const TARGET_SELFPLAY = 60000;   // Gen 20K more selfplay (depth 12 consistent)
const TARGET_OPENING = 40000;    // Skip (depth 3 too shallow)
const TARGET_TACTICAL = 10000;   // Skip (depth 8 mismatch with selfplay)
const TARGET_ENDGAME = 0;        // Skip for now
const TARGET_TOTAL = 110000;

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

function loadProgress(): Progress {
  if (fs.existsSync(PROGRESS_FILE)) {
    return JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf-8'));
  }
  return {
    totalGenerated: 0,
    selfplayCount: 0,
    openingCount: 0,
    tacticalCount: 0,
    endgameCount: 0,
    lastChunkFile: '',
    startTime: Date.now(),
  };
}

function saveProgress(progress: Progress) {
  fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
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

async function labelPosition(
  pos: Position,
  category: 'selfplay' | 'opening' | 'tactical' | 'endgame'
): Promise<TrainingExample | null> {
  const tt = new TT();
  const depth = DEPTH_CONFIG[category];
  const timeMs = TIME_CONFIG[category];

  const result = await iterativeDeepening(pos, timeMs, tt, undefined, [], undefined, depth);

  if (!result.best) return null;

  const normalizedValue = Math.max(-1, Math.min(1, result.score / 500));

  return {
    position: pos,
    bestMove: { from: result.best.from, to: result.best.to },
    positionValue: normalizedValue,
    depth: result.depth,
    nodes: result.nodes || 0,
    category,
  };
}

async function generateChunk(
  category: 'selfplay' | 'opening' | 'tactical' | 'endgame',
  count: number
): Promise<TrainingExample[]> {
  const examples: TrainingExample[] = [];

  for (let i = 0; i < count; i++) {
    let pos = getInitialPosition();

    // Different strategies per category
    if (category === 'selfplay') {
      // Random walk
      const steps = 5 + Math.floor(Math.random() * 20);
      for (let j = 0; j < steps; j++) {
        const moves = generateMoves(pos);
        if (moves.length === 0) break;
        const move = moves[Math.floor(Math.random() * moves.length)];
        pos = applyMove(pos, move);
      }
    } else if (category === 'opening') {
      // Short walks (opening phase)
      const steps = 5 + Math.floor(Math.random() * 10);
      for (let j = 0; j < steps; j++) {
        const moves = generateMoves(pos);
        if (moves.length === 0) break;
        const move = moves[Math.floor(Math.random() * Math.min(3, moves.length))];
        pos = applyMove(pos, move);
      }
      const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
      if (totalPieces < 12) continue; // Skip if not opening anymore
    } else if (category === 'tactical') {
      // Walk until forced captures
      for (let attempt = 0; attempt < 20; attempt++) {
        const steps = 10 + Math.floor(Math.random() * 10);
        pos = getInitialPosition();
        for (let j = 0; j < steps; j++) {
          const moves = generateMoves(pos);
          if (moves.length === 0) break;
          const move = moves[Math.floor(Math.random() * moves.length)];
          pos = applyMove(pos, move);
        }
        const moves = generateMoves(pos);
        if (moves.length > 0 && moves[0].captured.length > 0) break;
      }
    } else if (category === 'endgame') {
      // Walk to endgame
      const steps = 30 + Math.floor(Math.random() * 20);
      for (let j = 0; j < steps; j++) {
        const moves = generateMoves(pos);
        if (moves.length === 0) break;
        const move = moves[Math.floor(Math.random() * moves.length)];
        pos = applyMove(pos, move);
      }
      const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
      if (totalPieces > 8 || totalPieces < 4) continue; // Skip if not endgame
    }

    const example = await labelPosition(pos, category);
    if (example) {
      examples.push(example);
      process.stdout.write(`\r  ${category}: ${i + 1}/${count} generated`);
    }
  }
  process.stdout.write('\n');

  return examples;
}

async function main() {
  const progress = loadProgress();

  console.log('='.repeat(80));
  console.log('INCREMENTAL TRAINING DATA GENERATOR');
  console.log('='.repeat(80));
  console.log('');
  console.log('Progress:');
  console.log(`  Total: ${progress.totalGenerated} / ${TARGET_TOTAL} (${(progress.totalGenerated / TARGET_TOTAL * 100).toFixed(1)}%)`);
  console.log(`  Self-play: ${progress.selfplayCount} / ${TARGET_SELFPLAY}`);
  console.log(`  Opening: ${progress.openingCount} / ${TARGET_OPENING}`);
  console.log(`  Tactical: ${progress.tacticalCount} / ${TARGET_TACTICAL}`);
  console.log(`  Endgame: ${progress.endgameCount} / ${TARGET_ENDGAME}`);
  console.log('');

  if (progress.totalGenerated >= TARGET_TOTAL) {
    console.log('✅ ALREADY COMPLETE!');
    console.log('');
    console.log(`Total time: ${((Date.now() - progress.startTime) / 1000 / 60 / 60).toFixed(1)} hours`);
    console.log(`Output: ${OUTPUT_DIR}`);
    return;
  }

  // Determine next category to generate
  let category: 'selfplay' | 'opening' | 'tactical' | 'endgame';
  let remaining: number;

  if (progress.selfplayCount < TARGET_SELFPLAY) {
    category = 'selfplay';
    remaining = TARGET_SELFPLAY - progress.selfplayCount;
  } else if (progress.openingCount < TARGET_OPENING) {
    category = 'opening';
    remaining = TARGET_OPENING - progress.openingCount;
  } else if (progress.tacticalCount < TARGET_TACTICAL) {
    category = 'tactical';
    remaining = TARGET_TACTICAL - progress.tacticalCount;
  } else {
    category = 'endgame';
    remaining = TARGET_ENDGAME - progress.endgameCount;
  }

  const toGenerate = Math.min(CHUNK_SIZE, remaining);

  console.log(`Generating ${toGenerate} ${category} positions...`);
  console.log(`Estimated time: ${(toGenerate * TIME_CONFIG[category] / 1000 / 60).toFixed(1)} minutes`);
  console.log('');

  const chunkStartTime = Date.now();
  const examples = await generateChunk(category, toGenerate);

  // Save chunk with unique ID to avoid race conditions in parallel processing
  const uniqueId = generateId();
  const chunkFile = `chunk_${category}_${uniqueId}.json`;
  const chunkPath = path.join(OUTPUT_DIR, chunkFile);
  fs.writeFileSync(chunkPath, JSON.stringify(examples, null, 2));

  // Update progress
  progress.totalGenerated += examples.length;
  progress[`${category}Count`] += examples.length;
  progress.lastChunkFile = chunkFile;
  saveProgress(progress);

  const chunkTime = (Date.now() - chunkStartTime) / 1000 / 60;
  const totalTime = (Date.now() - progress.startTime) / 1000 / 60 / 60;

  console.log('');
  console.log('✅ Chunk complete!');
  console.log(`  Generated: ${examples.length} positions`);
  console.log(`  Chunk time: ${chunkTime.toFixed(1)} minutes`);
  console.log(`  Saved to: ${chunkFile}`);
  console.log('');
  console.log('Overall progress:');
  console.log(`  Total: ${progress.totalGenerated} / ${TARGET_TOTAL} (${(progress.totalGenerated / TARGET_TOTAL * 100).toFixed(1)}%)`);
  console.log(`  Total time: ${totalTime.toFixed(1)} hours`);
  console.log('');
  console.log('To continue, run this script again:');
  console.log('  npx tsx scripts/generateTrainingDataIncremental.ts');
  console.log('');
}

main().catch(console.error);
