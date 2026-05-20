#!/usr/bin/env tsx
/**
 * Check training data generation progress
 */

import * as fs from 'fs';
import * as path from 'path';

const DATA_DIR = '.tmp/training_data';

interface Progress {
  totalGenerated: number;
  selfplayCount: number;
  openingCount: number;
  tacticalCount: number;
  endgameCount: number;
}

function countExamples(): number {
  const files = fs.readdirSync(DATA_DIR)
    .filter(f => f.startsWith('chunk_') && f.endsWith('.json'));

  let total = 0;
  for (const file of files) {
    const filePath = path.join(DATA_DIR, file);
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      total += data.length;
    } catch (err) {
      // Skip invalid files
    }
  }

  return total;
}

function loadProgress(): Progress | null {
  const progressFile = path.join(DATA_DIR, 'progress.json');
  if (!fs.existsSync(progressFile)) {
    return null;
  }

  try {
    return JSON.parse(fs.readFileSync(progressFile, 'utf8'));
  } catch {
    return null;
  }
}

console.log('='.repeat(80));
console.log('TRAINING DATA PROGRESS');
console.log('='.repeat(80));

const total = countExamples();
const progress = loadProgress();

console.log(`\nTotal examples: ${total.toLocaleString()}`);
console.log(`Target: 10,000 examples`);
console.log(`Progress: ${(total / 10000 * 100).toFixed(1)}%`);
console.log(`Remaining: ${(10000 - total).toLocaleString()} examples`);

if (progress) {
  console.log(`\nBreakdown:`);
  console.log(`  Selfplay: ${progress.selfplayCount.toLocaleString()}`);
  console.log(`  Opening: ${progress.openingCount.toLocaleString()}`);
  console.log(`  Tactical: ${progress.tacticalCount.toLocaleString()}`);
  console.log(`  Endgame: ${progress.endgameCount.toLocaleString()}`);
}

console.log('='.repeat(80));
