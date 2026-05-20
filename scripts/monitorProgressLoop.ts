#!/usr/bin/env tsx
/**
 * Monitor training data generation progress every 5 minutes
 */

import * as fs from 'fs';
import * as path from 'path';

const DATA_DIR = '.tmp/training_data';
const TARGET = 10000;
const INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

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

async function monitor() {
  let prevCount = 0;

  while (true) {
    const count = countExamples();
    const progress = (count / TARGET * 100).toFixed(1);
    const remaining = TARGET - count;

    console.log(`[${new Date().toLocaleTimeString()}] Progress: ${count.toLocaleString()} / ${TARGET.toLocaleString()} (${progress}%)`);

    if (prevCount > 0) {
      const delta = count - prevCount;
      console.log(`  Change: +${delta} examples in 5 minutes`);

      if (delta > 0) {
        const rate = delta / 5; // examples per minute
        const etaMinutes = remaining / rate;
        console.log(`  Rate: ~${rate.toFixed(1)} examples/min`);
        console.log(`  ETA: ~${(etaMinutes / 60).toFixed(1)} hours`);
      }
    }

    if (count >= TARGET) {
      console.log(`\n✅ TARGET REACHED! ${count.toLocaleString()} examples generated!`);
      break;
    }

    prevCount = count;
    console.log(`  Next check in 5 minutes...\n`);

    await new Promise(resolve => setTimeout(resolve, INTERVAL_MS));
  }
}

monitor().catch(console.error);
