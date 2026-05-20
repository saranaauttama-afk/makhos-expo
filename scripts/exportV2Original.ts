#!/usr/bin/env tsx
/**
 * Export ORIGINAL v2 selfplay data with features
 * This exports the 4 original selfplay files that achieved 95% Val Acc
 */

import * as fs from 'fs';
import * as path from 'path';
import { Position, normalizePosition } from '../src/coreClaude/position';
import { extractNNFeatures } from '../src/coreClaude/nnFeatures';

interface TrainingExample {
  position: Position;
  bestMove: { from: number; to: number };
  positionValue: number;
  depth: number;
  nodes: number;
  category?: string;
}

interface TrainingExampleWithFeatures extends TrainingExample {
  features: number[]; // 320-dim array
}

/**
 * Process a single chunk file
 */
function processChunkFile(inputPath: string, outputPath: string) {
  console.log(`Processing ${path.basename(inputPath)}...`);

  // Load data
  const data: TrainingExample[] = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

  // Extract features
  const dataWithFeatures: TrainingExampleWithFeatures[] = data.map((example) => {
    // Normalize position from JSON format to runtime format
    const normalizedPos = normalizePosition(example.position);
    const features = extractNNFeatures(normalizedPos);

    return {
      ...example,
      position: normalizedPos, // Use normalized position
      features: Array.from(features), // Convert Float32Array to regular array
    };
  });

  // Save
  fs.writeFileSync(outputPath, JSON.stringify(dataWithFeatures, null, 2));

  console.log(`  ✅ Saved ${dataWithFeatures.length} examples with features`);
}

/**
 * Main
 */
function main() {
  const inputDir = '.tmp/training_data';
  const outputDir = '.tmp/v2_original_with_features';

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Find original 4 selfplay files
  const files = fs.readdirSync(inputDir)
    .filter((f) => f.startsWith('chunk_selfplay_') && f.endsWith('.json'))
    .sort();

  if (files.length === 0) {
    console.error('❌ No original selfplay files found in .tmp/training_data');
    process.exit(1);
  }

  console.log('========================================');
  console.log('EXPORT V2 ORIGINAL SELFPLAY DATA');
  console.log('========================================');
  console.log(`Found ${files.length} original selfplay files`);
  console.log('');

  // Process each file
  for (const file of files) {
    const inputPath = path.join(inputDir, file);
    const outputPath = path.join(outputDir, file);
    processChunkFile(inputPath, outputPath);
  }

  console.log('');
  console.log('========================================');
  console.log('✅✅✅ COMPLETE!');
  console.log('========================================');
  console.log(`Processed ${files.length} files`);
  console.log(`Output: ${outputDir}`);
  console.log('');
  console.log('Next: Upload to Colab and train v2.1');
  console.log('Expected: 95% Val Acc (same as v2)');
}

main();
