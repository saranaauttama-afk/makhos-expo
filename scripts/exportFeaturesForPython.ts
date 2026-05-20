#!/usr/bin/env tsx
/**
 * Export features for Python training
 *
 * Converts training data JSON to include extracted features
 * so Python training script doesn't need to re-extract
 */

import * as fs from 'fs';
import * as path from 'path';
import { Position } from '../src/coreClaude/position';
import { extractNNFeatures } from '../src/coreClaude/nnFeatures';

interface TrainingExample {
  position: Position;
  bestMove: { from: number; to: number };
  positionValue: number;
  depth: number;
  nodes: number;
}

interface TrainingExampleWithFeatures extends TrainingExample {
  features: number[]; // 320-dim array
}

/**
 * Process a single chunk file
 */
function processChunkFile(inputPath: string, outputPath: string) {
  console.log(`Processing ${inputPath}...`);

  // Load data
  const data: TrainingExample[] = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

  // Extract features
  const dataWithFeatures: TrainingExampleWithFeatures[] = data.map((example) => {
    const features = extractNNFeatures(example.position);

    return {
      ...example,
      features: Array.from(features), // Convert Float32Array to regular array
    };
  });

  // Save
  fs.writeFileSync(outputPath, JSON.stringify(dataWithFeatures, null, 2));

  console.log(`  Saved ${dataWithFeatures.length} examples with features`);
}

/**
 * Process all chunk files
 */
function main() {
  const inputDir = '.tmp/training_data';
  const outputDir = '.tmp/training_data_with_features';

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Find all chunk files
  const files = fs.readdirSync(inputDir).filter((f) => f.startsWith('chunk_') && f.endsWith('.json'));

  if (files.length === 0) {
    console.error('No chunk files found in .tmp/training_data');
    process.exit(1);
  }

  console.log('=' .repeat(80));
  console.log('EXPORT FEATURES FOR PYTHON TRAINING');
  console.log('=' .repeat(80));
  console.log(`Found ${files.length} chunk files`);
  console.log('');

  // Process each file
  for (const file of files) {
    const inputPath = path.join(inputDir, file);
    const outputPath = path.join(outputDir, file);
    processChunkFile(inputPath, outputPath);
  }

  console.log('');
  console.log('=' .repeat(80));
  console.log('COMPLETE!');
  console.log('=' .repeat(80));
  console.log(`Processed ${files.length} files`);
  console.log(`Output: ${outputDir}`);
  console.log('');
  console.log('Next: Upload to Colab and run train.py');
}

main();
