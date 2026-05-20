#!/usr/bin/env tsx
/**
 * extractFeaturesV2.ts
 *
 * Extract NN features from training data (selfplay + opening)
 * Converts positions to 320-dim feature vectors for Python training
 */

import * as fs from 'fs';
import * as path from 'path';
import { extractNNFeatures } from '../src/coreClaude/nnFeatures';

interface TrainingExample {
  fen?: string;  // Old selfplay format
  position?: {   // New opening format
    side: number;
    p1Men: number;
    p1Kings: number;
    p2Men: number;
    p2Kings: number;
    halfmoveClock?: number;
  };
  bestMove: { from: number; to: number };
  positionValue: number;
  ply?: number;
}

interface FeatureExample {
  features: number[];
  bestMove: { from: number; to: number };
  positionValue: number;
}

async function extractFeatures() {
  const dataDir = '.tmp/training_data';
  const outputDir = '.tmp/training_data_with_features';

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Find all training data files
  const files = fs.readdirSync(dataDir)
    .filter(f => f.startsWith('chunk_') && f.endsWith('.json'))
    .sort();

  console.log(`Found ${files.length} training data files`);
  console.log('');

  let totalExamples = 0;
  let processedFiles = 0;

  for (const filename of files) {
    const inputPath = path.join(dataDir, filename);
    const outputPath = path.join(outputDir, filename);

    // Skip if already processed
    if (fs.existsSync(outputPath)) {
      console.log(`⏭️  Skip ${filename} (already exists)`);
      const existing = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));
      totalExamples += existing.length;
      processedFiles++;
      continue;
    }

    console.log(`📦 Processing ${filename}...`);

    // Read training data
    const data = JSON.parse(fs.readFileSync(inputPath, 'utf-8')) as TrainingExample[];

    // Extract features for each example
    const featuresData: FeatureExample[] = [];

    for (let i = 0; i < data.length; i++) {
      const example = data[i];

      // Parse position (support both FEN and position object)
      const pos = example.position
        ? convertPositionObject(example.position)
        : parseFEN(example.fen!);

      // Extract 320-dim features
      const features = Array.from(extractNNFeatures(pos));

      featuresData.push({
        features,
        bestMove: example.bestMove,
        positionValue: example.positionValue,
      });

      if ((i + 1) % 1000 === 0) {
        process.stdout.write(`\r  Progress: ${i + 1}/${data.length} (${((i + 1) / data.length * 100).toFixed(1)}%)`);
      }
    }

    console.log(`\r  Progress: ${data.length}/${data.length} (100.0%)`);

    // Write features to file
    fs.writeFileSync(outputPath, JSON.stringify(featuresData));

    totalExamples += featuresData.length;
    processedFiles++;

    console.log(`✅ Saved ${featuresData.length} examples to ${filename}`);
    console.log('');
  }

  console.log('='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Processed files: ${processedFiles}/${files.length}`);
  console.log(`Total examples:  ${totalExamples.toLocaleString()}`);
  console.log(`Output dir:      ${outputDir}`);
  console.log('');
  console.log('✅ Feature extraction complete!');
  console.log('');
  console.log('Next steps:');
  console.log('1. Upload .tmp/training_data_with_features/ to G Drive');
  console.log('2. Train model on Colab with combined data');
  console.log('3. Download new model and benchmark!');
}

// Convert position object to internal format
function convertPositionObject(pos: any): any {
  // Numbers from JSON need to be converted to BigInt
  const p1Men = typeof pos.p1Men === 'number' ? BigInt(pos.p1Men) : pos.p1Men;
  const p1Kings = typeof pos.p1Kings === 'number' ? BigInt(pos.p1Kings) : pos.p1Kings;
  const p2Men = typeof pos.p2Men === 'number' ? BigInt(pos.p2Men) : pos.p2Men;
  const p2Kings = typeof pos.p2Kings === 'number' ? BigInt(pos.p2Kings) : pos.p2Kings;

  return {
    p1Men,
    p1Kings,
    p2Men,
    p2Kings,
    side: pos.side,
  };
}

// Simple FEN parser for Thai Checkers
function parseFEN(fen: string): any {
  const [board, side] = fen.split(' ');
  let p1Men = 0n, p1Kings = 0n, p2Men = 0n, p2Kings = 0n;

  const rows = board.split('/');
  let square = 0;

  for (let r = 0; r < 8; r++) {
    const row = rows[r];
    let col = 0;

    for (const char of row) {
      if (char >= '1' && char <= '8') {
        col += parseInt(char);
      } else {
        if ((r + col) % 2 === 1) { // Only dark squares
          const bit = 1n << BigInt(square);

          if (char === 'w') p1Men |= bit;
          else if (char === 'W') p1Kings |= bit;
          else if (char === 'b') p2Men |= bit;
          else if (char === 'B') p2Kings |= bit;

          square++;
        }
        col++;
      }
    }
  }

  return {
    p1Men,
    p1Kings,
    p2Men,
    p2Kings,
    side: side === 'w' ? 1 : -1,
  };
}

extractFeatures().catch(console.error);
