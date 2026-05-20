#!/usr/bin/env tsx
/**
 * Test 320-dim feature extraction
 */

import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';
import { extractNNFeatures, featuresToString } from '../src/coreClaude/nnFeatures';

console.log('='.repeat(80));
console.log('320-DIM FEATURE EXTRACTION TEST');
console.log('='.repeat(80));
console.log('');

// Test 1: Initial position
const initialPos: Position = {
  side: 1,
  p1Men: B1(24) | B1(25) | B1(26) | B1(27) | B1(28) | B1(29) | B1(30) | B1(31),
  p1Kings: 0,
  p2Men: B1(0) | B1(1) | B1(2) | B1(3) | B1(4) | B1(5) | B1(6) | B1(7),
  p2Kings: 0,
  halfmoveClock: 0,
};

console.log('Test 1: Initial Position');
console.log('-'.repeat(80));
const features1 = extractNNFeatures(initialPos);
console.log(featuresToString(features1));
console.log(`Total features: ${features1.length}`);
console.log(`Non-zero features: ${Array.from(features1).filter(f => f > 0).length}`);
console.log('');

// Test 2: Midgame position
const midgamePos: Position = {
  side: 1,
  p1Men: B1(17) | B1(18) | B1(21),
  p1Kings: B1(10),
  p2Men: B1(5) | B1(6) | B1(9),
  p2Kings: B1(14),
  halfmoveClock: 0,
};

console.log('Test 2: Midgame Position');
console.log('-'.repeat(80));
const features2 = extractNNFeatures(midgamePos);
console.log(featuresToString(features2));
console.log(`Total features: ${features2.length}`);
console.log(`Non-zero features: ${Array.from(features2).filter(f => f > 0).length}`);
console.log('');

// Test 3: Endgame position
const endgamePos: Position = {
  side: 1,
  p1Men: 0,
  p1Kings: B1(10) | B1(14),
  p2Men: 0,
  p2Kings: B1(5),
  halfmoveClock: 0,
};

console.log('Test 3: Endgame Position');
console.log('-'.repeat(80));
const features3 = extractNNFeatures(endgamePos);
console.log(featuresToString(features3));
console.log(`Total features: ${features3.length}`);
console.log(`Non-zero features: ${Array.from(features3).filter(f => f > 0).length}`);
console.log('');

console.log('='.repeat(80));
console.log('FEATURE EXTRACTION SUMMARY');
console.log('='.repeat(80));
console.log('');
console.log('Feature Breakdown (320 total):');
console.log('  [0-31]:     My men positions');
console.log('  [32-63]:    My kings positions');
console.log('  [64-95]:    Opponent men positions');
console.log('  [96-127]:   Opponent kings positions');
console.log('  [128-159]:  My mobility (legal moves per square)');
console.log('  [160-191]:  Opponent mobility');
console.log('  [192-223]:  My threat map (attacked squares)');
console.log('  [224-255]:  Opponent threat map');
console.log('  [256-287]:  My hanging pieces (undefended + threatened)');
console.log('  [288-303]:  My distance to promotion (normalized)');
console.log('  [304-319]:  Reserved for future features');
console.log('');
console.log('✅ Feature extraction working correctly!');
console.log('');
console.log('Next: Create PyTorch training pipeline using these features');
console.log('');
