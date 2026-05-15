#!/usr/bin/env tsx
/**
 * Test adaptive time management system
 */

import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';
import { allocateTime, createTimeManagerState, TimeManagerState } from '../src/coreClaude/search/timeManager';

// Test positions
const initialPosition: Position = {
  side: 1,
  p1Men: B1(24) | B1(25) | B1(26) | B1(27) | B1(28) | B1(29) | B1(30) | B1(31),
  p1Kings: 0,
  p2Men: B1(0) | B1(1) | B1(2) | B1(3) | B1(4) | B1(5) | B1(6) | B1(7),
  p2Kings: 0,
  halfmoveClock: 0,
};

const midgamePosition: Position = {
  side: 1,
  p1Men: B1(17) | B1(18) | B1(21),
  p1Kings: B1(10),
  p2Men: B1(5) | B1(6) | B1(9),
  p2Kings: B1(14),
  halfmoveClock: 0,
};

const endgamePosition: Position = {
  side: 1,
  p1Men: 0,
  p1Kings: B1(10) | B1(14),
  p2Men: 0,
  p2Kings: B1(5),
  halfmoveClock: 0,
};

const complexPosition: Position = {
  side: 1,
  p1Men: B1(13) | B1(14) | B1(17) | B1(18) | B1(21) | B1(22),
  p1Kings: B1(10),
  p2Men: B1(5) | B1(6) | B1(9) | B1(10) | B1(11),
  p2Kings: B1(2),
  halfmoveClock: 0,
};

function testTimeAllocation() {
  console.log('='.repeat(80));
  console.log('ADAPTIVE TIME MANAGEMENT TEST');
  console.log('='.repeat(80));
  console.log('');

  const baseTimeMs = 1000; // 1 second base time
  const state: TimeManagerState = createTimeManagerState(60000, 40); // 60 seconds total, 40 moves expected

  console.log(`Base time per move: ${baseTimeMs}ms`);
  console.log(`Total time remaining: ${state.totalTimeMs}ms (${state.totalTimeMs / 1000}s)`);
  console.log(`Expected moves remaining: ${state.expectedMovesRemaining}`);
  console.log('');

  const testCases = [
    { name: 'Initial Position (Opening)', pos: initialPosition },
    { name: 'Midgame Position', pos: midgamePosition },
    { name: 'Endgame Position', pos: endgamePosition },
    { name: 'Complex Tactical Position', pos: complexPosition },
  ];

  for (const testCase of testCases) {
    console.log('-'.repeat(80));
    console.log(testCase.name);
    console.log('-'.repeat(80));

    const allocation = allocateTime(testCase.pos, state, baseTimeMs);

    console.log(`Phase:       ${allocation.phase}`);
    console.log(`Complexity:  ${(allocation.complexity * 100).toFixed(0)}%`);
    console.log(`Target time: ${allocation.targetMs.toFixed(0)}ms`);
    console.log(`Min time:    ${allocation.minMs.toFixed(0)}ms`);
    console.log(`Max time:    ${allocation.maxMs.toFixed(0)}ms`);

    const ratio = allocation.targetMs / baseTimeMs;
    console.log(`Ratio:       ${ratio.toFixed(2)}x base time`);

    if (allocation.targetMs < baseTimeMs * 0.5) {
      console.log(`✓ Fast move (${ratio.toFixed(2)}x) - saves time`);
    } else if (allocation.targetMs > baseTimeMs * 1.5) {
      console.log(`⚠ Critical move (${ratio.toFixed(2)}x) - needs more time`);
    } else {
      console.log(`→ Normal move (${ratio.toFixed(2)}x)`);
    }

    console.log('');
  }

  console.log('='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log('');
  console.log('Time Management Strategy:');
  console.log('  • Opening (with book):  ~10% of base time (very fast)');
  console.log('  • Opening (no book):    ~70% of base time (normal)');
  console.log('  • Midgame:              ~120% of base time (more time)');
  console.log('  • Endgame:              ~80% of base time (less time)');
  console.log('  • Complexity modifier:  0.5x to 1.5x');
  console.log('  • Min time:             5% of base time');
  console.log('  • Max time:             2.5x base time');
  console.log('');
  console.log('Expected Benefits:');
  console.log('  ✓ Faster moves in opening (with 973-entry book)');
  console.log('  ✓ More time for critical midgame positions');
  console.log('  ✓ Balanced endgame (tablebase handles simple cases)');
  console.log('  ✓ Overall: +30-50 ELO strength gain');
  console.log('');
}

testTimeAllocation();
