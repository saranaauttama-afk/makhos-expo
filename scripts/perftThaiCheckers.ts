// scripts/perftThaiCheckers.ts
// Perft (Performance Test) for Thai Checkers - Phase 2
// Validates movegen correctness by counting leaf nodes at each depth

import { generateMoves, applyMove } from '../src/coreClaude/movegen';
import { Position, initialPosition } from '../src/coreClaude/position';
import { B1 } from '../src/coreClaude/bitboards';

// Count all leaf nodes at given depth
function perft(pos: Position, depth: number): number {
  if (depth === 0) return 1;

  const moves = generateMoves(pos);
  if (depth === 1) return moves.length;

  let nodes = 0;
  for (const move of moves) {
    const child = applyMove(pos, move);
    nodes += perft(child, depth - 1);
  }
  return nodes;
}

// Count nodes per root move (useful for debugging)
function divide(pos: Position, depth: number): Map<string, number> {
  const moves = generateMoves(pos);
  const results = new Map<string, number>();

  for (const move of moves) {
    const moveStr = `${move.from}->${move.to}${move.promote ? 'K' : ''}${move.captured.length > 0 ? `x${move.captured.length}` : ''}`;
    const child = applyMove(pos, move);
    const count = depth <= 1 ? 1 : perft(child, depth - 1);
    results.set(moveStr, count);
  }

  return results;
}

interface PerftTest {
  name: string;
  pos: Position;
  depths: number[];
  expected: number[];
}

// Test suite
const tests: PerftTest[] = [
  // Test 1: Initial position (8v8)
  {
    name: 'Initial Position (8v8)',
    pos: initialPosition(),
    depths: [1, 2, 3],
    expected: [7, 49, 392], // Verified by perft
  },

  // Test 2: Forced capture (must capture when available)
  {
    name: 'Forced Capture',
    pos: {
      side: 1,
      p1Men: B1(18), // P1 man at 18
      p1Kings: 0,
      p2Men: B1(14), // P2 man at 14 (diagonal UL from 18)
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [1], // Must have exactly 1 capture move 18x14->9
  },

  // Test 3: Max-capture rule (must take longest capture chain)
  {
    name: 'Max-Capture Rule',
    pos: {
      side: 1,
      p1Men: B1(23), // P1 man
      p1Kings: 0,
      p2Men: B1(18) | B1(14) | B1(13), // Three P2 pieces forming chain
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [1], // Only the max-length capture (must ignore shorter ones)
  },

  // Test 4: Men multi-capture
  {
    name: 'Men Multi-Capture',
    pos: {
      side: 1,
      p1Men: B1(27), // P1 man at (r=6, c=7)
      p1Kings: 0,
      p2Men: B1(23) | B1(14), // Two P2 men on UL diagonal: 23 at (r=5, c=6), 14 at (r=3, c=4)
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1, 2],
    expected: [1, 0], // Captures both 23 and 14, lands at 9, terminal (P2 has 0 pieces)
  },

  // Test 5: King fly capture (king can land anywhere after jump)
  {
    name: 'King Fly Capture',
    pos: {
      side: 1,
      p1Men: 0,
      p1Kings: B1(27), // P1 king at (r=6, c=6)
      p2Men: B1(22), // P2 man at (r=5, c=4) - diagonal UL from 27
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [7], // King can land on 23, 18, 14, 9, 5, 0, 31 after capturing 22
  },

  // Test 6: King multi-capture
  {
    name: 'King Multi-Capture',
    pos: {
      side: 1,
      p1Men: 0,
      p1Kings: B1(28), // P1 king
      p2Men: B1(23) | B1(14), // Two P2 men diagonal
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [1], // King captures both (multiple landing spots but counted as variations)
  },

  // Test 7: Promotion after capture (capturing onto back rank)
  {
    name: 'Promotion After Capture',
    pos: {
      side: 1,
      p1Men: B1(9), // P1 man at (r=2, c=3)
      p1Kings: 0,
      p2Men: B1(5) | B1(24) | B1(25) | B1(26) | B1(27), // P2: 5 to capture, 24-27 back rank safe
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [1], // Captures 5, promotes to king at 0 (just test promotion works)
  },

  // Test 8: Side -1 mirror (test P2 perspective)
  {
    name: 'Side -1 Mirror',
    pos: {
      side: -1, // P2 to move
      p1Men: B1(18), // P1 at (r=4, c=5)
      p1Kings: 0,
      p2Men: B1(14), // P2 at (r=3, c=4) - can capture downward
      p2Kings: 0,
      halfmoveClock: 0,
    },
    depths: [1],
    expected: [1], // P2 captures P1: 14x18->23
  },
];

async function main() {
  console.log('='.repeat(80));
  console.log('THAI CHECKERS PERFT TEST SUITE (Phase 2)');
  console.log('='.repeat(80));
  console.log('\nGoal: Validate movegen correctness after Phase 3 rewrite');
  console.log('Method: Count leaf nodes at each depth (perft)\n');

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    console.log(`\n${'─'.repeat(80)}`);
    console.log(`Test: ${test.name}`);
    console.log(`${'─'.repeat(80)}`);

    // Show position
    const p1Count = countBits(test.pos.p1Men) + countBits(test.pos.p1Kings);
    const p2Count = countBits(test.pos.p2Men) + countBits(test.pos.p2Kings);
    console.log(`Position: ${p1Count}v${p2Count}, side ${test.pos.side === 1 ? 'P1' : 'P2'} to move`);

    // Run perft for each depth
    let testPassed = true;
    for (let i = 0; i < test.depths.length; i++) {
      const depth = test.depths[i];
      const expected = test.expected[i];

      const startTime = Date.now();
      const result = perft(test.pos, depth);
      const elapsed = Date.now() - startTime;

      const status = result === expected ? '✓' : '✗';
      const match = result === expected;

      console.log(`  Depth ${depth}: ${result.toLocaleString()} nodes (expected ${expected.toLocaleString()}) ${status} [${elapsed}ms]`);

      if (!match) {
        testPassed = false;
        console.log(`    ERROR: Got ${result}, expected ${expected}`);

        // Show divide for debugging
        if (depth > 0) {
          console.log(`    Divide:`);
          const div = divide(test.pos, depth);
          for (const [move, count] of div) {
            console.log(`      ${move}: ${count.toLocaleString()}`);
          }
        }
      }
    }

    if (testPassed) {
      console.log(`✓ PASSED`);
      passed++;
    } else {
      console.log(`✗ FAILED`);
      failed++;
    }
  }

  console.log(`\n${'='.repeat(80)}`);
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Passed: ${passed}/${tests.length}`);
  console.log(`Failed: ${failed}/${tests.length}`);

  if (failed > 0) {
    console.log('\n⚠️  Some tests failed. Check movegen/bitboards implementation.');
    process.exit(1);
  } else {
    console.log('\n✓ All perft tests passed! Movegen correctness validated.');
    process.exit(0);
  }
}

function countBits(bb: number): number {
  let count = 0;
  while (bb) {
    count++;
    bb &= bb - 1;
  }
  return count;
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
