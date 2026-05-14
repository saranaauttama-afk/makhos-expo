// Performance Profiling Script
// Measures time spent in key functions during search

import { initialPosition } from '../src/coreClaude/position';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { applyMove } from '../src/coreClaude/movegen';
import { generateMoves } from '../src/coreClaude/movegen';

interface ProfileData {
  name: string;
  totalTime: number;
  callCount: number;
  avgTime: number;
}

// Simple profiler
const profiles = new Map<string, { totalTime: number; callCount: number }>();

function profileStart(name: string): number {
  return performance.now();
}

function profileEnd(name: string, startTime: number) {
  const elapsed = performance.now() - startTime;
  const existing = profiles.get(name) || { totalTime: 0, callCount: 0 };
  profiles.set(name, {
    totalTime: existing.totalTime + elapsed,
    callCount: existing.callCount + 1,
  });
}

// Test positions
const testPositions = [
  { name: 'Initial position', pos: initialPosition() },
  {
    name: 'Midgame',
    pos: (() => {
      let p = initialPosition();
      // Make a few moves to get to midgame
      for (let i = 0; i < 6; i++) {
        const moves = generateMoves(p);
        if (moves.length > 0) p = applyMove(p, moves[0]);
      }
      return p;
    })()
  },
];

async function profileSearch() {
  console.log('='.repeat(80));
  console.log('PERFORMANCE PROFILING');
  console.log('='.repeat(80));
  console.log('');

  for (const test of testPositions) {
    console.log(`\nProfiling: ${test.name}`);
    console.log('-'.repeat(40));

    const tt = new TT();
    const startTotal = performance.now();

    const result = await iterativeDeepening(
      test.pos,
      2000, // 2 second search
      tt,
      undefined,
      [],
      undefined,
      6 // max depth
    );

    const totalTime = performance.now() - startTotal;

    console.log(`Total search time: ${totalTime.toFixed(1)}ms`);
    console.log(`Final depth: ${result.depth}`);
    console.log(`Nodes searched: ${result.nodes || 0}`);
    console.log(`NPS: ${Math.round((result.nodes || 0) / (totalTime / 1000))} nodes/sec`);
  }

  // Print profile summary
  console.log('');
  console.log('='.repeat(80));
  console.log('PROFILE SUMMARY');
  console.log('='.repeat(80));
  console.log('');

  const sorted = Array.from(profiles.entries())
    .map(([name, data]): ProfileData => ({
      name,
      totalTime: data.totalTime,
      callCount: data.callCount,
      avgTime: data.totalTime / data.callCount,
    }))
    .sort((a, b) => b.totalTime - a.totalTime);

  console.log('Function'.padEnd(40) + 'Time(ms)'.padEnd(12) + 'Calls'.padEnd(12) + 'Avg(ms)');
  console.log('-'.repeat(80));

  for (const prof of sorted.slice(0, 20)) {
    console.log(
      prof.name.padEnd(40) +
      prof.totalTime.toFixed(1).padEnd(12) +
      prof.callCount.toString().padEnd(12) +
      prof.avgTime.toFixed(3)
    );
  }

  console.log('');
  console.log('='.repeat(80));

  // Performance metrics
  console.log('\nPERFORMANCE METRICS');
  console.log('-'.repeat(40));

  const evalTime = profiles.get('eval')?.totalTime || 0;
  const movegenTime = profiles.get('movegen')?.totalTime || 0;
  const ttTime = profiles.get('tt-lookup')?.totalTime || 0;
  const totalProfiled = Array.from(profiles.values()).reduce((sum, p) => sum + p.totalTime, 0);

  console.log(`Eval time: ${evalTime.toFixed(1)}ms (${(evalTime / totalProfiled * 100).toFixed(1)}%)`);
  console.log(`Movegen time: ${movegenTime.toFixed(1)}ms (${(movegenTime / totalProfiled * 100).toFixed(1)}%)`);
  console.log(`TT lookup time: ${ttTime.toFixed(1)}ms (${(ttTime / totalProfiled * 100).toFixed(1)}%)`);
}

// Run profiling
profileSearch().catch(console.error);
