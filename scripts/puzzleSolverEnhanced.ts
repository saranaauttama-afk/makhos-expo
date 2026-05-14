// Enhanced Puzzle Solver with Deeper Search
//
// This version uses significantly deeper search and longer time limits
// specifically tuned for tactical puzzle solving.

import { TACTICAL_PUZZLES, PuzzleFixture, PuzzleDifficulty } from './puzzleFixtures';
import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';

interface PuzzleResult {
  puzzle: PuzzleFixture;
  solved: boolean;
  chosenMove?: { from: number; to: number };
  score: number;
  depth: number;
  timeMs: number;
  correct: boolean;
}

interface PuzzleSummary {
  total: number;
  solved: number;
  correct: number;
  byDifficulty: Record<PuzzleDifficulty, { total: number; solved: number; correct: number }>;
  avgTimeMs: number;
  avgDepth: number;
}

// Puzzle-specific search configuration
const PUZZLE_CONFIG = {
  easy: { maxDepth: 12, timeLimitMs: 15000 },
  medium: { maxDepth: 16, timeLimitMs: 30000 },
  hard: { maxDepth: 20, timeLimitMs: 60000 },
  expert: { maxDepth: 24, timeLimitMs: 120000 }, // 2 minutes for expert puzzles
};

async function solvePuzzle(puzzle: PuzzleFixture): Promise<PuzzleResult> {
  const config = PUZZLE_CONFIG[puzzle.difficulty];
  const startTime = Date.now();
  const tt = new TT();

  try {
    const result = await iterativeDeepening(
      puzzle.pos,
      config.timeLimitMs,
      tt,
      undefined,
      [],
      undefined,
      config.maxDepth
    );

    const timeMs = Date.now() - startTime;
    const solved = result.best != null;
    const correct = puzzle.expectedMove
      ? (result.best?.from === puzzle.expectedMove.from && result.best?.to === puzzle.expectedMove.to)
      : solved;

    return {
      puzzle,
      solved,
      chosenMove: result.best ? { from: result.best.from, to: result.best.to } : undefined,
      score: result.score,
      depth: result.depth,
      timeMs,
      correct,
    };
  } catch (error) {
    const timeMs = Date.now() - startTime;
    return {
      puzzle,
      solved: false,
      score: 0,
      depth: 0,
      timeMs,
      correct: false,
    };
  }
}

function summarizeResults(results: PuzzleResult[]): PuzzleSummary {
  const byDifficulty: Record<PuzzleDifficulty, { total: number; solved: number; correct: number }> = {
    easy: { total: 0, solved: 0, correct: 0 },
    medium: { total: 0, solved: 0, correct: 0 },
    hard: { total: 0, solved: 0, correct: 0 },
    expert: { total: 0, solved: 0, correct: 0 },
  };

  let totalTime = 0;
  let totalDepth = 0;

  for (const result of results) {
    const diff = result.puzzle.difficulty;
    byDifficulty[diff].total++;
    if (result.solved) byDifficulty[diff].solved++;
    if (result.correct) byDifficulty[diff].correct++;
    totalTime += result.timeMs;
    totalDepth += result.depth;
  }

  return {
    total: results.length,
    solved: results.filter(r => r.solved).length,
    correct: results.filter(r => r.correct).length,
    byDifficulty,
    avgTimeMs: totalTime / results.length,
    avgDepth: totalDepth / results.length,
  };
}

function formatMove(move: { from: number; to: number } | undefined): string {
  if (!move) return 'none';
  return `${move.from}->${move.to}`;
}

async function main() {
  console.log('='.repeat(80));
  console.log('ENHANCED THAI CHECKERS PUZZLE SOLVER');
  console.log('Deep Search Mode - Optimized for Tactical Puzzles');
  console.log('='.repeat(80));
  console.log('');
  console.log(`Total puzzles: ${TACTICAL_PUZZLES.length}`);
  console.log('');
  console.log('Configuration:');
  console.log(`  Easy:   depth=${PUZZLE_CONFIG.easy.maxDepth}, time=${PUZZLE_CONFIG.easy.timeLimitMs}ms`);
  console.log(`  Medium: depth=${PUZZLE_CONFIG.medium.maxDepth}, time=${PUZZLE_CONFIG.medium.timeLimitMs}ms`);
  console.log(`  Hard:   depth=${PUZZLE_CONFIG.hard.maxDepth}, time=${PUZZLE_CONFIG.hard.timeLimitMs}ms`);
  console.log(`  Expert: depth=${PUZZLE_CONFIG.expert.maxDepth}, time=${PUZZLE_CONFIG.expert.timeLimitMs}ms`);
  console.log('');

  const results: PuzzleResult[] = [];

  for (let i = 0; i < TACTICAL_PUZZLES.length; i++) {
    const puzzle = TACTICAL_PUZZLES[i];
    const config = PUZZLE_CONFIG[puzzle.difficulty];
    process.stdout.write(`[${i + 1}/${TACTICAL_PUZZLES.length}] ${puzzle.name} (${puzzle.difficulty}, depth≤${config.maxDepth})... `);

    const result = await solvePuzzle(puzzle);
    results.push(result);

    const status = result.correct ? '✓' : '✗';
    const moveStr = formatMove(result.chosenMove);
    const expectedStr = formatMove(puzzle.expectedMove);
    const comparison = result.correct ? '' : `(expected ${expectedStr})`;
    console.log(`${status} ${moveStr} ${comparison} [${result.timeMs}ms, depth=${result.depth}]`);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));

  const summary = summarizeResults(results);

  console.log('');
  console.log(`Overall: ${summary.correct}/${summary.total} correct (${Math.round(summary.correct / summary.total * 100)}%)`);
  console.log(`Solved: ${summary.solved}/${summary.total} (${Math.round(summary.solved / summary.total * 100)}%)`);
  console.log(`Average time: ${Math.round(summary.avgTimeMs)}ms (${(summary.avgTimeMs / 1000).toFixed(1)}s)`);
  console.log(`Average depth: ${summary.avgDepth.toFixed(1)}`);
  console.log('');

  console.log('By Difficulty:');
  for (const diff of ['easy', 'medium', 'hard', 'expert'] as PuzzleDifficulty[]) {
    const stats = summary.byDifficulty[diff];
    if (stats.total === 0) continue;
    const pct = Math.round(stats.correct / stats.total * 100);
    console.log(`  ${diff.padEnd(8)}: ${stats.correct}/${stats.total} correct (${pct}%)`);
  }

  console.log('');

  // Show improvement from baseline
  const baselineCorrect = 1; // From PUZZLE_BASELINE.md
  const improvement = summary.correct - baselineCorrect;
  if (improvement > 0) {
    console.log(`🎉 Improvement: +${improvement} puzzles solved! (${baselineCorrect}→${summary.correct})`);
  } else if (improvement === 0) {
    console.log(`No improvement yet (still at ${baselineCorrect}/${summary.total})`);
  } else {
    console.log(`⚠️  Regression: ${improvement} fewer puzzles solved`);
  }
  console.log('');

  console.log('Failed Puzzles:');
  const failed = results.filter(r => !r.correct);
  if (failed.length === 0) {
    console.log('  None - all puzzles solved correctly! 🎉🎉🎉');
  } else {
    for (const result of failed) {
      console.log(`  - ${result.puzzle.id}: ${result.puzzle.name} (${result.puzzle.difficulty})`);
      console.log(`    Expected: ${formatMove(result.puzzle.expectedMove)}`);
      console.log(`    Chose: ${formatMove(result.chosenMove)}`);
      console.log(`    Depth reached: ${result.depth}`);
      console.log(`    Type: ${result.puzzle.type}`);
      console.log(`    Note: ${result.puzzle.note || 'none'}`);
    }
  }

  console.log('');
  console.log('='.repeat(80));

  // Save results to JSON
  const reportPath = '.tmp/puzzles/puzzle-results-enhanced.json';
  const fs = await import('fs');
  fs.mkdirSync('.tmp/puzzles', { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    mode: 'enhanced',
    configuration: PUZZLE_CONFIG,
    summary,
    results: results.map(r => ({
      id: r.puzzle.id,
      name: r.puzzle.name,
      difficulty: r.puzzle.difficulty,
      type: r.puzzle.type,
      solved: r.solved,
      correct: r.correct,
      chosenMove: r.chosenMove,
      expectedMove: r.puzzle.expectedMove,
      score: r.score,
      depth: r.depth,
      timeMs: r.timeMs,
    })),
  }, null, 2));

  console.log(`\nResults saved to: ${reportPath}`);
}

main().catch(console.error);
