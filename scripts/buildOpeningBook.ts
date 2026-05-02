// scripts/buildOpeningBook.ts
//
// Generates opening book entries by running the engine at each position.
// BFS from the initial position up to MAX_PLY plies:
//   - AI positions  (side = AI_SIDE):  run full engine → record hash→move
//   - Opp positions (side = OPP_SIDE): take top-N legal moves as candidate lines
//
// Usage:
//   npx tsx --tsconfig tsconfig.test.json scripts/buildOpeningBook.ts
//
// When done, paste the printed BOOK constant into openingBook.ts.

import { initialPosition } from '../src/coreClaude/position';
import { generateMoves, applyMove, Move } from '../src/coreClaude/movegen';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { iterativeDeepening, SearchResult } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';

// ── Config ────────────────────────────────────────────────────────────────────
// AI plays as P2 (side = -1) in the default app setup.
const AI_SIDE      = -1 as const;
const MAX_PLY      = 14;   // plies from start (7 rounds each)
const TOP_N        = 3;    // opponent's top-N responses to cover (captures + first quiet)
const BOOK_TOP_K   = 4;    // candidate moves stored per AI position
const BOOK_MARGIN  = 55;   // centipawn-ish margin from best move to keep in book
const AI_THINK_MS  = 1000; // ms for AI's book move — use high value for quality

// ── BFS ──────────────────────────────────────────────────────────────────────
interface QueueItem { pos: ReturnType<typeof initialPosition>; ply: number; }
interface BookCandidate { move: Move; score: number; weight: number; }
interface BookRow { from: number; to: number; score: number; weight: number; }

function topNMoves(pos: ReturnType<typeof initialPosition>, n: number) {
  const moves = generateMoves(pos);
  // Captures are forced in Thai checkers — always include them first.
  // Then take first (n - captures) quiet moves so we cover the likeliest lines.
  const caps = moves.filter(m => m.captured.length > 0);
  const quiet = moves.filter(m => m.captured.length === 0);
  return [...caps, ...quiet].slice(0, n);
}

function candidateWeight(score: number, bestScore: number, index: number): number {
  const gap = Math.max(0, bestScore - score);
  const rankDampener = 1 + index * 0.15;
  return Math.max(1, Math.round((100 * Math.exp(-gap / 32)) / rankDampener));
}

function bookCandidatesFromSearch(result: SearchResult): BookCandidate[] {
  const raw = result.rootCandidates?.length
    ? result.rootCandidates
    : result.best
      ? [{ move: result.best, score: result.score }]
      : [];
  const sorted = [...raw].sort((a, b) => b.score - a.score);
  const bestScore = sorted[0]?.score;
  if (bestScore == null) return [];

  return sorted
    .filter(candidate => candidate.score >= bestScore - BOOK_MARGIN)
    .slice(0, BOOK_TOP_K)
    .map((candidate, index) => ({
      move: candidate.move,
      score: candidate.score,
      weight: candidateWeight(candidate.score, bestScore, index),
    }));
}

async function buildBook() {
  const tt       = new TT();
  const book     = new Map<number, BookRow[]>();
  const visited  = new Set<number>();
  const queue: QueueItem[] = [{ pos: initialPosition(), ply: 0 }];

  let processed = 0;
  const start = Date.now();
  console.log(`Building opening book  MAX_PLY=${MAX_PLY}  TOP_N=${TOP_N}  AI_THINK_MS=${AI_THINK_MS}`);

  while (queue.length > 0) {
    const { pos, ply } = queue.shift()!;
    const hash = hashPosition(pos);

    if (ply >= MAX_PLY) continue;
    if (visited.has(hash))  continue;
    visited.add(hash);

    const moves = generateMoves(pos);
    if (!moves.length) continue;

    if (pos.side === AI_SIDE) {
      // ── AI's turn: compute best move and record it ────────────────────────
      const result = await iterativeDeepening(
        pos, AI_THINK_MS, tt, () => {}, [], { cancelled: false },
      );
      const candidates = bookCandidatesFromSearch(result);
      if (!candidates.length) continue;

      book.set(hash, candidates.map(candidate => ({
        from: candidate.move.from,
        to: candidate.move.to,
        score: candidate.score,
        weight: candidate.weight,
      })));
      processed++;
      process.stdout.write(
        `\r  ply ${ply} | entries ${processed} | queue ${queue.length} | candidates ${candidates.length}   `,
      );

      // After AI's move, opponent faces a position — expand it
      queue.push({ pos: applyMove(pos, candidates[0].move), ply: ply + 1 });

    } else {
      // ── Opponent's turn: branch into top-N likely responses ───────────────
      for (const m of topNMoves(pos, TOP_N)) {
        queue.push({ pos: applyMove(pos, m), ply: ply + 1 });
      }
    }
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`\n\nDone — ${book.size} entries in ${elapsed}s\n`);

  // ── Output TypeScript ─────────────────────────────────────────────────────
  console.log('// Paste this into src/coreClaude/search/openingBook.ts');
  console.log('// [hash, from, to, weight, score]');
  console.log('const BOOK: BookRow[] = [');
  for (const [hash, candidates] of book) {
    for (const move of candidates) {
      console.log(`  [${hash}, ${move.from}, ${move.to}, ${move.weight}, ${move.score}],`);
    }
  }
  console.log('];');
}

buildBook().catch(console.error);
