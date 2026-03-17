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

import { initialPosition } from '../src/coreCodex/position';
import { generateMoves, applyMove } from '../src/coreCodex/movegen';
import { hashPosition } from '../src/coreCodex/search/zobrist';
import { iterativeDeepening } from '../src/coreCodex/search/alphabeta';
import { TT } from '../src/coreCodex/search/tt';

// ── Config ────────────────────────────────────────────────────────────────────
// AI plays as P2 (side = -1) in the default app setup.
const AI_SIDE      = -1 as const;
const MAX_PLY      = 10;   // plies from start (5 rounds each)
const TOP_N        = 4;    // opponent's top-N responses to cover (captures + first quiet)
const AI_THINK_MS  = 1200; // ms for AI's book move — use high value for quality

// ── BFS ──────────────────────────────────────────────────────────────────────
interface QueueItem { pos: ReturnType<typeof initialPosition>; ply: number; }

function topNMoves(pos: ReturnType<typeof initialPosition>, n: number) {
  const moves = generateMoves(pos);
  // Captures are forced in Thai checkers — always include them first.
  // Then take first (n - captures) quiet moves so we cover the likeliest lines.
  const caps = moves.filter(m => m.captured.length > 0);
  const quiet = moves.filter(m => m.captured.length === 0);
  return [...caps, ...quiet].slice(0, n);
}

async function buildBook() {
  const tt       = new TT();
  const book     = new Map<number, { from: number; to: number }>();
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
      if (!result.best) continue;

      book.set(hash, { from: result.best.from, to: result.best.to });
      processed++;
      process.stdout.write(
        `\r  ply ${ply} | entries ${processed} | queue ${queue.length}   `,
      );

      // After AI's move, opponent faces a position — expand it
      queue.push({ pos: applyMove(pos, result.best), ply: ply + 1 });

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
  console.log('// Paste this into src/coreCodex/search/openingBook.ts');
  console.log('// [hash, from, to]');
  console.log('const BOOK: [number, number, number][] = [');
  for (const [hash, move] of book) {
    console.log(`  [${hash}, ${move.from}, ${move.to}],`);
  }
  console.log('];');
}

buildBook().catch(console.error);
