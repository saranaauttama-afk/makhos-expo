// scripts/matchup.ts — head-to-head: Enhanced (book + engine) vs Base (engine only)
//
// Run with:
//   npx tsx --tsconfig tsconfig.test.json scripts/matchup.ts

import { iterativeDeepening } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { generateMoves, applyMove, Move } from '../src/coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../src/coreClaude/position';
import { hashPosition } from '../src/coreClaude/search/zobrist';
import { lookupOpeningBook } from '../src/coreClaude/search/openingBook';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';

const THINK_MS  = 600;   // ms per move (same budget for both)
const NUM_GAMES = 10;    // play N games, alternating who goes first
const MAX_PLIES = 300;   // safety cap

// ── Move selection ────────────────────────────────────────────────────────────

async function pickMove(
  pos: Position,
  history: number[],
  tt: TT,
  useBook: boolean,
): Promise<Move | undefined> {
  if (useBook) {
    const hit = lookupOpeningBook(pos, { source: 'matchup' });
    if (hit) return hit.move;
  }
  const res = await iterativeDeepening(pos, THINK_MS, tt, undefined, history);
  return res.best;
}

// ── Single game ───────────────────────────────────────────────────────────────

type GameResult = 'enhanced' | 'base' | 'draw';

async function playGame(enhancedSide: 1 | -1, gameNum: number): Promise<GameResult> {
  let pos = initialPosition();
  const history: number[] = [hashPosition(pos)];
  const ttEnh  = new TT();
  const ttBase = new TT();

  process.stdout.write(`  Game ${String(gameNum).padStart(2)}: Enhanced=${enhancedSide === 1 ? 'P1' : 'P2'} `);

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const moves = generateMoves(pos);

    // Terminal checks
    if (!moves.length) {
      const winner: GameResult = pos.side === enhancedSide ? 'base' : 'enhanced';
      console.log(`→ ${winner} wins at ply ${ply + 1} (no moves)`);
      return winner;
    }
    if (isDrawByInactivity(pos)) {
      console.log(`→ draw at ply ${ply + 1} (inactivity)`);
      return 'draw';
    }
    const repCounts = buildRepetitionCounts(history);
    if (isThreefoldRepetition(repCounts, hashPosition(pos))) {
      console.log(`→ draw at ply ${ply + 1} (threefold)`);
      return 'draw';
    }

    const isEnhanced = pos.side === enhancedSide;
    const tt   = isEnhanced ? ttEnh : ttBase;
    const move = await pickMove(pos, history, tt, isEnhanced);

    if (!move) {
      const pieces = (pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings).toString(2).replace(/0/g,'').length;
      const legalMoves = generateMoves(pos).length;
      console.log(`→ no move at ply ${ply + 1}  pieces=${pieces}  legalMoves=${legalMoves}  side=${pos.side}`);
      return 'draw';
    }

    pos = applyMove(pos, move);
    history.push(hashPosition(pos));
  }

  console.log(`→ draw (max plies)`);
  return 'draw';
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nMatchup: Enhanced (book+engine) vs Base (engine only)`);
  console.log(`Think: ${THINK_MS}ms/move  |  Games: ${NUM_GAMES}\n`);

  let enhW = 0, baseW = 0, draws = 0;

  for (let g = 0; g < NUM_GAMES; g++) {
    // Alternate which side Enhanced plays so results aren't biased by going first
    const enhancedSide: 1 | -1 = g % 2 === 0 ? 1 : -1;
    const result = await playGame(enhancedSide, g + 1);
    if (result === 'enhanced') enhW++;
    else if (result === 'base') baseW++;
    else draws++;
  }

  const total = enhW + baseW + draws;
  const pct = (n: number) => ((n / total) * 100).toFixed(0) + '%';

  console.log(`\n${'═'.repeat(48)}`);
  console.log('RESULTS');
  console.log(`${'─'.repeat(48)}`);
  console.log(`Enhanced (book+engine) : ${String(enhW).padStart(3)} wins  (${pct(enhW)})`);
  console.log(`Base (engine only)     : ${String(baseW).padStart(3)} wins  (${pct(baseW)})`);
  console.log(`Draws                  : ${String(draws).padStart(3)}        (${pct(draws)})`);
  console.log(`${'─'.repeat(48)}`);
  const verdict = enhW > baseW ? '✓ Book helps'
    : baseW > enhW ? '✗ Book hurts (or book moves are bad)'
    : '~ No clear difference';
  console.log(`Verdict: ${verdict}`);
  console.log(`${'═'.repeat(48)}\n`);
}

main().catch(console.error);
