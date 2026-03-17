// selfplay.mts — AI vs AI self-play test (run with: npx tsx selfplay.mts)
import { iterativeDeepening } from './src/coreCodex/search/alphabeta';
import { TT } from './src/coreCodex/search/tt';
import { generateMoves } from './src/coreCodex/movegen';
import { applyMove, initialPosition, isDrawByInactivity } from './src/coreCodex/position';
import { hashPosition } from './src/coreCodex/search/zobrist';
import { bitCount } from './src/coreCodex/bitboards';

const THINK_MS  = 600;   // ms per move
const MAX_PLIES = 300;   // safety cap

type MoveRecord = {
  ply: number; side: string; moveStr: string;
  thinkMs: number; depth: number; score: number; nodes: number;
};

async function selfPlay() {
  let pos = initialPosition();
  const tt1 = new TT();
  const tt2 = new TT();
  const history: number[] = [hashPosition(pos)];
  const log: MoveRecord[] = [];
  let totalMs = 0;

  console.log(`\nMakhos AI self-play  (${THINK_MS} ms/move)\n${'─'.repeat(64)}`);
  console.log('Ply  Side   Move        Depth    Score      Nodes     Time');
  console.log('─'.repeat(64));

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const moves = generateMoves(pos);

    if (!moves.length) {
      const winner = pos.side === 1 ? 'P2 ▲' : 'P1 ●';
      console.log(`\n★ ${winner} wins (opponent has no moves)`);
      break;
    }
    if (isDrawByInactivity(pos)) {
      console.log('\n½ Draw — inactivity rule');
      break;
    }

    const sideName = pos.side === 1 ? 'P1 ●' : 'P2 ▲';
    const tt       = pos.side === 1 ? tt1 : tt2;

    const t0      = Date.now();
    const result  = await iterativeDeepening(pos, THINK_MS, tt, undefined, history);
    const elapsed = Date.now() - t0;
    totalMs      += elapsed;

    if (!result.best) {
      console.log(`\n${sideName} has no best move — game abandoned`);
      break;
    }

    const m = result.best;
    const capStr = m.captured.length ? `×${m.captured.length}` : '  ';
    const moveStr = `${String(m.from).padStart(2)}→${String(m.to).padStart(2)} ${capStr}`;
    const pieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);

    console.log(
      `${String(ply + 1).padStart(3)}  ${sideName}  ${moveStr}` +
      `  d${String(result.depth).padStart(2)}` +
      `  ${String(result.score).padStart(7)}cp` +
      `  ${String(result.nodes).padStart(8)}` +
      `  ${String(elapsed).padStart(5)}ms` +
      (pieces <= 8 ? '  [endgame]' : '')
    );

    log.push({ ply: ply + 1, side: sideName, moveStr, thinkMs: elapsed, depth: result.depth, score: result.score, nodes: result.nodes });
    pos = applyMove(pos, m);
    history.push(hashPosition(pos));
  }

  // ── Summary ────────────────────────────────────────────────────────────────
  const avgThink = totalMs / (log.length || 1);
  const maxThink = Math.max(...log.map(r => r.thinkMs));
  const avgDepth = log.reduce((s, r) => s + r.depth, 0) / (log.length || 1);
  const avgNodes = log.reduce((s, r) => s + r.nodes, 0) / (log.length || 1);
  const captures = log.filter(r => r.moveStr.includes('×')).length;
  const slowMoves = log.filter(r => r.thinkMs > avgThink * 2 && r.thinkMs > 300);

  console.log(`\n${'═'.repeat(64)}`);
  console.log('SUMMARY');
  console.log(`${'─'.repeat(64)}`);
  console.log(`Plies played     : ${log.length}  (P1: ${log.filter(r=>r.side.startsWith('P1')).length}, P2: ${log.filter(r=>r.side.startsWith('P2')).length})`);
  console.log(`Capture moves    : ${captures}`);
  console.log(`Total think time : ${(totalMs / 1000).toFixed(1)}s`);
  console.log(`Avg per move     : ${avgThink.toFixed(0)}ms`);
  console.log(`Max single move  : ${maxThink}ms`);
  console.log(`Avg search depth : ${avgDepth.toFixed(1)}`);
  console.log(`Avg nodes/move   : ${Math.round(avgNodes).toLocaleString()}`);
  if (slowMoves.length)
    console.log(`Slow moves (>2×avg): ${slowMoves.map(r => `ply${r.ply}(${r.thinkMs}ms d${r.depth})`).join(', ')}`);
  console.log(`${'═'.repeat(64)}\n`);
}

selfPlay().catch(console.error);
