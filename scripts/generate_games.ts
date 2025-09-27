import * as fs from 'fs';
import { Position, initialPosition, isDrawByInactivity } from '../src/core/position';
import { generateMoves, applyMove, Move } from '../src/core/movegen';
import { iterativeDeepening } from '../src/core/search/alphabeta';
import { TT } from '../src/core/search/tt';
import { bitCount } from '../src/core/bitboards';
import { evaluate } from '../src/core/eval';

interface PositionData {
  state: number[];
  legalMoves: number[][];
  policyTarget: number[];
  searchDepth: number;
  searchScore: number;
  searchNodes: number;
  evaluation: number;
}

interface GameRecord {
  positions: PositionData[];
  result: number;
}

function positionToArray(p: Position): number[] {
  return [
    p.p1Men >>> 0,
    p.p1Kings >>> 0,
    p.p2Men >>> 0,
    p.p2Kings >>> 0,
    p.side,
    p.halfmoveClock
  ];
}

function moveToArray(m: Move): number[] {
  return [m.from, m.to, m.captured.length, m.promote ? 1 : 0];
}

function moveToPolicyArray(m: Move): number[] {
  const policy = new Array(32 * 32).fill(0);
  const idx = m.from * 32 + m.to;
  policy[idx] = 1.0;
  return policy;
}

function playOneGame(timePerMove: number): GameRecord | null {
  const positions: PositionData[] = [];

  let pos = initialPosition();
  const tt = new TT();
  let plyCount = 0;
  const MAX_PLIES = 200;

  while (plyCount < MAX_PLIES) {
    if (isDrawByInactivity(pos)) {
      return { positions, result: 0 };
    }

    const legalMoves = generateMoves(pos);
    if (legalMoves.length === 0) {
      const result = pos.side === 1 ? -1 : 1;
      return { positions, result };
    }

    const searchResult = iterativeDeepening(pos, timePerMove, tt);
    if (!searchResult.best) {
      console.error('No move found');
      return null;
    }

    const posData: PositionData = {
      state: positionToArray(pos),
      legalMoves: legalMoves.map(moveToArray),
      policyTarget: moveToPolicyArray(searchResult.best),
      searchDepth: searchResult.depth,
      searchScore: searchResult.score,
      searchNodes: searchResult.nodes,
      evaluation: evaluate(pos)
    };

    positions.push(posData);

    pos = applyMove(pos, searchResult.best);
    plyCount++;
  }

  return { positions, result: 0 };
}

function main() {
  const args = process.argv.slice(2);
  const numGames = parseInt(args[0] || '100');
  const timePerMove = parseInt(args[1] || '500');
  const outputFile = args[2] || 'games_data.json';

  console.log(`Generating ${numGames} games with ${timePerMove}ms per move...`);

  const allGames: GameRecord[] = [];

  for (let i = 0; i < numGames; i++) {
    console.log(`Game ${i + 1}/${numGames}...`);
    const game = playOneGame(timePerMove);
    if (game) {
      allGames.push(game);
      const resultStr = game.result === 1 ? 'P1 wins' : game.result === -1 ? 'P2 wins' : 'Draw';
      const avgDepth = game.positions.reduce((s, p) => s + p.searchDepth, 0) / game.positions.length;
      console.log(`  ${resultStr} (${game.positions.length} moves, avg depth: ${avgDepth.toFixed(1)})`);
    }
  }

  fs.writeFileSync(outputFile, JSON.stringify(allGames, null, 2));
  console.log(`\nSaved ${allGames.length} games to ${outputFile}`);
}

main();