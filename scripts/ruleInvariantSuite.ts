// scripts/ruleInvariantSuite.ts
//
// Rule-level regression checks for Thai Checkers move generation and applyMove.
// Run with:
//   npm run test:rules

import { B1, bitCount } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, hasCapturesAvailable, Move } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';

function makePosition(fields: Partial<Position> & Pick<Position, 'side'>): Position {
  return {
    p1Men: 0,
    p1Kings: 0,
    p2Men: 0,
    p2Kings: 0,
    halfmoveClock: 0,
    ...fields,
  };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function allPieces(pos: Position): number {
  return (pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings) >>> 0;
}

function sidePieces(pos: Position): number {
  return pos.side === 1
    ? ((pos.p1Men | pos.p1Kings) >>> 0)
    : ((pos.p2Men | pos.p2Kings) >>> 0);
}

function opponentPieces(pos: Position): number {
  return pos.side === 1
    ? ((pos.p2Men | pos.p2Kings) >>> 0)
    : ((pos.p1Men | pos.p1Kings) >>> 0);
}

function moveLabel(move: Move): string {
  const caps = move.captured.length ? `x${move.captured.join(',')}` : '-';
  const path = move.path?.length ? ` path=${move.path.join(',')}` : '';
  return `${move.from}->${move.to} caps=${caps}${move.promote ? ' promote' : ''}${path}`;
}

function moveIdentity(move: Move): string {
  return [
    move.from,
    move.to,
    move.promote ? 1 : 0,
    move.captured.join(','),
    move.path?.join(',') ?? '',
  ].join('|');
}

function assertNoOverlap(pos: Position, label: string) {
  const masks = [
    ['p1Men', pos.p1Men],
    ['p1Kings', pos.p1Kings],
    ['p2Men', pos.p2Men],
    ['p2Kings', pos.p2Kings],
  ] as const;

  for (let i = 0; i < masks.length; i++) {
    for (let j = i + 1; j < masks.length; j++) {
      assert((masks[i][1] & masks[j][1]) === 0, `${label}: ${masks[i][0]} overlaps ${masks[j][0]}`);
    }
  }
}

function validateMove(pos: Position, move: Move, label: string) {
  assert(move.from >= 0 && move.from < 32, `${label}: invalid from in ${moveLabel(move)}`);
  assert(move.to >= 0 && move.to < 32, `${label}: invalid to in ${moveLabel(move)}`);
  assert((sidePieces(pos) & B1(move.from)) !== 0, `${label}: from square has no moving-side piece in ${moveLabel(move)}`);
  assert((allPieces(pos) & B1(move.to)) === 0, `${label}: destination is occupied in ${moveLabel(move)}`);

  const seenCaptures = new Set<number>();
  for (const captured of move.captured) {
    assert(captured >= 0 && captured < 32, `${label}: invalid captured square in ${moveLabel(move)}`);
    assert(!seenCaptures.has(captured), `${label}: duplicate captured square in ${moveLabel(move)}`);
    seenCaptures.add(captured);
    assert((opponentPieces(pos) & B1(captured)) !== 0, `${label}: captured square is not an opponent piece in ${moveLabel(move)}`);
  }

  if (move.path) {
    assert(move.path.length >= 1, `${label}: empty path array in ${moveLabel(move)}`);
    assert(move.path[move.path.length - 1] === move.to, `${label}: path does not end at destination in ${moveLabel(move)}`);
  }
}

function validateApply(pos: Position, move: Move, label: string) {
  const beforeTotal = bitCount(allPieces(pos));
  const next = applyMove(pos, move);
  assertNoOverlap(next, `${label} after ${moveLabel(move)}`);
  assert(next.side === (pos.side === 1 ? -1 : 1), `${label}: side did not toggle after ${moveLabel(move)}`);
  assert(bitCount(allPieces(next)) === beforeTotal - move.captured.length, `${label}: piece count mismatch after ${moveLabel(move)}`);

  for (const captured of move.captured) {
    assert((allPieces(next) & B1(captured)) === 0, `${label}: captured piece still exists after ${moveLabel(move)}`);
  }

  const moverMask = pos.side === 1
    ? ((next.p1Men | next.p1Kings) >>> 0)
    : ((next.p2Men | next.p2Kings) >>> 0);
  assert((moverMask & B1(move.to)) !== 0, `${label}: moved piece missing at destination after ${moveLabel(move)}`);

  if (move.captured.length > 0) {
    assert(next.halfmoveClock === 0, `${label}: capture did not reset halfmoveClock after ${moveLabel(move)}`);
  } else {
    assert(next.halfmoveClock === pos.halfmoveClock + 1, `${label}: quiet move did not increment halfmoveClock after ${moveLabel(move)}`);
  }

  if (move.promote) {
    const promotedKings = pos.side === 1 ? next.p1Kings : next.p2Kings;
    const promotedMen = pos.side === 1 ? next.p1Men : next.p2Men;
    assert((promotedKings & B1(move.to)) !== 0, `${label}: promote move did not create king after ${moveLabel(move)}`);
    assert((promotedMen & B1(move.to)) === 0, `${label}: promote move left man at destination after ${moveLabel(move)}`);
  }
}

function validatePosition(pos: Position, label: string): number {
  assertNoOverlap(pos, label);

  const moves = generateMoves(pos);
  const fastHasCapture = hasCapturesAvailable(pos);
  const fullHasCapture = moves.length > 0 && moves[0].captured.length > 0;
  assert(fastHasCapture === fullHasCapture, `${label}: fast capture detection disagrees with generateMoves`);

  const identities = new Set<string>();
  for (const move of moves) {
    validateMove(pos, move, label);
    validateApply(pos, move, label);
    const id = moveIdentity(move);
    assert(!identities.has(id), `${label}: duplicate move ${moveLabel(move)}`);
    identities.add(id);
  }

  if (fullHasCapture) {
    const maxCaps = Math.max(...moves.map(move => move.captured.length));
    for (const move of moves) {
      assert(move.captured.length === maxCaps, `${label}: non-max capture leaked through ${moveLabel(move)}`);
    }
  } else {
    for (const move of moves) {
      assert(move.captured.length === 0, `${label}: capture mixed with quiet move ${moveLabel(move)}`);
    }
  }

  return moves.length;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function runCuratedPositions(): number {
  const positions: Array<[string, Position, (moves: Move[]) => void]> = [
    ['initial', initialPosition(), moves => {
      assert(moves.length > 0, 'initial: expected legal moves');
      assert(!hasCapturesAvailable(initialPosition()), 'initial: expected no capture');
    }],
    ['forced double capture', makePosition({
      side: 1,
      p1Men: B1(22) | B1(30),
      p2Men: B1(17) | B1(9),
    }), moves => {
      assert(moves.length === 1, `forced double capture: expected one move, got ${moves.map(moveLabel).join(' | ')}`);
      assert(moves[0].from === 22 && moves[0].to === 6 && moves[0].captured.length === 2, 'forced double capture: wrong chain');
    }],
    ['max capture filter', makePosition({
      side: 1,
      p1Men: B1(22) | B1(25),
      p2Men: B1(17) | B1(9) | B1(20),
    }), moves => {
      assert(moves.length >= 1, 'max capture filter: expected at least one capture');
      assert(moves.every(move => move.captured.length === 2), `max capture filter: shorter capture was returned ${moves.map(moveLabel).join(' | ')}`);
    }],
    ['king fly immediate landing', makePosition({
      side: 1,
      p1Kings: B1(22),
      p2Men: B1(17),
    }), moves => {
      assert(moves.some(move => move.from === 22 && move.to === 13 && move.captured[0] === 17), `king fly immediate landing: capture not found ${moves.map(moveLabel).join(' | ')}`);
    }],
    ['promotion quiet move', makePosition({
      side: 1,
      p1Men: B1(4),
      p2Men: B1(31),
    }), moves => {
      assert(moves.some(move => move.promote), `promotion quiet move: no promoting move found ${moves.map(moveLabel).join(' | ')}`);
    }],
    ['p2 promotion quiet move', makePosition({
      side: -1,
      p1Men: B1(0),
      p2Men: B1(27),
    }), moves => {
      assert(moves.some(move => move.promote), `p2 promotion quiet move: no promoting move found ${moves.map(moveLabel).join(' | ')}`);
    }],
    ['blocked side has no move', makePosition({
      side: 1,
      p1Men: B1(0),
      p2Men: B1(4),
    }), moves => {
      assert(moves.length === 0, `blocked side has no move: expected terminal no-move, got ${moves.map(moveLabel).join(' | ')}`);
    }],
  ];

  let checks = 0;
  for (const [label, pos, extra] of positions) {
    const movesCount = validatePosition(pos, label);
    extra(generateMoves(pos));
    checks += 1 + movesCount;
  }
  return checks;
}

function runRandomPlayouts(): number {
  const random = mulberry32(0xA110CA7E);
  let checks = 0;

  for (let game = 0; game < 80; game++) {
    let pos = initialPosition();
    for (let ply = 0; ply < 120; ply++) {
      const label = `random game ${game + 1} ply ${ply + 1}`;
      const moveCount = validatePosition(pos, label);
      checks++;
      if (moveCount === 0) break;

      const moves = generateMoves(pos);
      const move = moves[Math.floor(random() * moves.length)];
      pos = applyMove(pos, move);
    }
  }

  return checks;
}

function main() {
  const curatedChecks = runCuratedPositions();
  const randomChecks = runRandomPlayouts();
  console.log(`ruleInvariantSuite: ${curatedChecks + randomChecks} checks passed`);
}

main();
