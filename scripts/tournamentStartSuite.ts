import { Move, applyMove, generateMoves } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';

export interface TournamentStart {
  id: string;
  /** Defaults to the standard initial position. Useful for explicit fixture suites. */
  initialPosition?: Position;
  position: Position;
  openingMoves: Move[];
}

export interface TournamentStartSuite {
  version: string;
  seed: number;
  generator: string;
  starts: TournamentStart[];
}

function moveKey(move: Move): string {
  return `${move.from}-${move.to}-${move.captured.join('.')}-${move.promote ? 1 : 0}`;
}

function positionKey(position: Position): string {
  return [position.side, position.p1Men >>> 0, position.p1Kings >>> 0,
    position.p2Men >>> 0, position.p2Kings >>> 0, position.halfmoveClock].join(':');
}

/** Frozen v1 pseudo-random legal playout generator. This is measurement
 * diversity, not an authoritative Thai opening book. */
export function generateTournamentStartSuite(seed = 0x4d414b48, count = 8): TournamentStartSuite {
  let state = seed >>> 0;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return state >>> 0;
  };
  const starts: TournamentStart[] = [];
  const seen = new Set<string>();
  for (let attempt = 0; starts.length < count && attempt < count * 20; attempt++) {
    let position = initialPosition();
    const openingMoves: Move[] = [];
    const plies = 2 + (attempt % 7);
    for (let ply = 0; ply < plies; ply++) {
      const legal = generateMoves(position).slice().sort((a, b) => moveKey(a).localeCompare(moveKey(b)));
      if (!legal.length) break;
      const move = legal[random() % legal.length];
      openingMoves.push(move);
      position = applyMove(position, move);
    }
    const key = positionKey(position);
    if (openingMoves.length === plies && !seen.has(key)) {
      seen.add(key);
      starts.push({ id: `makhos-measurement-v1-${String(starts.length + 1).padStart(2, '0')}`, position, openingMoves });
    }
  }
  if (starts.length !== count) throw new Error(`could only generate ${starts.length}/${count} unique starts`);
  return { version: 'makhos-measurement-starts-v1', seed: seed >>> 0,
    generator: 'xorshift32/legal-sorted/v1 (not an authoritative opening book)', starts };
}

export const TOURNAMENT_START_SUITE = generateTournamentStartSuite();

/** Phase 3A's frozen, larger legal-start corpus. Opening lengths deliberately
 * span early and developed positions; this is generated measurement data, not
 * an opening book and carries no opening-theory provenance. */
export function generateSearchAblationStartSuite(seed = 0x3341424c, count = 64): TournamentStartSuite {
  let state = seed >>> 0;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return state >>> 0;
  };
  const starts: TournamentStart[] = [];
  const seen = new Set<string>();
  for (let attempt = 0; starts.length < count && attempt < count * 100; attempt++) {
    let position = initialPosition();
    const openingMoves: Move[] = [];
    const plies = 2 + (attempt % 16);
    for (let ply = 0; ply < plies; ply++) {
      const legal = generateMoves(position).slice().sort((a, b) => moveKey(a).localeCompare(moveKey(b)));
      if (!legal.length) break;
      const move = legal[random() % legal.length];
      openingMoves.push(move);
      position = applyMove(position, move);
    }
    const key = positionKey(position);
    if (openingMoves.length === plies && generateMoves(position).length && !seen.has(key)) {
      seen.add(key);
      starts.push({ id: `search-ablation-v1-${String(starts.length + 1).padStart(2, '0')}`, position, openingMoves });
    }
  }
  if (starts.length !== count) throw new Error(`could only generate ${starts.length}/${count} unique ablation starts`);
  return { version: 'makhos-search-ablation-starts-v1', seed: seed >>> 0,
    generator: 'xorshift32/legal-sorted/diverse-plies-2-17/v1 (generated measurement corpus; not an opening book)', starts };
}

export const SEARCH_ABLATION_START_SUITE = generateSearchAblationStartSuite();
