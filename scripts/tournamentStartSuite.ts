import { Move, applyMove, generateMoves } from '../src/coreClaude/movegen';
import { initialPosition, Position } from '../src/coreClaude/position';
import { createHash } from 'node:crypto';

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

/** Canonical content identity includes replay identity rather than merely the
 * generated final boards. Object field order below is part of fingerprint v1. */
export function searchAblationSuiteFingerprint(suite: TournamentStartSuite): string {
  const content = suite.starts.map(start => ({
    id: start.id,
    initialPosition: start.initialPosition ? {
      side:start.initialPosition.side,p1Men:start.initialPosition.p1Men,p1Kings:start.initialPosition.p1Kings,
      p2Men:start.initialPosition.p2Men,p2Kings:start.initialPosition.p2Kings,halfmoveClock:start.initialPosition.halfmoveClock,
    } : null,
    openingMoves: start.openingMoves.map(move => ({ from:move.from,to:move.to,captured:[...move.captured],
      path:[...(move.path ?? [])],promote:move.promote })),
    finalPosition: { side:start.position.side,p1Men:start.position.p1Men,p1Kings:start.position.p1Kings,
      p2Men:start.position.p2Men,p2Kings:start.position.p2Kings,halfmoveClock:start.position.halfmoveClock },
  }));
  return createHash('sha256').update(JSON.stringify(content)).digest('hex');
}

export const SEARCH_ABLATION_START_SUITE_V1_FINGERPRINT =
  searchAblationSuiteFingerprint(SEARCH_ABLATION_START_SUITE);

/** Phase 3B independent confirmation corpus. The seed is mechanically the
 * first eight hexadecimal digits of the Phase 3A merge SHA 4149c4b1..., and
 * was frozen before any games were inspected. */
export const EXTENSION_CONFIRMATION_SEED = 0x4149c4b1;
export const EXTENSION_CONFIRMATION_START_SUITE: TournamentStartSuite = (() => {
  const phase3aStates=new Set(SEARCH_ABLATION_START_SUITE.starts.map(s=>positionKey(s.position)));
  const raw=generateSearchAblationStartSuite(EXTENSION_CONFIRMATION_SEED,128);
  const independent=raw.starts.filter(s=>!phase3aStates.has(positionKey(s.position))).slice(0,64);
  if(independent.length!==64)throw new Error('could not generate 64 Phase 3B states disjoint from Phase 3A');
  return {...raw,version:'makhos-extension-confirmation-starts-v1',generator:'xorshift32/legal-sorted/diverse-plies-2-17/v1; seed=first8hex(Phase3A merge SHA)',
    starts:independent.map((s,i)=>({...s,id:`extension-confirmation-v1-${String(i+1).padStart(2,'0')}`}))};
})();
export const EXTENSION_CONFIRMATION_V1_FINGERPRINT=searchAblationSuiteFingerprint(EXTENSION_CONFIRMATION_START_SUITE);

/** Phase 3C promotion confirmation corpus. Its seed is the first eight hex
 * digits of the pre-confirmation PR head 3a144f82..., frozen before games were
 * run. Complete final states are disjoint from both earlier ablation suites. */
export const PHASE3C_CONFIRMATION_SEED = 0x3a144f82;
export const PHASE3C_CONFIRMATION_START_SUITE: TournamentStartSuite = (() => {
  const priorStates = new Set([...SEARCH_ABLATION_START_SUITE.starts,
    ...EXTENSION_CONFIRMATION_START_SUITE.starts].map(s => positionKey(s.position)));
  const raw = generateSearchAblationStartSuite(PHASE3C_CONFIRMATION_SEED, 192);
  const independent = raw.starts.filter(s => !priorStates.has(positionKey(s.position))).slice(0, 64);
  if (independent.length !== 64) throw new Error('could not generate 64 Phase 3C states disjoint from Phase 3A/3B');
  return { ...raw, version: 'makhos-phase3c-confirmation-starts-v1',
    generator: 'xorshift32/legal-sorted/diverse-plies-2-17/v1; seed=first8hex(pre-confirmation PR head)',
    starts: independent.map((s, i) => ({ ...s, id: `phase3c-confirmation-v1-${String(i + 1).padStart(2, '0')}` })) };
})();
export const PHASE3C_CONFIRMATION_V1_FINGERPRINT = searchAblationSuiteFingerprint(PHASE3C_CONFIRMATION_START_SUITE);
