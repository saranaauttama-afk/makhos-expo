import { B1 } from '../src/coreClaude/bitboards';
import { Position } from '../src/coreClaude/position';

export type EndgameWeaknessFixtureId =
  | 'small-piece-king-vs-men'
  | 'small-endgame'
  | 'low-mobility-squeeze'
  | 'low-mobility-squeeze-p2'
  | 'quiet-hanging-piece-p1'
  | 'quiet-hanging-piece-p2';

export interface EndgameWeaknessFixture {
  id: EndgameWeaknessFixtureId;
  bucket: string;
  note: string;
  pos: Position;
}

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

// Debug-only fixture set reused by inspection tooling. These positions mirror
// the benchmark cases so weak endgame/tactical behavior can be inspected
// offline without duplicating board data across scripts.
export const ENDGAME_WEAKNESS_FIXTURES: EndgameWeaknessFixture[] = [
  {
    id: 'small-piece-king-vs-men',
    bucket: 'small-piece endgame',
    note: 'Known warning-case endgame weakness from the regression harness.',
    pos: makePosition({ side: 1, p1Kings: B1(21), p2Men: B1(13) | B1(6) }),
  },
  {
    id: 'small-endgame',
    bucket: 'small-piece endgame',
    note: 'Small endgame case that currently passes but has weak eval separation.',
    pos: makePosition({ side: -1, p1Kings: B1(18), p1Men: B1(25), p2Kings: B1(10), p2Men: B1(6) }),
  },
  {
    id: 'low-mobility-squeeze',
    bucket: 'low mobility',
    note: 'Primary squeeze-pattern case used in eval and override inspections.',
    pos: makePosition({ side: 1, p1Men: B1(24) | B1(25) | B1(29), p2Men: B1(16) | B1(17) | B1(20) }),
  },
  {
    id: 'low-mobility-squeeze-p2',
    bucket: 'low mobility',
    note: 'Mirrored squeeze-side case from the tactical benchmark.',
    pos: makePosition({ side: -1, p1Men: B1(11) | B1(14) | B1(15), p2Men: B1(4) | B1(5) | B1(8) }),
  },
  {
    id: 'quiet-hanging-piece-p1',
    bucket: 'quiet hanging piece',
    note: 'Current static hanging term under-explains why this case is solved.',
    pos: makePosition({ side: 1, p1Men: B1(21) | B1(25) | B1(30), p2Men: B1(13) | B1(14) | B1(17) }),
  },
  {
    id: 'quiet-hanging-piece-p2',
    bucket: 'quiet hanging piece',
    note: 'P2-side quiet hanging analogue for future fixture-based debugging.',
    pos: makePosition({ side: -1, p1Men: B1(14) | B1(18) | B1(21), p2Men: B1(1) | B1(6) | B1(10) }),
  },
];

const ENDGAME_WEAKNESS_FIXTURE_MAP = new Map(
  ENDGAME_WEAKNESS_FIXTURES.map(fixture => [fixture.id, fixture] as const),
);

export function getEndgameWeaknessFixture(id: EndgameWeaknessFixtureId): EndgameWeaknessFixture {
  const fixture = ENDGAME_WEAKNESS_FIXTURE_MAP.get(id);
  if (!fixture) throw new Error(`Unknown endgame weakness fixture: ${id}`);
  return fixture;
}

export function getEndgameWeaknessFixtures(ids?: EndgameWeaknessFixtureId[]): EndgameWeaknessFixture[] {
  if (!ids?.length) return [...ENDGAME_WEAKNESS_FIXTURES];
  return ids.map(getEndgameWeaknessFixture);
}
