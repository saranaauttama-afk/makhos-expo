import { generateMoves, type Move } from '../src/coreClaude/movegen';
import { getEndgameWeaknessFixture, type EndgameWeaknessFixtureId } from './endgameWeaknessFixtures';
import { iterativeDeepening, moveKey } from '../src/coreClaude/search/alphabeta';
import { TT } from '../src/coreClaude/search/tt';
import { hashPosition } from '../src/coreClaude/search/zobrist';

const DEFAULT_FIXTURE_ID: EndgameWeaknessFixtureId = 'low-mobility-squeeze-p2';
const DEFAULT_MAX_DEPTH = 8;
const DEFAULT_TIME_MS = 20_000;

function fmtMove(move: Move | undefined): string {
  if (!move) return '(none)';
  const caps = move.captured.length ? `x${move.captured.length}` : '';
  const promo = move.promote ? 'K' : '';
  return `${move.from + 1}->${move.to + 1}${caps}${promo}`;
}

async function main(): Promise<void> {
  const fixtureId = (process.argv[2] as EndgameWeaknessFixtureId | undefined) ?? DEFAULT_FIXTURE_ID;
  const maxDepth = Number(process.argv[3] ?? DEFAULT_MAX_DEPTH);
  const timeMs = Number(process.argv[4] ?? DEFAULT_TIME_MS);
  const fixture = getEndgameWeaknessFixture(fixtureId);
  const pos = fixture.pos;

  console.log('Search depth sweep debug');
  console.log(`fixture=${fixture.id}`);
  console.log(`bucket=${fixture.bucket}`);
  console.log(`note=${fixture.note}`);
  console.log(`side=${pos.side}`);
  console.log(`maxDepth=${maxDepth}`);
  console.log(`timeMs=${timeMs}`);

  const legal = generateMoves(pos);
  console.log(`legalMoves=${legal.length}`);
  for (const move of legal) {
    console.log(`  legal ${fmtMove(move)}`);
  }

  for (let depthLimit = 1; depthLimit <= maxDepth; depthLimit++) {
    const result = await iterativeDeepening(
      pos,
      timeMs,
      new TT(),
      undefined,
      [hashPosition(pos)],
      { cancelled: false },
      depthLimit,
    );
    console.log('');
    console.log(
      `depthLimit=${depthLimit} reached=${result.depth} timedOut=${result.timedOut} ` +
      `best=${fmtMove(result.best)} score=${result.score} ` +
      `override=${result.overrideReason ?? '(none)'} nodes=${result.nodes} qnodes=${result.qnodes}`,
    );
    for (const candidate of result.rootCandidates ?? []) {
      console.log(
        `  cand ${fmtMove(candidate.move)} score=${candidate.score}` +
        `${result.best && moveKey(candidate.move) === moveKey(result.best) ? ' <-best' : ''}`,
      );
    }
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
