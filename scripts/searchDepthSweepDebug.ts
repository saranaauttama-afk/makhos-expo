import { evaluateWithBreakdown } from '../src/coreClaude/eval';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import type { Position } from '../src/coreClaude/position';
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

function fmtBreakdown(root: Position, move: Move): string {
  const child = applyMove(root, move);
  const breakdown = evaluateWithBreakdown(child);
  return (
    `childStatic=${breakdown.finalScore} ` +
    `mat=${breakdown.material} psqt=${breakdown.psqt} mob=${breakdown.mobility} ` +
    `lowMob=${breakdown.lowMobilityResearch} hang=${breakdown.hangingPieces} ` +
    `promo=${breakdown.promotionThreat}`
  );
}

async function main(): Promise<void> {
  const fixtureId = (process.argv[2] as EndgameWeaknessFixtureId | undefined) ?? DEFAULT_FIXTURE_ID;
  const maxDepth = Number(process.argv[3] ?? DEFAULT_MAX_DEPTH);
  const timeMs = Number(process.argv[4] ?? DEFAULT_TIME_MS);
  const fixture = getEndgameWeaknessFixture(fixtureId);
  const pos = fixture.pos;

  console.log('Search depth sweep debug');
  console.log('mode=debug-only');
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

    const exactTies = (result.rootCandidates ?? []).filter(candidate => candidate.score === result.score);
    if (exactTies.length > 1) {
      const incumbent = exactTies[0];
      console.log(
        `  exactTie topScore=${result.score} incumbent=${fmtMove(incumbent.move)} ` +
        `reason=incumbent survives because root best only updates on strictly greater score`,
      );
      console.log(`    incumbent ${fmtBreakdown(pos, incumbent.move)}`);
      for (const challenger of exactTies.slice(1)) {
        console.log(
          `    challenger ${fmtMove(challenger.move)} ` +
          `${fmtBreakdown(pos, challenger.move)}`,
        );
      }
    }
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
