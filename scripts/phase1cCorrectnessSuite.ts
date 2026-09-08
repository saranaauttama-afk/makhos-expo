import { B1 } from '../src/coreClaude/bitboards';
import { applyMove, generateMoves, type Move } from '../src/coreClaude/movegen';
import { isDrawByInactivity, isTerminal, type Position } from '../src/coreClaude/position';
import {
  clearEndgameTablebaseCache,
  probeSmallEndgame,
  probeSmallEndgameDeterministic,
} from '../src/coreClaude/search/endgameTablebase';
import { buildRepetitionCounts, isThreefoldRepetition } from '../src/coreClaude/search/repetition';
import { hashPosition } from '../src/coreClaude/search/zobrist';

function pos(fields: Partial<Position> & Pick<Position, 'side'>): Position {
  return { p1Men: 0, p1Kings: 0, p2Men: 0, p2Kings: 0, halfmoveClock: 0, ...fields };
}

function assert(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}

function sameMove(a: Move | undefined, b: Move | undefined): boolean {
  return !!a && !!b && a.from === b.from && a.to === b.to &&
    a.captured.join(',') === b.captured.join(',');
}

function outcome(score: number): number { return Math.sign(score); }

function main() {
  let checks = 0;

  // Makhos captures are mandatory, but every complete sequence is a legal
  // choice; there is no global majority-capture priority.
  const mandatory = pos({ side: 1, p1Men: B1(22) | B1(30), p2Men: B1(17) });
  const mandatoryMoves = generateMoves(mandatory);
  assert(mandatoryMoves.length === 1 && mandatoryMoves[0].captured.join(',') === '17',
    'mandatory capture leaked a quiet move'); checks++;

  const unequalChoices = pos({
    side: 1,
    p1Men: B1(8) | B1(16),
    p2Men: B1(2) | B1(5) | B1(13),
  });
  const unequalLengths = generateMoves(unequalChoices)
    .map(move => move.captured.length).sort((a, b) => a - b);
  assert(unequalLengths.join(',') === '1,2',
    `expected both shorter and longer complete captures, got ${unequalLengths}`); checks++;

  const alternatives = pos({ side: 1, p1Men: B1(8) | B1(9), p2Men: B1(2) | B1(5) });
  const alternativeMoves = generateMoves(alternatives);
  assert(alternativeMoves.length === 2 && alternativeMoves.every(m => m.captured.length === 1),
    'multiple equal-length legal captures were not preserved'); checks++;

  const chain = pos({ side: 1, p1Men: B1(22), p2Men: B1(17) | B1(9) });
  const chainMove = generateMoves(chain)[0];
  assert(chainMove.from === 22 && chainMove.to === 6 && chainMove.captured.join(',') === '17,9',
    'multi-capture was not represented as one complete move'); checks++;

  const backwardsMan = pos({ side: 1, p1Men: B1(13), p2Men: B1(17) });
  assert(generateMoves(backwardsMan).every(m => m.captured.length === 0),
    'P1 man captured backward'); checks++;
  const backwardsP2Man = pos({ side: -1, p1Men: B1(13), p2Men: B1(17) });
  assert(generateMoves(backwardsP2Man).every(m => m.captured.length === 0),
    'P2 man captured backward'); checks++;

  // Men are crowned after the complete move, not while a capture sequence is
  // being generated. Landing on the last rank therefore ends this sequence.
  const deferredCrown = pos({ side: 1, p1Men: B1(8), p2Men: B1(5) | B1(6) });
  const crownMoves = generateMoves(deferredCrown);
  assert(crownMoves.length === 1 && crownMoves[0].to === 1 && crownMoves[0].promote &&
    crownMoves[0].captured.join(',') === '5', 'promotion occurred during capture sequence');
  const crowned = applyMove(deferredCrown, crownMoves[0]);
  assert((crowned.p1Kings & B1(1)) !== 0 && crowned.halfmoveClock === 0,
    'capture promotion was not applied after the move'); checks += 2;

  const flying = pos({ side: 1, p1Kings: B1(22), p2Men: B1(17) });
  const flyingCaptures = generateMoves(flying);
  assert(flyingCaptures.length === 1 && flyingCaptures[0].to === 13 &&
    flyingCaptures[0].captured.join(',') === '17',
    'Thai flying king must land on the first empty square after the captured piece'); checks++;
  const quietKing = pos({ side: 1, p1Kings: B1(22), p2Men: B1(0) });
  assert(generateMoves(quietKing).some(m => m.to === 4 && m.captured.length === 0),
    'flying king could not traverse multiple empty squares'); checks++;

  const quiet = applyMove(quietKing, generateMoves(quietKing)[0]);
  assert(quiet.halfmoveClock === 1, 'quiet king move did not increment halfmoveClock'); checks++;
  const kingsOnly = { ...quietKing, p2Men: 0, p2Kings: B1(0) };
  assert(isDrawByInactivity({ ...kingsOnly, halfmoveClock: 16 }), 'all-kings 16-ply draw missing'); checks++;
  assert(!isDrawByInactivity({ ...quietKing, p2Men: B1(0), p2Kings: 0, halfmoveClock: 16 }) &&
    isDrawByInactivity({ ...quietKing, p2Men: B1(0), p2Kings: 0, halfmoveClock: 32 }),
    '32-ply mixed-material inactivity boundary is wrong'); checks++;

  const eliminated = pos({ side: -1, p1Kings: B1(9) });
  assert(isTerminal(eliminated) && generateMoves(eliminated).length === 0,
    'piece elimination was not terminal'); checks++;
  const blocked = pos({ side: 1, p1Men: B1(0), p2Men: B1(4) });
  assert(!isTerminal(blocked) && generateMoves(blocked).length === 0,
    'no-legal-move terminal fixture is invalid'); checks++;

  const forced = pos({ side: 1, p1Kings: B1(18), p2Men: B1(14) });
  const forcedHash = hashPosition(forced);
  const threefold = [forcedHash, forcedHash, forcedHash];
  assert(isThreefoldRepetition(buildRepetitionCounts(threefold), forcedHash),
    'threefold repetition count failed'); checks++;

  // Production exact claims are limited to direct proofs and cannot depend on
  // legacy shared-cache warmth or a wall-clock race.
  const oracleWin = probeSmallEndgameDeterministic(forced, [forcedHash], 10_000);
  const productionWin = probeSmallEndgame(forced, [forcedHash], 0);
  assert(oracleWin.probe?.exact && productionWin?.exact &&
    outcome(oracleWin.probe.score) === outcome(productionWin.score) &&
    sameMove(oracleWin.probe.best, productionWin.best),
    'production forced win disagreed with deterministic oracle'); checks++;

  const productionLoss = probeSmallEndgame(eliminated, [hashPosition(eliminated)], 0);
  const oracleLoss = probeSmallEndgameDeterministic(eliminated, [hashPosition(eliminated)], 100);
  assert(productionLoss?.exact && oracleLoss.probe?.exact &&
    outcome(productionLoss.score) === outcome(oracleLoss.probe.score),
    'production terminal loss disagreed with deterministic oracle'); checks++;

  const repetitionDraw = probeSmallEndgame(forced, threefold, 0);
  assert(repetitionDraw?.exact && repetitionDraw.score === 0 && !repetitionDraw.best,
    'history-sensitive same board did not override its forced win'); checks++;
  const inactivity = { ...forced, halfmoveClock: 32 };
  assert(probeSmallEndgame(inactivity, [hashPosition(inactivity)], 0)?.score === 0,
    'production inactivity draw was not exact'); checks++;

  const unresolved = pos({ side: 1, p1Kings: B1(21), p2Men: B1(13) | B1(6) });
  clearEndgameTablebaseCache();
  const cold = probeSmallEndgame(unresolved, [hashPosition(unresolved)], 10_000);
  // Warm legacy state through the deterministic-independent production cache
  // lifecycle, then prove that repeated calls cannot manufacture exactness.
  const warm = probeSmallEndgame(unresolved, [hashPosition(unresolved)], 10_000);
  assert(cold === undefined && warm === undefined,
    'unproven king-vs-men position became exact through cache warmth'); checks++;

  console.log(`phase1cCorrectnessSuite: ${checks} assertions passed`);
}

main();
