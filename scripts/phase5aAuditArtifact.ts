import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { buildCanonicalTablebase, buildFreshRuleTablebase, canonicalMoveKey, canonicalStateKey, FRESH_RULE_TABLEBASE_VERSION, CANONICAL_ENCODING_VERSION } from '../src/coreClaude/search/exactEndgameTablebase';
import { applyMove, generateMoves } from '../src/coreClaude/movegen';
import { fixedNodeSearch, resetSearchHeuristicsForMeasurement } from '../src/coreClaude/search/alphabeta';
import { bitCount } from '../src/coreClaude/bitboards';

const SOURCE_COMMIT = 'bf1dd419d51e281b1ce3e94e51d24790cf3c0eaa';
const TEACHER_V1 = 'b2e6a35db6a50ea294a10f6b76a90b4e70e0689f';
const SAMPLE = 4096;

async function main() {
  const boardOnly = buildCanonicalTablebase();
  const tb = buildFreshRuleTablebase();
  const requestedDomain = [...tb.positions.entries()].filter(([, p]) => {
    const n = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
    return (n === 2 || n === 3) && (p.p1Men | p.p1Kings) !== 0 && (p.p2Men | p.p2Kings) !== 0;
  });
  const teacherDomain = requestedDomain.filter(([, p]) => generateMoves(p).length > 0);
  const sample = Array.from({ length: SAMPLE }, (_, i) => teacherDomain[Math.floor(i * teacherDomain.length / SAMPLE)]);
  let outcomeAgreement = 0, preserving = 0, dtmOptimal = 0, dtmEligible = 0, identity = 0, alternatives = 0;
  for (const [key, pos] of sample) {
    resetSearchHeuristicsForMeasurement();
    const result = await fixedNodeSearch(pos, 500);
    const entry = tb.entries.get(key)!;
    const predicted = result.score > 1000 ? 'WIN' : result.score < -1000 ? 'LOSS' : 'DRAW';
    if (predicted === entry.outcome) outcomeAgreement++;
    if (!result.best) continue;
    const moveKey = canonicalMoveKey(result.best);
    const child = tb.entries.get(canonicalStateKey({ ...applyMove(pos, result.best), halfmoveClock: 0 }))!;
    const moveOutcome = child.outcome === 'WIN' ? 'LOSS' : child.outcome === 'LOSS' ? 'WIN' : 'DRAW';
    if (moveOutcome === entry.outcome) preserving++;
    if (entry.dtm !== undefined) {
      dtmEligible++;
      if (entry.bestMoveKeys.includes(moveKey)) dtmOptimal++;
      if (entry.bestMoveKeys[0] === moveKey) identity++;
      if (entry.bestMoveKeys.length > 1 && entry.bestMoveKeys.includes(moveKey)) alternatives++;
    }
  }
  const pct = (n: number, denominator = SAMPLE) => Number((100 * n / denominator).toFixed(3));
  const lookupKeys = [...tb.entries.keys()];
  const lookupStarted = process.hrtime.bigint();
  let lookupGuard = 0;
  for (let i = 0; i < 1_000_000; i++) if (tb.entries.get(lookupKeys[i % lookupKeys.length])) lookupGuard++;
  const lookupNs = Number(process.hrtime.bigint() - lookupStarted);
  const existingTiming = (() => { try { return JSON.parse(readFileSync('.tmp/phase5a-summary.json', 'utf8')).generationMs; } catch { return tb.generationMs; } })();
  const artifact = {
    schemaVersion: 'phase5a-audit-v1', solverVersion: FRESH_RULE_TABLEBASE_VERSION,
    canonicalEncodingVersion: CANONICAL_ENCODING_VERSION, sourceCommit: SOURCE_COMMIT,
    teacherV1Identity: TEACHER_V1, canonicalStateCount: tb.entries.size,
    legalRequestedDomainStateCount: requestedDomain.length, terminalSinkStateCount: tb.entries.size - requestedDomain.length,
    countsByMaterialSignature: Object.fromEntries(Object.entries(tb.materialCounts).sort()),
    countsByPieceTypes: Object.fromEntries(Object.entries(tb.materialCounts).sort()), wdlTotals: tb.counts,
    maxDtm: tb.maxDtm, exactCanonicalCoverage: { pieces: [2, 3], sidesToMove: [1, -1], unknown: 0,
      rootSemantics: 'current position occurs once; halfmoveClock=0', inactivityClockCarried: true,
      repetitionJustification: 'quiet-cycle-draw-equivalence-v1', symmetryReduction: false,
      productionMoveGenerationAuthority: true },
    historyAwareTestCoverage: ['32-ply threshold', '16-ply king-only threshold', 'one ply before threshold',
      'second/third occurrence', 'unproven cycle remains UNKNOWN', 'child draw adjudication before mate-in-one'],
    verifier: { boardOnlyStructuralBellmanStates: boardOnly.entries.size,
      independentAlgorithm: 'top-down memoized material/clock solver; candidate entries unread until truth derived',
      exhaustiveRequestedTwoPieceStates: 6976, deterministicThreePieceSample: 100000,
      existingInScopeTinyEndgameFixtures: 1 },
    teacherAgreement: { sampleSize: SAMPLE, searchNodeBudget: 500,
      wdlMetricKind: 'heuristic score-sign classification; abs(score) <= 1000 is labeled DRAW',
      wdlAgreementCount: outcomeAgreement, wdlAgreementPercent: pct(outcomeAgreement), outcomePreservingMoveCount: preserving,
      outcomePreservingMovePercent: pct(preserving), dtmOptimalMoveCount: dtmOptimal,
      dtmOptimalMoveDenominator: dtmEligible, dtmOptimalMovePercent: pct(dtmOptimal, dtmEligible),
      canonicalBestMoveIdentityCount: identity, canonicalBestMoveIdentityDenominator: dtmEligible,
      canonicalBestMoveIdentityPercent: pct(identity, dtmEligible), equalOutcomeOptimalAlternativeCount: alternatives },
    performance: { canonicalBuildMilliseconds: existingTiming, lookupModel: 'Map.get O(1)', lookupCount: lookupGuard,
      lookupNanosecondsPerProbe: Number((lookupNs / lookupGuard).toFixed(1)),
      estimatedInMemoryBytes: tb.entries.size * 360, estimatedCompactSerializedBytes: tb.estimatedCompactBytes,
      timingAndMemoryAreExcludedFromFingerprint: true }, fingerprint: tb.fingerprint,
    boardTheoreticOnlyFingerprint: boardOnly.fingerprint,
  };
  mkdirSync('benchmarks/phase5a', { recursive: true });
  writeFileSync('benchmarks/phase5a/tablebase-audit-v1.json', JSON.stringify(artifact, null, 2) + '\n');
  console.log(JSON.stringify(artifact.teacherAgreement));
}
main();
