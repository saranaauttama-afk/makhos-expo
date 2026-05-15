// eval.ts — Codex v3 evaluation
// v3 vs v2:
//   1. PSQT tables — men (advancement + column safety) and kings (centrality)
//      replace promotionProgress + centerControl + kingCentralization
//   2. King endgame proximity bonus — king hunts enemy men when winning
//   3. Removed captureNetScore entirely (was calling generateMoves() twice
//      per leaf node — the single biggest eval performance drain)
//   4. Removed dead see() function

import { B1, BB, bitCount, bits, STEPS, toRC } from './bitboards';
import { Position } from './position';

const START_TOTAL = 16;
const VAL_MAN     = 100;
const VAL_KING    = 280;
const ENABLE_EVAL_EXPERIMENTS = process.env.MAKHOS_ENABLE_EVAL_EXPERIMENTS === '1';

interface EvalExperimentConfig {
  mobilityScalePct: number;
  promotionThreatScalePct: number;
  hangingPiecesScalePct: number;
  backRankGuardScalePct: number;
  enableLowMobilityResearch: boolean;
  lowMobilityResearchScalePct: number;
}

// Infrastructure only: keep experiment scales neutral while disabled so the
// current stable eval remains identical until a future tiny experiment is approved.
const EVAL_EXPERIMENTS: EvalExperimentConfig = {
  mobilityScalePct: 100,
  promotionThreatScalePct: 100,
  hangingPiecesScalePct: 100,
  backRankGuardScalePct: 100,
  enableLowMobilityResearch: process.env.MAKHOS_ENABLE_LOW_MOBILITY_RESEARCH === '1',
  lowMobilityResearchScalePct: Number(process.env.MAKHOS_LOW_MOBILITY_RESEARCH_SCALE_PCT ?? '100'),
};

export interface EvalBreakdown {
  material: number;
  psqt: number;
  mobility: number;
  lowMobilityResearch: number;
  lowLibertyEdgeLock: number;
  promotionThreat: number;
  hangingPieces: number;
  backRankGuard: number;
  simplification: number;
  kingEndgame: number;
  allKingsEndgame: number;
  totalPieces: number;
  endgameFactor: number;
  kingValue: number;
  finalScore: number;
}

function applyEvalExperimentScale(base: number, scalePct: number): number {
  if (!ENABLE_EVAL_EXPERIMENTS || scalePct === 100) return base;
  return Math.round(base * scalePct / 100);
}

// ── Piece-Square Tables (built once at module load, zero runtime cost) ────────
//
// P1_MAN_PST[sq]  : bonus for a P1 man at square sq
//                   P1 advances UP (toward row 0) — low row index = good
// P2_MAN_PST[sq]  : bonus for a P2 man at square sq
//                   P2 advances DOWN (toward row 7) — high row index = good
// KING_PST[sq]    : centrality bonus for a king at sq (same for both sides)
//                   Thai kings fly everywhere — central kings control more diagonals

const P1_MAN_PST = new Int16Array(32);
const P2_MAN_PST = new Int16Array(32);
const KING_PST   = new Int16Array(32);

(function buildPST() {
  // Row advancement bonus for P1 men:
  //   row 0 = about to promote (won't have a man there, included for completeness)
  //   row 7 = P1's starting back rank
  // Texel-tuned (300 games, 18k positions): row 1 near-promotion gets big bonus
  const rowBonus = [42, 80, 0, 20, 8, 0, 0, 0]; // index = row 0..7

  // Texel-tuned: col 2/5 are structurally strong; edges modestly rewarded
  const colBonus = [11, 28, 51, 20, 24, 29, 12, 0]; // index = col 0..7

  for (let sq = 0; sq < 32; sq++) {
    const { r, c } = toRC(sq);

    P1_MAN_PST[sq] = rowBonus[r]     + colBonus[c]; // lower r = more advanced for P1
    P2_MAN_PST[sq] = rowBonus[7 - r] + colBonus[c]; // higher r = more advanced for P2

    // King centrality: Chebyshev distance from the centre band (rows/cols 3-4)
    // 0 = on/near centre, 3 = corner — cap contribution at 8 cp
    const dr = r <= 3 ? 3 - r : r - 4;
    const dc = c <= 3 ? 3 - c : c - 4;
    KING_PST[sq] = Math.max(0, 8 - Math.max(dr, dc) * 2);
  }
})();

// ── Material ──────────────────────────────────────────────────────────────────
function materialScore(p: Position, kingVal: number): number {
  const s = p.side;
  return (
    VAL_MAN  * (bitCount(s === 1 ? p.p1Men   : p.p2Men)   - bitCount(s === 1 ? p.p2Men   : p.p1Men)) +
    kingVal  * (bitCount(s === 1 ? p.p1Kings : p.p2Kings) - bitCount(s === 1 ? p.p2Kings : p.p1Kings))
  );
}

// ── PSQT ──────────────────────────────────────────────────────────────────────
function psqtScore(p: Position): number {
  const s      = p.side;
  const myMPST = s === 1 ? P1_MAN_PST : P2_MAN_PST;
  const opMPST = s === 1 ? P2_MAN_PST : P1_MAN_PST;
  let score = 0;
  for (const sq of bits(s === 1 ? p.p1Men   : p.p2Men))   score += myMPST[sq];
  for (const sq of bits(s === 1 ? p.p2Men   : p.p1Men))   score -= opMPST[sq];
  for (const sq of bits(s === 1 ? p.p1Kings : p.p2Kings)) score += KING_PST[sq];
  for (const sq of bits(s === 1 ? p.p2Kings : p.p1Kings)) score -= KING_PST[sq];
  return score;
}

// ── Mobility (fast — no generateMoves call) ───────────────────────────────────
// Count available step-squares for each side.  Kings use adjacent squares only
// (approximation of full ray mobility — O(pieces), safe to call every node).
function mobilityScore(p: Position): number {
  const occ  = (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
  const side = p.side;
  const opp  = side === 1 ? -1 : 1 as 1 | -1;
  let my = 0, op = 0;

  const kingRayMobility = (sq: number) => {
    let total = 0;
    for (const first of STEPS[sq]) {
      let cur = first.to;
      while (cur >= 0) {
        if (occ & B1(cur)) break;
        total++;
        const next = STEPS[cur].find(st => st.dir === first.dir);
        if (!next) break;
        cur = next.to;
      }
    }
    return total;
  };

  for (const sq of bits(side === 1 ? p.p1Men : p.p2Men))
    for (const st of STEPS[sq]) {
      if (side === 1  && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (!(occ & B1(st.to))) my++;
    }
  for (const sq of bits(side === 1 ? p.p1Kings : p.p2Kings))
    my += kingRayMobility(sq);

  for (const sq of bits(side === 1 ? p.p2Men : p.p1Men))
    for (const st of STEPS[sq]) {
      if (opp === 1  && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (opp === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (!(occ & B1(st.to))) op++;
    }
  for (const sq of bits(side === 1 ? p.p2Kings : p.p1Kings))
    op += kingRayMobility(sq);

  return 1 * (my - op); // Texel-tuned: 5→1
}

function lowMobilityResearchSignal(p: Position): number {
  const total = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  if (total > 8 || p.p1Kings !== 0 || p.p2Kings !== 0) return 0;

  const occ = (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
  const side = p.side;
  const opp = (side === 1 ? -1 : 1) as 1 | -1;

  const statsForMen = (men: BB, menSide: 1 | -1) => {
    let blocked = 0;
    let cramped = 0;
    let free = 0;
    let totalForward = 0;
    for (const sq of bits(men)) {
      let forwardSteps = 0;
      for (const st of STEPS[sq]) {
        if (menSide === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
        if (menSide === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
        if (!(occ & B1(st.to))) forwardSteps++;
      }
      totalForward += forwardSteps;
      if (forwardSteps === 0) blocked++;
      else if (forwardSteps === 1) cramped++;
      else free++;
    }
    return { blocked, cramped, free, totalForward };
  };

  const myMen = side === 1 ? p.p1Men : p.p2Men;
  const opMen = side === 1 ? p.p2Men : p.p1Men;
  const my = statsForMen(myMen, side);
  const op = statsForMen(opMen, opp);
  const hasBlockedAsymmetry = op.blocked !== my.blocked;
  const bothNoFreeMen = my.free === 0 && op.free === 0;
  const nearTotalExhaustion = my.totalForward <= 3 && op.totalForward <= 3;
  if (!hasBlockedAsymmetry || !bothNoFreeMen || !nearTotalExhaustion) return 0;
  return 4 * ((op.blocked * 2 + op.cramped) - (my.blocked * 2 + my.cramped));
}

export function lowLibertyEdgeLockSignal(p: Position): number {
  if (p.p1Kings !== 0 || p.p2Kings !== 0) return 0;

  const total = bitCount(p.p1Men | p.p2Men);
  if (total > 6) return 0;

  const side = p.side;
  const myMen = side === 1 ? p.p1Men : p.p2Men;
  const myMenCount = bitCount(myMen);
  if (myMenCount !== 3) return 0;

  const occ = (p.p1Men | p.p2Men) >>> 0;
  let totalForwardSteps = 0;
  let blockedMen = 0;
  let crampedMen = 0;
  let freeMen = 0;
  let backBandMen = 0;
  let leftNearEdgeMen = 0;
  let rightNearEdgeMen = 0;
  let leftHardEdgeMen = 0;
  let rightHardEdgeMen = 0;

  for (const sq of bits(myMen)) {
    const { r, c } = toRC(sq);
    const inBackBand = side === 1 ? r >= 5 : r <= 2;
    if (!inBackBand) return 0;
    backBandMen++;

    if (c <= 2) leftNearEdgeMen++;
    if (c >= 5) rightNearEdgeMen++;
    if (c <= 1) leftHardEdgeMen++;
    if (c >= 6) rightHardEdgeMen++;

    let forwardSteps = 0;
    for (const st of STEPS[sq]) {
      if (side === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (occ & B1(st.to)) continue;
      forwardSteps++;
      totalForwardSteps++;
    }
    if (forwardSteps === 0) blockedMen++;
    else if (forwardSteps === 1) crampedMen++;
    else freeMen++;
  }

  const sameFlankNearEdgeMen = Math.max(leftNearEdgeMen, rightNearEdgeMen);
  const sameFlankHardEdgeMen = Math.max(leftHardEdgeMen, rightHardEdgeMen);
  if (backBandMen !== myMenCount) return 0;
  if (sameFlankNearEdgeMen < 2) return 0;
  if (sameFlankHardEdgeMen < 1) return 0;
  if (blockedMen < 1) return 0;
  if (crampedMen < 1) return 0;
  if (freeMen > 1) return 0;
  if (totalForwardSteps > 3) return 0;

  return 1;
}

// ── Back rank guard ───────────────────────────────────────────────────────────
// Reward keeping men on your own back rank as promotion-stoppers.
// Less important in pure endgame (scaled to zero there).
function backRankGuard(p: Position): number {
  const side = p.side;
  let score = 0;
  for (const sq of bits(side === 1 ? p.p1Men : p.p2Men)) {
    const { r } = toRC(sq);
    if ((side === 1 && r === 7) || (side === -1 && r === 0)) score += 5;
  }
  for (const sq of bits(side === 1 ? p.p2Men : p.p1Men)) {
    const { r } = toRC(sq);
    if ((side === 1 && r === 0) || (side === -1 && r === 7)) score -= 5;
  }
  return score;
}

// ── Promotion threats ─────────────────────────────────────────────────────────
// Reward men close to promotion when at least one forward lane is still open.
// Cheap enough for leaf eval: adjacency only, no full move generation.
function promotionThreatScore(p: Position): number {
  const occ = (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
  const side = p.side;

  const scoreMen = (men: BB, menSide: 1 | -1) => {
    let score = 0;
    for (const sq of bits(men)) {
      const { r } = toRC(sq);
      const dist = menSide === 1 ? r : 7 - r;
      if (dist > 2) continue;

      let hasLane = false;
      for (const st of STEPS[sq]) {
        if (menSide === 1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
        if (menSide === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
        if (!(occ & B1(st.to))) { hasLane = true; break; }
      }
      if (!hasLane) continue;
      score += dist <= 1 ? 28 : 10;
    }
    return score;
  };

  const myMen = side === 1 ? p.p1Men : p.p2Men;
  const opMen = side === 1 ? p.p2Men : p.p1Men;
  const opSide = (side === 1 ? -1 : 1) as 1 | -1;
  return scoreMen(myMen, side) - scoreMen(opMen, opSide);
}

// When ahead in piece count and we have kings vs enemy men:
// reward our kings for being close to enemy men (guides the king to hunt them).
function kingEndgameScore(p: Position): number {
  const side    = p.side;
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  const opMen   = side === 1 ? p.p2Men   : p.p1Men;
  if (!myKings || !opMen) return 0;

  const myCount = bitCount(side === 1 ? p.p1Men | p.p1Kings : p.p2Men | p.p2Kings);
  const opCount = bitCount(side === 1 ? p.p2Men | p.p2Kings : p.p1Men | p.p1Kings);
  if (myCount <= opCount) return 0; // only apply when winning

  let score = 0;
  for (const kSq of bits(myKings)) {
    const { r: kr, c: kc } = toRC(kSq);
    for (const mSq of bits(opMen)) {
      const { r: mr, c: mc } = toRC(mSq);
      const dist = Math.abs(kr - mr) + Math.abs(kc - mc); // Manhattan approx
      score += Math.max(0, 12 - dist) * 3; // max +36 when adjacent
    }
  }
  return score;
}

function allKingsEndgameScore(p: Position): number {
  if (p.p1Men !== 0 || p.p2Men !== 0) return 0;

  const side = p.side;
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  const opKings = side === 1 ? p.p2Kings : p.p1Kings;
  const myCount = bitCount(myKings);
  const opCount = bitCount(opKings);
  if (!myKings || !opKings || myCount <= opCount) return 0;

  const opSquares = [...bits(opKings)];
  let score = 0;

  for (const opSq of opSquares) {
    const { r: or, c: oc } = toRC(opSq);
    const edgePressure = Math.max(
      Math.abs(or - 3.5),
      Math.abs(oc - 3.5),
    );
    score += Math.round(edgePressure * 12);

    let nearest = 99;
    for (const mySq of bits(myKings)) {
      const { r: mr, c: mc } = toRC(mySq);
      nearest = Math.min(nearest, Math.abs(mr - or) + Math.abs(mc - oc));
    }
    score += Math.max(0, 14 - nearest) * 5;
  }

  return score;
}

// ── Protected men ─────────────────────────────────────────────────────────────
// A man is "protected" when a friendly piece sits in the direction it came from
// (its "behind" diagonal).  Protected men are harder to capture safely because
// the attacker gets recaptured.  Reward formations that cover each other.
function protectedMenBonus(p: Position): number {
  const side  = p.side;
  const myMen = side === 1 ? p.p1Men  : p.p2Men;
  const opMen = side === 1 ? p.p2Men  : p.p1Men;
  const myAll = (myMen | (side === 1 ? p.p1Kings : p.p2Kings)) >>> 0;
  const opAll = (opMen | (side === 1 ? p.p2Kings : p.p1Kings)) >>> 0;
  let score   = 0;

  // For P1 (advances UL/UR), "behind" = DL or DR
  // For P2 (advances DL/DR), "behind" = UL or UR
  for (const sq of bits(myMen)) {
    for (const st of STEPS[sq]) {
      if (side === 1  && (st.dir === 'UL' || st.dir === 'UR')) continue; // skip forward
      if (side === -1 && (st.dir === 'DL' || st.dir === 'DR')) continue;
      if (myAll & B1(st.to)) { score += 9; break; } // friendly piece behind → protected
    }
  }
  for (const sq of bits(opMen)) {
    for (const st of STEPS[sq]) {
      if (side === 1  && (st.dir === 'DL' || st.dir === 'DR')) continue; // opponent's behind
      if (side === -1 && (st.dir === 'UL' || st.dir === 'UR')) continue;
      if (opAll & B1(st.to)) { score -= 9; break; }
    }
  }
  return score;
}

// ── Hanging pieces penalty ────────────────────────────────────────────────────
// Detect pieces that are threatened by opponent but not defended.
// This is the inverse of protection: a piece is "hanging" when an enemy piece
// can capture it and either (a) we have no piece that can recapture, or
// (b) the attacker is protected but our piece is not.
// This addresses roadmap requirement: "Add cheap tactical features" & "hanging-piece signals"
function hangingPiecesPenalty(p: Position): number {
  const side  = p.side;
  const opp   = (side === 1 ? -1 : 1) as 1 | -1;
  const myMen = side === 1 ? p.p1Men   : p.p2Men;
  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  const opMen = side === 1 ? p.p2Men   : p.p1Men;
  const opKings = side === 1 ? p.p2Kings : p.p1Kings;
  const myAll = (myMen | myKings) >>> 0;
  const opAll = (opMen | opKings) >>> 0;
  let penalty = 0;

  // Check if a square is threatened by opponent
  const isThreatened = (sq: number): boolean => {
    for (const st of STEPS[sq]) {
      const attackerSq = st.to;
      if (!(opAll & B1(attackerSq))) continue;

      // Check if this opponent piece can capture toward our square
      const isOpMen = !!(opMen & B1(attackerSq));
      if (isOpMen) {
        // Men can only capture in their forward directions
        if (opp === 1  && (st.dir === 'DL' || st.dir === 'DR')) return true;
        if (opp === -1 && (st.dir === 'UL' || st.dir === 'UR')) return true;
      } else {
        // Kings can capture from any diagonal
        return true;
      }
    }
    return false;
  };

  // Check if a square is defended by our pieces
  const isDefended = (sq: number): boolean => {
    for (const st of STEPS[sq]) {
      const defenderSq = st.to;
      if (!(myAll & B1(defenderSq))) continue;

      // Check if this friendly piece can recapture
      const isMyMen = !!(myMen & B1(defenderSq));
      if (isMyMen) {
        // Men defend from their "behind" diagonals
        if (side === 1  && (st.dir === 'DL' || st.dir === 'DR')) return true;
        if (side === -1 && (st.dir === 'UL' || st.dir === 'UR')) return true;
      } else {
        // Kings defend from any diagonal
        return true;
      }
    }
    return false;
  };

  // Check our men for hanging
  for (const sq of bits(myMen)) {
    if (isThreatened(sq) && !isDefended(sq)) {
      penalty += 80; // hanging man
    }
  }

  // Check our kings for hanging (more valuable, bigger penalty)
  for (const sq of bits(myKings)) {
    if (isThreatened(sq) && !isDefended(sq)) {
      penalty += 150; // hanging king
    }
  }

  return -penalty;
}

// ── Simplification bonus ──────────────────────────────────────────────────────
// When ahead in pieces, reward trading (fewer pieces = easier technical win).
function simplificationBonus(p: Position): number {
  const side = p.side;
  const myN  = bitCount(side === 1 ? p.p1Men | p.p1Kings : p.p2Men | p.p2Kings);
  const opN  = bitCount(side === 1 ? p.p2Men | p.p2Kings : p.p1Men | p.p1Kings);
  if (myN <= opN) return 0;
  return (START_TOTAL - (myN + opN)) * 6; // Texel-tuned: 3→6
}

// ── Main evaluation ───────────────────────────────────────────────────────────
export function handEvaluate(p: Position): number {
  const total = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  // eg: 0 = opening/midgame (16 pieces), 1 = pure endgame (≤8 pieces)
  const eg = total <= 8 ? (8 - total) / 8 : 0;
  // King value scales 280 (opening) → 380 (pure endgame)
  const kingVal = (VAL_KING + eg * 100) | 0;

  let score = 0;
  score += materialScore(p, kingVal);
  score += psqtScore(p);
  score += applyEvalExperimentScale(mobilityScore(p), EVAL_EXPERIMENTS.mobilityScalePct);
  if (ENABLE_EVAL_EXPERIMENTS && EVAL_EXPERIMENTS.enableLowMobilityResearch) {
    score += applyEvalExperimentScale(
      lowMobilityResearchSignal(p),
      EVAL_EXPERIMENTS.lowMobilityResearchScalePct,
    );
  }
  score += applyEvalExperimentScale(promotionThreatScore(p), EVAL_EXPERIMENTS.promotionThreatScalePct);
  score += applyEvalExperimentScale(hangingPiecesPenalty(p), EVAL_EXPERIMENTS.hangingPiecesScalePct); // NEW: detect undefended pieces
  // protectedMenBonus: Texel tuning found weight 0 — omitted
  score += applyEvalExperimentScale(backRankGuard(p) * (1 - eg), EVAL_EXPERIMENTS.backRankGuardScalePct); // less critical in endgame
  score += simplificationBonus(p);
  score += Math.round(kingEndgameScore(p) * eg * 0.35);
  score += Math.round(allKingsEndgameScore(p) * eg * 0.5);

  return score | 0;
}

export function evaluate(p: Position): number {
  return handEvaluate(p);
}

export function createEmptyEvalBreakdown(): EvalBreakdown {
  return {
    material: 0,
    psqt: 0,
    mobility: 0,
    lowMobilityResearch: 0,
    lowLibertyEdgeLock: 0,
    promotionThreat: 0,
    hangingPieces: 0,
    backRankGuard: 0,
    simplification: 0,
    kingEndgame: 0,
    allKingsEndgame: 0,
    totalPieces: 0,
    endgameFactor: 0,
    kingValue: 0,
    finalScore: 0,
  };
}

// Debug-only helper path for eval inspection. Normal search should keep using
// evaluate()/handEvaluate() so the hot path remains allocation-free.
export function fillEvalBreakdown(p: Position, out: EvalBreakdown): number {
  const total = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  const eg = total <= 8 ? (8 - total) / 8 : 0;
  const kingVal = (VAL_KING + eg * 100) | 0;

  out.totalPieces = total;
  out.endgameFactor = eg;
  out.kingValue = kingVal;

  out.material = materialScore(p, kingVal);
  out.psqt = psqtScore(p);
  out.mobility = applyEvalExperimentScale(mobilityScore(p), EVAL_EXPERIMENTS.mobilityScalePct);
  out.lowMobilityResearch =
    ENABLE_EVAL_EXPERIMENTS && EVAL_EXPERIMENTS.enableLowMobilityResearch
      ? applyEvalExperimentScale(
          lowMobilityResearchSignal(p),
          EVAL_EXPERIMENTS.lowMobilityResearchScalePct,
        )
      : 0;
  out.lowLibertyEdgeLock = lowLibertyEdgeLockSignal(p);
  out.promotionThreat = applyEvalExperimentScale(promotionThreatScore(p), EVAL_EXPERIMENTS.promotionThreatScalePct);
  out.hangingPieces = applyEvalExperimentScale(hangingPiecesPenalty(p), EVAL_EXPERIMENTS.hangingPiecesScalePct);
  out.backRankGuard = applyEvalExperimentScale(backRankGuard(p) * (1 - eg), EVAL_EXPERIMENTS.backRankGuardScalePct);
  out.simplification = simplificationBonus(p);
  out.kingEndgame = Math.round(kingEndgameScore(p) * eg * 0.35);
  out.allKingsEndgame = Math.round(allKingsEndgameScore(p) * eg * 0.5);

  out.finalScore = (
    out.material +
    out.psqt +
    out.mobility +
    out.lowMobilityResearch +
    out.promotionThreat +
    out.hangingPieces +
    out.backRankGuard +
    out.simplification +
    out.kingEndgame +
    out.allKingsEndgame
  ) | 0;

  return out.finalScore;
}

export function evaluateWithBreakdown(p: Position): EvalBreakdown {
  const breakdown = createEmptyEvalBreakdown();
  fillEvalBreakdown(p, breakdown);
  return breakdown;
}
