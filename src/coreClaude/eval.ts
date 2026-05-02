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
  score += mobilityScore(p);
  score += promotionThreatScore(p);
  // protectedMenBonus: Texel tuning found weight 0 — omitted
  score += backRankGuard(p) * (1 - eg); // less critical in endgame
  score += simplificationBonus(p);
  score += Math.round(kingEndgameScore(p) * eg * 0.35);

  return score | 0;
}

export function evaluate(p: Position): number {
  return handEvaluate(p);
}
