// src/coreCodex/eval.ts
import { B1, BB, bitCount, bits, STEPS, toIndex, toRC } from './bitboards';
import { generateMoves } from './movegen';
import { Position } from './position';

const START_TOTAL = 16;

const W_BASE = {
  man: 100,
  king: 210,
  mobilityMen: 2,
  mobilityKing: 3,
  center: 2,
  promoteProgress: 6,
  backRankGuard: 3,
  kingProximity: 2,
  trappedKing: -12,
  simplification: 6,
  captureSwing: 90,
  captureTargets: 45,
};

interface CaptureInfo {
  maxChain: number;
  targets: number;
}

function occupied(p: Position): BB {
  return (p.p1Men | p.p1Kings | p.p2Men | p.p2Kings) >>> 0;
}

function nextInDir(from: number, dir: 'UL' | 'UR' | 'DL' | 'DR'): number {
  const step = STEPS[from].find((candidate) => candidate.dir === dir);
  return step ? step.to : -1;
}

function promotionDistanceSum(p: Position, side: 1 | -1): number {
  const men = side === 1 ? p.p1Men : p.p2Men;
  let sum = 0;
  for (const square of bits(men)) {
    const { r } = toRC(square);
    sum += side === 1 ? r : 7 - r;
  }
  return sum;
}

function centerScore(p: Position, side: 1 | -1): number {
  let score = 0;
  const men = side === 1 ? p.p1Men : p.p2Men;
  const kings = side === 1 ? p.p1Kings : p.p2Kings;
  for (const square of [...bits(men), ...bits(kings)]) {
    const { r, c } = toRC(square);
    if (r >= 2 && r <= 5 && c >= 2 && c <= 5) score++;
  }
  return score;
}

function backRankGuards(p: Position, side: 1 | -1): number {
  const men = side === 1 ? p.p1Men : p.p2Men;
  let guards = 0;
  for (const square of bits(men)) {
    const { r } = toRC(square);
    if ((side === 1 && r === 7) || (side === -1 && r === 0)) guards++;
  }
  return guards;
}

function runnerLaneBonus(p: Position, side: 1 | -1): number {
  const occ = occupied(p);
  const men = side === 1 ? p.p1Men : p.p2Men;
  const dirs = side === 1 ? (['UL', 'UR'] as const) : (['DL', 'DR'] as const);
  let bonus = 0;

  for (const square of bits(men)) {
    const { r } = toRC(square);
    const promotionDistance = side === 1 ? r : 7 - r;
    let openLane = false;

    for (const dir of dirs) {
      let cur = square;
      let blocked = false;
      while (true) {
        cur = nextInDir(cur, dir);
        if (cur < 0) break;
        if (occ & B1(cur)) {
          blocked = true;
          break;
        }
      }
      if (!blocked) {
        openLane = true;
        break;
      }
    }

    if (openLane) {
      bonus += 12 + Math.max(0, 6 - promotionDistance) * 3;
      if (promotionDistance <= 2) bonus += 10;
    }
  }

  return bonus;
}

function supportScore(p: Position, side: 1 | -1): number {
  const mine = side === 1 ? (p.p1Men | p.p1Kings) : (p.p2Men | p.p2Kings);
  const supportRowDelta = side === 1 ? 1 : -1;
  let score = 0;

  for (const square of bits(mine)) {
    const { r, c } = toRC(square);
    const supportRow = r + supportRowDelta;
    if (supportRow < 0 || supportRow > 7) continue;

    let supported = false;
    for (const dc of [-1, 1]) {
      const idx = toIndex(supportRow, c + dc);
      if (idx >= 0 && (mine & B1(idx))) {
        supported = true;
        break;
      }
    }

    if (supported) score++;
  }

  return score;
}

function edgeMenPenalty(p: Position, side: 1 | -1): number {
  const men = side === 1 ? p.p1Men : p.p2Men;
  let penalty = 0;

  for (const square of bits(men)) {
    const { r, c } = toRC(square);
    if (c !== 0 && c !== 7) continue;
    const closeToPromotion = side === 1 ? r <= 2 : r >= 5;
    penalty += closeToPromotion ? 1 : 2;
  }

  return penalty;
}

function mobility(p: Position, side: 1 | -1): { men: number; king: number } {
  const occ = occupied(p);
  const men = side === 1 ? p.p1Men : p.p2Men;
  const kings = side === 1 ? p.p1Kings : p.p2Kings;
  let menMoves = 0;
  let kingMoves = 0;

  for (const square of bits(men)) {
    for (const step of STEPS[square]) {
      if (side === 1 && (step.dir === 'DL' || step.dir === 'DR')) continue;
      if (side === -1 && (step.dir === 'UL' || step.dir === 'UR')) continue;
      if (((~occ) >>> 0) & B1(step.to)) menMoves++;
    }
  }

  for (const square of bits(kings)) {
    for (const step of STEPS[square]) {
      if (((~occ) >>> 0) & B1(step.to)) kingMoves++;
    }
  }

  return { men: menMoves, king: kingMoves };
}

function trappedKings(p: Position, side: 1 | -1): number {
  const occ = occupied(p);
  const kings = side === 1 ? p.p1Kings : p.p2Kings;
  let trapped = 0;

  for (const square of bits(kings)) {
    let exits = 0;
    for (const step of STEPS[square]) {
      if (((~occ) >>> 0) & B1(step.to)) exits++;
    }
    if (exits === 0) trapped++;
  }

  return trapped;
}

function kingProximityGain(p: Position, side: 1 | -1): number {
  const kings = side === 1 ? p.p1Kings : p.p2Kings;
  const enemies = side === 1 ? (p.p2Men | p.p2Kings) : (p.p1Men | p.p1Kings);
  if (enemies === 0) return 0;

  let distanceSum = 0;
  let count = 0;
  for (const king of bits(kings)) {
    const { r: rk, c: ck } = toRC(king);
    let best = 99;
    for (const enemy of bits(enemies)) {
      const { r: re, c: ce } = toRC(enemy);
      const dist = Math.max(Math.abs(rk - re), Math.abs(ck - ce));
      if (dist < best) best = dist;
    }
    if (best < 99) {
      distanceSum += best;
      count++;
    }
  }

  if (!count) return 0;
  return Math.max(0, 6 - distanceSum / count);
}

function cornerPressureBonus(p: Position, side: 1 | -1): number {
  const totalMen = bitCount(p.p1Men | p.p2Men);
  if (totalMen > 0) return 0;

  const myKings = side === 1 ? p.p1Kings : p.p2Kings;
  const enemies = side === 1 ? (p.p2Men | p.p2Kings) : (p.p1Men | p.p1Kings);
  if (myKings === 0 || enemies === 0) return 0;

  let score = 0;
  for (const enemy of bits(enemies)) {
    const { r, c } = toRC(enemy);
    const edgeDist = Math.min(r, 7 - r, c, 7 - c);
    score += Math.max(0, 2 - edgeDist) * 8;
  }

  return score;
}

function captureInfo(p: Position, side: 1 | -1): CaptureInfo {
  const view = p.side === side ? p : ({ ...p, side } as Position);
  const moves = generateMoves(view);
  if (!moves.length || moves[0].captured.length === 0) {
    return { maxChain: 0, targets: 0 };
  }

  let maxChain = 0;
  let capturedMask = 0;
  for (const move of moves) {
    if (move.captured.length > maxChain) maxChain = move.captured.length;
    for (const square of move.captured) {
      capturedMask = (capturedMask | B1(square)) >>> 0;
    }
  }

  return { maxChain, targets: bitCount(capturedMask) };
}

export function evaluate(p: Position): number {
  const totalPieces = bitCount(p.p1Men | p.p1Kings | p.p2Men | p.p2Kings);
  const gp = Math.max(0, Math.min(1, totalPieces / START_TOTAL));
  const eg = 1 - gp;

  const myMen = p.side === 1 ? p.p1Men : p.p2Men;
  const myKings = p.side === 1 ? p.p1Kings : p.p2Kings;
  const opMen = p.side === 1 ? p.p2Men : p.p1Men;
  const opKings = p.side === 1 ? p.p2Kings : p.p1Kings;

  const myMenN = bitCount(myMen);
  const myKingsN = bitCount(myKings);
  const opMenN = bitCount(opMen);
  const opKingsN = bitCount(opKings);
  const myAll = myMenN + myKingsN;
  const opAll = opMenN + opKingsN;

  const leadSimple = (myMenN - opMenN) + 2 * (myKingsN - opKingsN);
  const leadSign = leadSimple > 0 ? 1 : leadSimple < 0 ? -1 : 0;
  const leading = leadSign > 0;

  let kingWeight = W_BASE.king;
  if (eg >= 0.5 && leading) kingWeight -= 60;
  if (eg >= 0.8 && leading && opAll <= 2) kingWeight -= 90;

  const promoteWeight = W_BASE.promoteProgress + Math.round(6 * eg);

  let simplificationWeight = W_BASE.simplification;
  if (leading) simplificationWeight += Math.round(8 * eg);
  if (leading && opAll <= 2) simplificationWeight += 10;

  let score = 0;
  score += W_BASE.man * (myMenN - opMenN);
  score += kingWeight * (myKingsN - opKingsN);

  const myMob = mobility(p, p.side);
  const opSide = p.side === 1 ? -1 : 1;
  const opMob = mobility({ ...p, side: opSide } as Position, opSide);
  score += W_BASE.mobilityMen * (myMob.men - opMob.men);
  score += W_BASE.mobilityKing * (myMob.king - opMob.king);

  score += W_BASE.center * (centerScore(p, p.side) - centerScore(p, opSide));
  score += 10 * (runnerLaneBonus(p, p.side) - runnerLaneBonus(p, opSide));
  score += 8 * (supportScore(p, p.side) - supportScore(p, opSide));
  score += 6 * (edgeMenPenalty(p, opSide) - edgeMenPenalty(p, p.side));

  const myProm = promotionDistanceSum(p, p.side);
  const opProm = promotionDistanceSum(p, opSide);
  score += promoteWeight * (opProm - myProm) / 10;
  score += W_BASE.backRankGuard * (backRankGuards(p, p.side) - backRankGuards(p, opSide));
  score += W_BASE.kingProximity * (kingProximityGain(p, p.side) - kingProximityGain(p, opSide));
  score += W_BASE.trappedKing * (trappedKings(p, p.side) - trappedKings(p, opSide));
  score += cornerPressureBonus(p, p.side) - cornerPressureBonus(p, opSide);

  if (totalPieces <= 10 || eg >= 0.35) {
    const myCap = captureInfo(p, p.side);
    const opCap = captureInfo(p, opSide);
    if (myCap.maxChain || opCap.maxChain) {
      let captureWeight = W_BASE.captureSwing;
      if (eg >= 0.7) captureWeight += 20;
      score += captureWeight * (myCap.maxChain - opCap.maxChain);
    }
    if (myCap.targets || opCap.targets) {
      const threatWeight = W_BASE.captureTargets + Math.round(4 * eg);
      score += threatWeight * (myCap.targets - opCap.targets);
    }
  }

  score += simplificationWeight * leadSign * (START_TOTAL - (myAll + opAll));
  if (leading && opAll === 1) score += 140;
  if (leading && opAll <= 2) score += 70;

  return score | 0;
}
