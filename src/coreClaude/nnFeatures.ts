/**
 * nnFeatures.ts - Feature extraction for Neural Network V2
 *
 * Extracts 320 float32 features from a Thai Checkers position:
 * - [0-127]:   Positional features (same as V1)
 * - [128-191]: Mobility features (NEW!)
 * - [192-255]: Threat maps (NEW!)
 * - [256-287]: Hanging piece flags (NEW!)
 * - [288-319]: Distance to promotion (NEW!)
 *
 * All features are side-to-move relative (board flipped for P2)
 */

import { bits, bitCount, STEPS } from './bitboards';
import { Position } from './position';
import { generateMoves, Move } from './movegen';

/**
 * Extract 320 features from position
 */
export function extractNNFeatures(pos: Position): Float32Array {
  const features = new Float32Array(320);

  // Determine my/opponent pieces (relative to side-to-move)
  const myMen = pos.side === 1 ? pos.p1Men : pos.p2Men;
  const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen = pos.side === 1 ? pos.p2Men : pos.p1Men;
  const opKings = pos.side === 1 ? pos.p2Kings : pos.p1Kings;

  const flipBoard = pos.side === -1; // Flip board for P2

  // Helper: map square index (flip if P2)
  const mapSquare = (sq: number) => (flipBoard ? 31 - sq : sq);

  // ────────────────────────────────────────────────────────────────────────────
  // [0-127] Positional Features (Binary)
  // ────────────────────────────────────────────────────────────────────────────

  for (const sq of bits(myMen)) features[mapSquare(sq)] = 1;
  for (const sq of bits(myKings)) features[32 + mapSquare(sq)] = 1;
  for (const sq of bits(opMen)) features[64 + mapSquare(sq)] = 1;
  for (const sq of bits(opKings)) features[96 + mapSquare(sq)] = 1;

  // ────────────────────────────────────────────────────────────────────────────
  // [128-191] Mobility Features (Float - normalized legal moves per square)
  // ────────────────────────────────────────────────────────────────────────────

  const myMobility = calculateMobility(pos, true);
  const opMobility = calculateMobility(pos, false);

  for (let sq = 0; sq < 32; sq++) {
    // Normalize: 0-4 moves → 0.0-1.0
    features[128 + mapSquare(sq)] = myMobility[sq] / 4.0;
    features[160 + mapSquare(sq)] = opMobility[sq] / 4.0;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // [192-255] Threat Maps (Binary - squares attacked by pieces)
  // ────────────────────────────────────────────────────────────────────────────

  const myThreats = calculateThreats(pos, true);
  const opThreats = calculateThreats(pos, false);

  for (let sq = 0; sq < 32; sq++) {
    features[192 + mapSquare(sq)] = myThreats[sq] ? 1 : 0;
    features[224 + mapSquare(sq)] = opThreats[sq] ? 1 : 0;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // [256-287] Hanging Piece Flags (Binary - undefended + threatened)
  // ────────────────────────────────────────────────────────────────────────────

  const myHanging = calculateHanging(pos, myMen, myKings, myThreats, opThreats);
  const opHanging = calculateHanging(
    { ...pos, side: (pos.side === 1 ? -1 : 1) as 1 | -1 },
    opMen,
    opKings,
    opThreats,
    myThreats
  );

  for (let sq = 0; sq < 32; sq++) {
    features[256 + mapSquare(sq)] = myHanging[sq] ? 1 : 0;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // [288-319] Distance to Promotion (Float - normalized 0-7 rows → 0.0-1.0)
  // ────────────────────────────────────────────────────────────────────────────

  const myPromoDist = calculatePromoDistance(myMen, pos.side);
  const opPromoDist = calculatePromoDistance(opMen, pos.side === 1 ? -1 : 1);

  for (let sq = 0; sq < 32; sq++) {
    features[288 + mapSquare(sq)] = myPromoDist[sq];
  }

  // Note: Only 304 features used (288+16), remaining 16 reserved for future

  return features;
}

/**
 * Calculate mobility for each square (number of legal moves)
 */
function calculateMobility(pos: Position, forMySide: boolean): number[] {
  const mobility = new Array(32).fill(0);

  // Generate moves for the appropriate side
  const testPos: Position = forMySide
    ? pos
    : { ...pos, side: (pos.side === 1 ? -1 : 1) as 1 | -1 };

  const moves = generateMoves(testPos);

  for (const move of moves) {
    mobility[move.from]++;
  }

  return mobility;
}

/**
 * Calculate threat map (squares that can be captured by pieces)
 */
function calculateThreats(pos: Position, forMySide: boolean): boolean[] {
  const threats = new Array(32).fill(false);

  const myMen = forMySide
    ? pos.side === 1
      ? pos.p1Men
      : pos.p2Men
    : pos.side === 1
    ? pos.p2Men
    : pos.p1Men;

  const myKings = forMySide
    ? pos.side === 1
      ? pos.p1Kings
      : pos.p2Kings
    : pos.side === 1
    ? pos.p2Kings
    : pos.p1Kings;

  const side = forMySide ? pos.side : (pos.side === 1 ? -1 : 1);

  // For each piece, mark squares it can attack
  for (const sq of bits(myMen | myKings)) {
    const isKing = (myKings & (1 << sq)) !== 0;
    const moves = STEPS[sq];

    for (const step of moves) {
      const toSq = step.to;
      const dir = step.dir;

      // Men: only forward directions
      if (!isKing) {
        if (side === 1 && (dir === 'DL' || dir === 'DR')) continue;
        if (side === -1 && (dir === 'UL' || dir === 'UR')) continue;
      }

      threats[toSq] = true;

      // Kings: check capture range (diagonal rays)
      if (isKing) {
        let cur = toSq;
        while (true) {
          const next = STEPS[cur]?.find(s => s.dir === dir)?.to;
          if (next === undefined) break;
          threats[next] = true;
          cur = next;
        }
      }
    }
  }

  return threats;
}

/**
 * Calculate hanging pieces (threatened + undefended)
 */
function calculateHanging(
  pos: Position,
  myMen: number,
  myKings: number,
  myThreats: boolean[],
  opThreats: boolean[]
): boolean[] {
  const hanging = new Array(32).fill(false);

  for (const sq of bits(myMen | myKings)) {
    const threatened = opThreats[sq];
    const defended = myThreats[sq];

    if (threatened && !defended) {
      hanging[sq] = true;
    }
  }

  return hanging;
}

/**
 * Calculate distance to promotion for men
 * Returns normalized distance: 0.0 (far) to 1.0 (promotion row)
 */
function calculatePromoDistance(men: number, side: 1 | -1): number[] {
  const distance = new Array(32).fill(0);

  for (const sq of bits(men)) {
    const row = Math.floor(sq / 4);

    // P1 promotes at row 0 (moving up), P2 at row 7 (moving down)
    const distToPromo = side === 1 ? row : 7 - row;

    // Normalize: 0-7 → 1.0-0.0 (closer = higher value)
    distance[sq] = 1.0 - distToPromo / 7.0;
  }

  return distance;
}

/**
 * Convert features to string for debugging
 */
export function featuresToString(features: Float32Array): string {
  const lines: string[] = [];

  lines.push('Positional Features [0-127]:');
  lines.push(`  My men:    ${countNonZero(features, 0, 32)}`);
  lines.push(`  My kings:  ${countNonZero(features, 32, 64)}`);
  lines.push(`  Opp men:   ${countNonZero(features, 64, 96)}`);
  lines.push(`  Opp kings: ${countNonZero(features, 96, 128)}`);

  lines.push('Mobility Features [128-191]:');
  lines.push(`  My mobility:  ${sumRange(features, 128, 160).toFixed(1)}`);
  lines.push(`  Opp mobility: ${sumRange(features, 160, 192).toFixed(1)}`);

  lines.push('Threat Maps [192-255]:');
  lines.push(`  My threats:  ${countNonZero(features, 192, 224)}`);
  lines.push(`  Opp threats: ${countNonZero(features, 224, 256)}`);

  lines.push('Hanging Pieces [256-287]:');
  lines.push(`  My hanging: ${countNonZero(features, 256, 288)}`);

  lines.push('Promotion Distance [288-319]:');
  lines.push(`  My promo dist: ${sumRange(features, 288, 304).toFixed(1)}`);

  return lines.join('\n');
}

function countNonZero(arr: Float32Array, start: number, end: number): number {
  let count = 0;
  for (let i = start; i < end; i++) {
    if (arr[i] > 0) count++;
  }
  return count;
}

function sumRange(arr: Float32Array, start: number, end: number): number {
  let sum = 0;
  for (let i = start; i < end; i++) {
    sum += arr[i];
  }
  return sum;
}
