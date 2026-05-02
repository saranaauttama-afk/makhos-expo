import { azInfer } from '../azNet';
import { getFeatures } from '../azFeatures';
import { generateMoves, Move } from '../movegen';
import { Position } from '../position';
import { moveKey, RootMoveScores } from './alphabeta';

function policyIndex(pos: Position, move: Move): number {
  return pos.side === 1
    ? move.from * 32 + move.to
    : (31 - move.from) * 32 + (31 - move.to);
}

export async function getAZRootMoveScores(
  pos: Position,
  moves: Move[] = generateMoves(pos),
): Promise<RootMoveScores | undefined> {
  if (moves.length <= 1) return undefined;

  const { policyLogits } = await azInfer(getFeatures(pos));
  let maxLogit = -Infinity;
  const indices = moves.map(move => policyIndex(pos, move));
  for (const idx of indices) {
    const value = policyLogits[idx];
    if (value > maxLogit) maxLogit = value;
  }

  let sum = 0;
  const expScores = indices.map(idx => {
    const score = Math.exp(policyLogits[idx] - maxLogit);
    sum += score;
    return score;
  });
  if (!Number.isFinite(sum) || sum <= 0) return undefined;

  const scores = new Map<number, number>();
  for (let i = 0; i < moves.length; i++) {
    scores.set(moveKey(moves[i]), expScores[i] / sum);
  }
  return scores;
}
