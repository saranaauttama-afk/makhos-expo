// azFeatures.ts — Feature extraction for AlphaZero network
//
// 128-dim float32 vector (side-to-move relative, board flipped for P2):
//   x[  0.. 31] = 1 if my man at square i      (flipped: 31-sq for P2)
//   x[ 32.. 63] = 1 if my king at square i
//   x[ 64.. 95] = 1 if enemy man at square i
//   x[ 96..127] = 1 if enemy king at square i
//
// Flipping ensures the model always sees "my pieces at the bottom, moving up"
// regardless of which side is to move.

import { bits } from './bitboards';
import { Position } from './position';

export function getFeatures(pos: Position): Float32Array {
  const x = new Float32Array(128);
  const myMen   = pos.side === 1 ? pos.p1Men   : pos.p2Men;
  const myKings = pos.side === 1 ? pos.p1Kings : pos.p2Kings;
  const opMen   = pos.side === 1 ? pos.p2Men   : pos.p1Men;
  const opKings = pos.side === 1 ? pos.p2Kings : pos.p1Kings;

  if (pos.side === 1) {
    for (const sq of bits(myMen))   x[sq]           = 1;
    for (const sq of bits(myKings)) x[32 + sq]      = 1;
    for (const sq of bits(opMen))   x[64 + sq]      = 1;
    for (const sq of bits(opKings)) x[96 + sq]      = 1;
  } else {
    for (const sq of bits(myMen))   x[31 - sq]      = 1;
    for (const sq of bits(myKings)) x[32 + 31 - sq] = 1;
    for (const sq of bits(opMen))   x[64 + 31 - sq] = 1;
    for (const sq of bits(opKings)) x[96 + 31 - sq] = 1;
  }

  return x;
}
