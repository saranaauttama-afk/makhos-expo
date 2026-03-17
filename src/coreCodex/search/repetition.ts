export type RepetitionCounts = Map<number, number>;

export function buildRepetitionCounts(hashes: number[]): RepetitionCounts {
  const counts = new Map<number, number>();
  for (const hash of hashes) {
    counts.set(hash, (counts.get(hash) ?? 0) + 1);
  }
  return counts;
}

export function getRepetitionCount(counts: RepetitionCounts, hash: number): number {
  return counts.get(hash) ?? 0;
}

export function pushRepetition(counts: RepetitionCounts, hash: number): number {
  const next = (counts.get(hash) ?? 0) + 1;
  counts.set(hash, next);
  return next;
}

export function popRepetition(counts: RepetitionCounts, hash: number) {
  const prev = counts.get(hash) ?? 0;
  if (prev <= 1) counts.delete(hash);
  else counts.set(hash, prev - 1);
}

export function isThreefoldRepetition(counts: RepetitionCounts, hash: number): boolean {
  return getRepetitionCount(counts, hash) >= 3;
}
