import { B1, bitCount } from '../bitboards';
import { applyMove, generateMoves, Move } from '../movegen';
import { isDrawByInactivity, Position } from '../position';
import {
  buildRepetitionCounts,
  getRepetitionCount,
  isThreefoldRepetition,
  popRepetition,
  pushRepetition,
  RepetitionCounts,
} from './repetition';
import { hashPosition } from './zobrist';

type Outcome = -1 | 0 | 1;

interface SolveResult {
  outcome: Outcome;
  dtm: number;
  bestMoveKey: number;
}

export interface EndgameProbe {
  score: number;
  best?: Move;
  dtm: number;
  exact: boolean;
  nodes?: number;
}

export interface DeterministicEndgameResult {
  probe?: EndgameProbe;
  nodes: number;
  limitReached: boolean;
  limitReason?: 'nodes' | 'depth' | 'cycle';
}

const TABLEBASE_WIN = 500_000;
const NO_MOVE_KEY = -1;
const sharedMemo = new Map<string, SolveResult>();

export function clearEndgameTablebaseCache(): void {
  sharedMemo.clear();
  _tablebaseReady = false;
}

// Precompute status — true once background precomputation finishes
let _tablebaseReady = false;
let _tablebasePrecompute: Promise<void> | undefined;
export function isEndgameTablebaseReady(): boolean { return _tablebaseReady; }

function keyMove(move: Move) {
  return (move.from << 5) | move.to;
}

function stateKey(pos: Position, repetitionCount: number) {
  return [
    pos.side,
    pos.p1Men >>> 0,
    pos.p1Kings >>> 0,
    pos.p2Men >>> 0,
    pos.p2Kings >>> 0,
    pos.halfmoveClock,
    repetitionCount,
  ].join(':');
}

function deterministicStateKey(pos: Position, repetitionCounts: RepetitionCounts): string {
  const history = [...repetitionCounts]
    .filter(([, count]) => count > 0)
    .sort(([a], [b]) => a - b)
    .map(([hash, count]) => `${hash >>> 0}.${Math.min(3, count)}`)
    .join(',');
  return `${pos.side}:${pos.p1Men >>> 0}:${pos.p1Kings >>> 0}:${pos.p2Men >>> 0}:${pos.p2Kings >>> 0}:${pos.halfmoveClock}:${history}`;
}

function chooseBetter(current: SolveResult | undefined, candidate: SolveResult): SolveResult {
  if (!current) return candidate;
  if (candidate.outcome !== current.outcome) {
    return candidate.outcome > current.outcome ? candidate : current;
  }
  if (candidate.outcome === 1) {
    return candidate.dtm < current.dtm ? candidate : current;
  }
  if (candidate.outcome === -1) {
    return candidate.dtm > current.dtm ? candidate : current;
  }
  return candidate.dtm < current.dtm ? candidate : current;
}

function scoreFromSolve(result: SolveResult): number {
  if (result.outcome > 0) return TABLEBASE_WIN - result.dtm;
  if (result.outcome < 0) return -TABLEBASE_WIN + result.dtm;
  return 0;
}

function canProbe(pos: Position): boolean {
  const totalPieces = bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
  const totalMen = bitCount(pos.p1Men | pos.p2Men);
  // Extended: solve 4-piece endgames with ≤2 men (covers KMvKM, KKvMM, etc.)
  return (totalPieces <= 4 && totalMen <= 2) || totalPieces <= 3;
}

// Max recursion depth for the endgame solver.  Flying-king positions can have
// very long forced mates; cap at 60 plies to prevent stack overflow on mobile.
const SOLVE_MAX_DEPTH = 60;

function solveNode(
  pos: Position,
  repetitionCounts: RepetitionCounts,
  visiting: Set<string>,
  depth = 0,
): SolveResult {
  // Hard depth cap — return "unknown draw" rather than overflowing the stack
  if (depth >= SOLVE_MAX_DEPTH) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const hash = hashPosition(pos);
  const repCount = Math.min(3, getRepetitionCount(repetitionCounts, hash));
  const key = stateKey(pos, repCount);

  if (isDrawByInactivity(pos) || isThreefoldRepetition(repetitionCounts, hash)) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const cached = sharedMemo.get(key);
  if (cached) return cached;
  if (visiting.has(key)) {
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  }

  const moves = generateMoves(pos);
  if (moves.length === 0) {
    const terminal = { outcome: -1 as Outcome, dtm: 0, bestMoveKey: NO_MOVE_KEY };
    sharedMemo.set(key, terminal);
    return terminal;
  }

  visiting.add(key);
  let best: SolveResult | undefined;

  for (const move of moves) {
    const child = applyMove(pos, move);
    const childHash = hashPosition(child);
    pushRepetition(repetitionCounts, childHash);
    const childResult = solveNode(child, repetitionCounts, visiting, depth + 1);
    popRepetition(repetitionCounts, childHash);

    const candidate: SolveResult = {
      outcome: (-childResult.outcome) as Outcome,
      dtm: childResult.dtm + 1,
      bestMoveKey: keyMove(move),
    };
    best = chooseBetter(best, candidate);

    if (best.outcome === 1 && best.dtm === 1) break;
  }

  visiting.delete(key);
  const resolved = best ?? { outcome: 0 as Outcome, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  sharedMemo.set(key, resolved);
  return resolved;
}

function findBestMove(pos: Position, bestMoveKey: number): Move | undefined {
  if (bestMoveKey < 0) return undefined;
  const from = (bestMoveKey >> 5) & 31;
  const to = bestMoveKey & 31;
  return generateMoves(pos).find((move) => move.from === from && move.to === to);
}

// ── Root-level probe (called once per think(), not in search hot path) ────────
// maxMs: give up and return undefined if computation exceeds this budget.
// After background precompute finishes the budget is irrelevant — every hit is
// an instant Map.get() on sharedMemo.
export function probeSmallEndgame(
  pos: Position,
  historyHashes: number[] = [],
  maxMs = 3000,
): EndgameProbe | undefined {
  if (!canProbe(pos)) return undefined;

  const hash = hashPosition(pos);
  const normalizedHistory = historyHashes.length ? historyHashes : [hash];
  const repetitionCounts = buildRepetitionCounts(normalizedHistory);
  if ((repetitionCounts.get(hash) ?? 0) === 0) {
    repetitionCounts.set(hash, 1);
  }

  // Fast path: if precompute already cached this exact state, return instantly
  const repCount = Math.min(3, getRepetitionCount(repetitionCounts, hash));
  const fastKey = stateKey(pos, repCount);
  const cached = sharedMemo.get(fastKey);
  if (cached) {
    return { score: scoreFromSolve(cached), best: findBestMove(pos, cached.bestMoveKey), dtm: cached.dtm, exact: true };
  }

  // Slow path: compute with time budget so we never hang mid-game
  const deadline = Date.now() + maxMs;
  const result = solveNodeBudgeted(pos, repetitionCounts, new Set<string>(), 0, deadline);
  if (!result) return undefined; // budget exceeded

  return {
    score: scoreFromSolve(result),
    best: findBestMove(pos, result.bestMoveKey),
    dtm: result.dtm,
    exact: true,
  };
}

/** Fixed-work, history-complete oracle intended for regression fixtures. */
export function probeSmallEndgameDeterministic(
  pos: Position,
  historyHashes: number[] = [],
  nodeLimit = 100_000,
  maxDepth = SOLVE_MAX_DEPTH,
): DeterministicEndgameResult {
  if (!Number.isInteger(nodeLimit) || nodeLimit < 1)
    throw new Error(`nodeLimit must be a positive integer, got ${nodeLimit}`);
  if (!Number.isInteger(maxDepth) || maxDepth < 0)
    throw new Error(`maxDepth must be a non-negative integer, got ${maxDepth}`);
  if (!canProbe(pos)) return { nodes: 0, limitReached: false };

  const hash = hashPosition(pos);
  const repetitions = buildRepetitionCounts(historyHashes.length ? historyHashes : [hash]);
  if (getRepetitionCount(repetitions, hash) === 0) repetitions.set(hash, 1);
  const memo = new Map<string, SolveResult>();
  const visiting = new Set<string>();
  let nodes = 0;
  let incompleteReason: DeterministicEndgameResult['limitReason'];

  function solve(current: Position, depth: number): SolveResult | undefined {
    if (nodes >= nodeLimit) {
      incompleteReason ??= 'nodes';
      return undefined;
    }
    nodes++;
    const currentHash = hashPosition(current);
    const key = deterministicStateKey(current, repetitions);
    if (isDrawByInactivity(current) || isThreefoldRepetition(repetitions, currentHash))
      return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };
    // A search horizon is not game evidence and must never become an exact draw.
    if (depth >= maxDepth) {
      incompleteReason ??= 'depth';
      return undefined;
    }
    const cached = memo.get(key);
    if (cached) return cached;
    // A DFS back-edge is not itself a threefold repetition. Real repetition
    // draws are handled above from the explicit history counts.
    if (visiting.has(key)) {
      incompleteReason ??= 'cycle';
      return undefined;
    }
    const moves = generateMoves(current);
    if (!moves.length) return { outcome: -1, dtm: 0, bestMoveKey: NO_MOVE_KEY };

    visiting.add(key);
    let best: SolveResult | undefined;
    let sawIncomplete = false;
    for (const move of moves) {
      const child = applyMove(current, move);
      const childHash = hashPosition(child);
      pushRepetition(repetitions, childHash);
      const childResult = solve(child, depth + 1);
      popRepetition(repetitions, childHash);
      if (!childResult) {
        sawIncomplete = true;
        continue;
      }
      best = chooseBetter(best, {
        outcome: (-childResult.outcome) as Outcome,
        dtm: childResult.dtm + 1,
        bestMoveKey: keyMove(move),
      });
      if (best.outcome === 1 && best.dtm === 1) break;
    }
    visiting.delete(key);
    // A proven winning continuation resolves the node. A loss or draw is only
    // exact after every legal continuation has resolved.
    if (best?.outcome === 1) {
      memo.set(key, best);
      return best;
    }
    if (sawIncomplete) return undefined;
    const resolved = best!;
    memo.set(key, resolved);
    return resolved;
  }

  const solved = solve(pos, 0);
  if (!solved) return { nodes, limitReached: true, limitReason: incompleteReason ?? 'cycle' };
  return {
    nodes,
    limitReached: false,
    probe: {
      score: scoreFromSolve(solved),
      best: findBestMove(pos, solved.bestMoveKey),
      dtm: solved.dtm,
      exact: true,
      nodes,
    },
  };
}

// solveNode variant that aborts when the deadline fires.
// Returns undefined if time ran out (caller falls through to regular search).
function solveNodeBudgeted(
  pos: Position,
  repetitionCounts: RepetitionCounts,
  visiting: Set<string>,
  depth: number,
  deadline: number,
): SolveResult | undefined {
  if (Date.now() > deadline) return undefined;
  if (depth >= SOLVE_MAX_DEPTH) return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };

  const hash = hashPosition(pos);
  const repCount = Math.min(3, getRepetitionCount(repetitionCounts, hash));
  const key = stateKey(pos, repCount);

  if (isDrawByInactivity(pos) || isThreefoldRepetition(repetitionCounts, hash))
    return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };

  const cached2 = sharedMemo.get(key);
  if (cached2) return cached2;
  if (visiting.has(key)) return { outcome: 0, dtm: 0, bestMoveKey: NO_MOVE_KEY };

  const moves = generateMoves(pos);
  if (moves.length === 0) {
    const terminal: SolveResult = { outcome: -1, dtm: 0, bestMoveKey: NO_MOVE_KEY };
    sharedMemo.set(key, terminal);
    return terminal;
  }

  visiting.add(key);
  let best: SolveResult | undefined;

  for (const move of moves) {
    const child = applyMove(pos, move);
    const childHash = hashPosition(child);
    pushRepetition(repetitionCounts, childHash);
    const childResult = solveNodeBudgeted(child, repetitionCounts, visiting, depth + 1, deadline);
    popRepetition(repetitionCounts, childHash);

    if (!childResult) { visiting.delete(key); return undefined; } // budget blown

    const candidate: SolveResult = {
      outcome: (-childResult.outcome) as Outcome,
      dtm: childResult.dtm + 1,
      bestMoveKey: keyMove(move),
    };
    best = chooseBetter(best, candidate);
    if (best.outcome === 1 && best.dtm === 1) break;
  }

  visiting.delete(key);
  const resolved = best ?? { outcome: 0 as Outcome, dtm: 0, bestMoveKey: NO_MOVE_KEY };
  sharedMemo.set(key, resolved);
  return resolved;
}

// ── Background precomputation ─────────────────────────────────────────────────
// Enumerate all positions satisfying canProbe() and warm up sharedMemo so that
// every probe during a game is an instant Map.get().
//
// Runs as an async background task — yields every BATCH positions so the JS
// event loop stays responsive.  Typical runtime: 1–3 s on a mid-range device.
//
// onProgress(done, total) — optional callback for loading indicators.

const SQUARES = 32;
const BATCH   = 20; // positions per yield — keep UI frame budget small

function combinations(n: number, k: number): number[][] {
  const result: number[][] = [];
  const combo: number[] = [];
  function pick(start: number) {
    if (combo.length === k) { result.push([...combo]); return; }
    for (let i = start; i <= n - (k - combo.length); i++) {
      combo.push(i); pick(i + 1); combo.pop();
    }
  }
  pick(0);
  return result;
}

export function precomputeEndgameTablebase(
  onProgress?: (done: number, total: number) => void,
): Promise<void> {
  if (_tablebaseReady) return Promise.resolve();
  if (!_tablebasePrecompute) {
    _tablebasePrecompute = precomputeEndgameTablebaseImpl(onProgress).finally(() => {
      if (!_tablebaseReady) _tablebasePrecompute = undefined;
    });
  }
  return _tablebasePrecompute;
}

async function precomputeEndgameTablebaseImpl(
  onProgress?: (done: number, total: number) => void,
): Promise<void> {

  // Piece configurations satisfying canProbe():
  //   (a) totalPieces ≤ 3, both sides have ≥ 1 piece
  //   (b) totalPieces ≤ 4, totalMen = 0 (kings only)
  type Cfg = [p1m: number, p1k: number, p2m: number, p2k: number];
  const configs: Cfg[] = [];

  for (let p1m = 0; p1m <= 3; p1m++)
  for (let p1k = 0; p1k <= 3; p1k++)
  for (let p2m = 0; p2m <= 3; p2m++)
  for (let p2k = 0; p2k <= 3; p2k++) {
    const total = p1m + p1k + p2m + p2k;
    const men   = p1m + p2m;
    const hasP1 = (p1m + p1k) > 0;
    const hasP2 = (p2m + p2k) > 0;
    if (!hasP1 || !hasP2) continue;
    // Only precompute ≤3-piece positions — fast to solve, highest practical value.
    // 4-king (total=4, men=0) positions are too numerous (71k+) and complex; they
    // are handled on-demand by probeSmallEndgame with a 3 s budget instead.
    if (total <= 3) configs.push([p1m, p1k, p2m, p2k]);
  }

  // Count total positions to enumerate (for progress reporting)
  function countPositions(cfg: Cfg): number {
    const [p1m, p1k, p2m, p2k] = cfg;
    const n = p1m + p1k + p2m + p2k;
    if (n > SQUARES) return 0;
    // C(32, n) * 2 sides (we'll enumerate per-side inside the loop)
    let c = 1;
    for (let i = 0; i < n; i++) c = c * (SQUARES - i) / (i + 1);
    return Math.round(c) * 2;
  }

  const totalEstimate = configs.reduce((s, cfg) => s + countPositions(cfg), 0);
  let done = 0;

  const rep0 = buildRepetitionCounts([]);

  for (const [p1m, p1k, p2m, p2k] of configs) {
    const n = p1m + p1k + p2m + p2k;
    const allCombos = combinations(SQUARES, n);

    for (const squares of allCombos) {
      for (const side of [1, -1] as const) {
        // Assign squares: first p1m to P1 men, next p1k to P1 kings, etc.
        let idx = 0;
        let p1Men = 0, p1Kings = 0, p2Men = 0, p2Kings = 0;
        for (let i = 0; i < p1m; i++) p1Men   |= B1(squares[idx++]);
        for (let i = 0; i < p1k; i++) p1Kings |= B1(squares[idx++]);
        for (let i = 0; i < p2m; i++) p2Men   |= B1(squares[idx++]);
        for (let i = 0; i < p2k; i++) p2Kings |= B1(squares[idx++]);

        const pos: Position = { side, p1Men: p1Men >>> 0, p1Kings: p1Kings >>> 0, p2Men: p2Men >>> 0, p2Kings: p2Kings >>> 0, halfmoveClock: 0 };

        // Warm up sharedMemo — 5 ms budget is plenty for ≤3-piece positions.
        solveNodeBudgeted(pos, rep0, new Set<string>(), 0, Date.now() + 5);

        done++;
        if (done % BATCH === 0) {
          onProgress?.(done, totalEstimate);
          // Yield to event loop so UI stays responsive
          await new Promise<void>(r => setTimeout(r, 0));
        }
      }
    }
  }

  _tablebaseReady = true;
  onProgress?.(done, done);
}

export function probeSmallEndgameFromCounts(pos: Position, repetitionCounts: RepetitionCounts): EndgameProbe | undefined {
  if (!canProbe(pos)) return undefined;
  const result = solveNode(pos, repetitionCounts, new Set<string>());
  return {
    score: scoreFromSolve(result),
    best: findBestMove(pos, result.bestMoveKey),
    dtm: result.dtm,
    exact: true,
  };
}
