// useCodexEngine.ts - async wrapper around the hybrid engine
//
// Hybrid flow:
//   opening -> book
//   tactical / endgame -> alpha-beta
//   midgame -> AlphaZero MCTS
//   alpha -> Neural Network

import { useCallback, useEffect, useRef, useState } from 'react';
import { TT } from '../coreClaude/search/tt';
import { CancelToken, iterativeDeepening, moveKey, RootMoveScores, SearchInfo } from '../coreClaude/search/alphabeta';
import { lookupOpeningBookCandidates } from '../coreClaude/search/openingBook';
import { precomputeEndgameTablebase } from '../coreClaude/search/endgameTablebase';
import { selectStrictLevelMove, StrictDifficulty } from '../coreClaude/search/levelPolicy';
import { Position } from '../coreClaude/position';
import { Move, generateMoves } from '../coreClaude/movegen';
import { Difficulty } from './types';
import { evaluateNN, selectBestNNMove, initNNInference } from '../coreClaude/nnInference';

// Production profile: keep turns responsive on mobile while giving higher
// levels enough budget to avoid shallow tactical blunders.
const STRICT_NO_LIMIT_BUDGET_MS = 12_000;
const USE_ENGINE_WORKER = false;

type EngineWorkerLike = {
  postMessage: (message: unknown) => void;
  addEventListener?: (type: 'message', listener: (event: { data: unknown }) => void) => void;
  removeEventListener?: (type: 'message', listener: (event: { data: unknown }) => void) => void;
  onmessage?: ((event: { data: unknown }) => void) | null;
  terminate?: () => void;
};

type WorkerMessageInfo = {
  type: 'info';
  gen: number;
  depth: number;
  score: number;
  nodes: number;
  pv: Move[];
};

type WorkerMessageResult = {
  type: 'result';
  gen: number;
  best?: Move;
  score: number;
  nodes: number;
  depth: number;
};

type WorkerMessage = WorkerMessageInfo | WorkerMessageResult;

type PendingWorkerSearch = {
  gen: number;
  token: CancelToken;
  noTimeLimit: boolean;
  maxDepth: number;
  onInfo?: (info: SearchInfo) => void;
  resolve: (move: Move | undefined) => void;
};

// Initialize NN model once at startup
let nnInitialized = false;
async function ensureNNInitialized() {
  if (!nnInitialized) {
    try {
      await initNNInference();
      nnInitialized = true;
    } catch (error) {
      console.error('Failed to initialize NN model:', error);
      throw error;
    }
  }
}

export function useCodexEngine() {
  const [thinking, setThinking] = useState(false);
  const [lastInfo, setLastInfo] = useState<SearchInfo | null>(null);

  const ttRef = useRef(new TT());
  const cancelRef = useRef<CancelToken | null>(null);
  const workerRef = useRef<EngineWorkerLike | null>(null);
  const workerSupportedRef = useRef<boolean | null>(null);
  const workerGenRef = useRef(0);
  const pendingWorkerRef = useRef<PendingWorkerSearch | null>(null);
  const workerMessageHandlerRef = useRef<((event: { data: unknown }) => void) | null>(null);
  const preloaded = useRef(false);

  if (!preloaded.current) {
    preloaded.current = true;
    setTimeout(() => {
      precomputeEndgameTablebase().catch(() => {});
    }, 0);
  }

  const runNeuralNetwork = useCallback(async (
    pos: Position,
    token: CancelToken,
    onInfo?: (info: SearchInfo) => void,
    historyHashes: number[] = [],
  ): Promise<Move | undefined> => {
    try {
      // Ensure NN is initialized
      await ensureNNInitialized();

      if (token.cancelled) {
        setThinking(false);
        return undefined;
      }

      // Get legal moves
      const legalMoves = generateMoves(pos);
      if (legalMoves.length === 0) {
        setThinking(false);
        return undefined;
      }

      // Evaluate position with NN
      const { policyLogits, value } = await evaluateNN(pos);

      if (token.cancelled) {
        setThinking(false);
        return undefined;
      }

      // Select best move
      const bestMove = selectBestNNMove(policyLogits, legalMoves, pos);

      // Create info for display
      const info: SearchInfo = {
        depth: 0, // NN doesn't use depth
        score: Math.round(value * 100), // Convert to centipawn-like scale
        nodes: 1, // Single NN evaluation
        pv: bestMove ? [bestMove] : [],
      };

      setLastInfo(info);
      setThinking(false);
      if (onInfo) onInfo(info);

      return bestMove;
    } catch (error) {
      console.error('NN inference error, falling back to minimax:', error);

      // Show error to user
      const errorMsg = error instanceof Error ? error.message : String(error);
      setTimeout(() => {
        alert(`NN Error: ${errorMsg}\n\nFalling back to Expert Minimax (depth 12)`);
      }, 100);

      // Fallback to expert-level minimax (depth 12)
      return runStrictOnMainThread(
        pos,
        5000,
        historyHashes,
        12,
        token,
        onInfo,
        false,
        'expert',
      );
    }
  }, [runStrictOnMainThread]);

  const handleStrictResult = useCallback((
    best: Move | undefined,
    depth: number,
    score: number,
    nodes: number,
    token: CancelToken,
    noTimeLimit: boolean,
    maxDepth: number,
    guideLabel: string | undefined,
    onInfo?: (info: SearchInfo) => void,
  ) => {
    setThinking(false);
    if (token.cancelled) return undefined;
    const info: SearchInfo | null = depth > 0
      ? { depth, score, nodes, pv: best ? [best] : [] }
      : null;
    setLastInfo(info);
    if (info) onInfo?.(info);
    return best;
  }, []);

  const runStrictOnMainThread = useCallback((
    pos: Position,
    budgetMs: number,
    historyHashes: number[],
    maxDepth: number,
    token: CancelToken,
    onInfo?: (info: SearchInfo) => void,
    noTimeLimit = false,
    strictDifficulty?: StrictDifficulty,
  ): Promise<Move | undefined> => {
    return (async () => {
      const openingPly = historyHashes.length > 0 && historyHashes.length <= 14;
      const guideLabels: string[] = [];
      const blendedScores = new Map<number, number>();

      if (openingPly) {
        const book = lookupOpeningBookCandidates(pos, { source: 'ui' });
        if (book) {
          const bestWeight = Math.max(0.001, book.candidates[0]?.weight ?? 1);
          for (const candidate of book.candidates) {
            const key = moveKey(candidate.move);
            const hint = Math.max(0.05, candidate.weight / bestWeight);
            blendedScores.set(key, Math.max(blendedScores.get(key) ?? 0, hint));
          }
          guideLabels.push(book.candidates.length > 1 ? 'top-k opening book' : 'book-guided opening');
        }
      }

      // AZ guidance removed (Phase 0 cleanup)

      const rootMoveScores: RootMoveScores | undefined = blendedScores.size ? blendedScores : undefined;
      const diversifyRoot = openingPly && maxDepth >= 5;
      const res = await iterativeDeepening(
        pos,
        budgetMs,
        ttRef.current,
        onInfo,
        historyHashes,
        token,
        maxDepth,
        rootMoveScores,
        diversifyRoot,
      );
      const selected = strictDifficulty ? selectStrictLevelMove(strictDifficulty, pos, res) : res.best;
      return handleStrictResult(
        selected,
        res.depth,
        res.score,
        res.nodes,
        token,
        noTimeLimit,
        maxDepth,
        guideLabels.length ? guideLabels.join(', ') : undefined,
        onInfo,
      );
    })().catch(() => {
      setThinking(false);
      return undefined;
    });
  }, [handleStrictResult]);

  const handleWorkerMessage = useCallback((event: { data: unknown }) => {
    const msg = event.data as WorkerMessage;
    const pending = pendingWorkerRef.current;
    if (!pending || msg.gen !== pending.gen) return;

    if (msg.type === 'info') {
      const info: SearchInfo = {
        depth: msg.depth,
        score: msg.score,
        nodes: msg.nodes,
        pv: msg.pv ?? [],
      };
      setLastInfo(info);
      pending.onInfo?.(info);
      return;
    }

    pendingWorkerRef.current = null;
    const best = handleStrictResult(
      msg.best,
      msg.depth,
      msg.score,
      msg.nodes,
      pending.token,
      pending.noTimeLimit,
      pending.maxDepth,
      undefined,
      pending.onInfo,
    );
    pending.resolve(best);
  }, [handleStrictResult]);

  const ensureWorker = useCallback((): EngineWorkerLike | null => {
    if (!USE_ENGINE_WORKER) {
      workerSupportedRef.current = false;
      return null;
    }
    if (workerSupportedRef.current === false) return null;
    if (workerRef.current) return workerRef.current;

    // Disabled for Expo/Hermes until we wire a Hermes-safe worker bundler.
    workerSupportedRef.current = false;
    return null;
  }, [handleWorkerMessage]);

  const cancel = useCallback(() => {
    if (cancelRef.current) cancelRef.current.cancelled = true;

    const pending = pendingWorkerRef.current;
    if (pending) {
      pending.token.cancelled = true;
      pendingWorkerRef.current = null;
      try { workerRef.current?.postMessage({ type: 'cancel' }); } catch {}
      setThinking(false);
      pending.resolve(undefined);
      return;
    }
    try { workerRef.current?.postMessage({ type: 'cancel' }); } catch {}
  }, []);

  const think = useCallback((
    pos: Position,
    ms = 1800,
    historyHashes: number[] = [],
    onInfo?: (info: SearchInfo) => void,
    difficulty: Difficulty = 'easy',
  ): Promise<Move | undefined> => {
    cancel();

    const token: CancelToken = { cancelled: false };
    cancelRef.current = token;
    setThinking(true);

    // Alpha NN mode: use neural network for move selection
    if (difficulty === 'alpha') {
      return runNeuralNetwork(pos, token, onInfo, historyHashes);
    }

    const noTimeLimit = !Number.isFinite(ms) || ms <= 0;
    // Hybrid mode removed - fall back to strict mode
    const maxDepth = difficulty === 'easy' ? 4
      : difficulty === 'normal' ? 6
      : difficulty === 'hard' ? 9
      : 12;
    const budgetMs = noTimeLimit ? 0 : Math.max(350, ms);

    return runStrictOnMainThread(
      pos,
      budgetMs,
      historyHashes,
      maxDepth,
      token,
      onInfo,
      noTimeLimit,
      difficulty as StrictDifficulty,
    );
  }, [cancel, runNeuralNetwork, runStrictOnMainThread]);

  const thinkStrict = useCallback((
    pos: Position,
    ms = 2000,
    historyHashes: number[] = [],
    maxDepth = 7,
    onInfo?: (info: SearchInfo) => void,
    strictDifficulty?: StrictDifficulty,
  ): Promise<Move | undefined> => {
    cancel();

    const token: CancelToken = { cancelled: false };
    cancelRef.current = token;
    setThinking(true);

    const noTimeLimit = !Number.isFinite(ms) || ms <= 0;
    const budgetMs = noTimeLimit ? STRICT_NO_LIMIT_BUDGET_MS : Math.max(400, ms);
    const worker = ensureWorker();
    if (!worker) {
      return runStrictOnMainThread(
        pos,
        budgetMs,
        historyHashes,
        maxDepth,
        token,
        onInfo,
        noTimeLimit,
        strictDifficulty,
      );
    }

    const gen = ++workerGenRef.current;
    return new Promise<Move | undefined>((resolve, reject) => {
      pendingWorkerRef.current = {
        gen,
        token,
        noTimeLimit,
        maxDepth,
        onInfo,
        resolve,
      };
      try {
        worker.postMessage({
          type: 'search',
          gen,
          pos,
          timeMs: budgetMs,
          historyHashes,
          maxDepth,
        });
      } catch (error) {
        pendingWorkerRef.current = null;
        reject(error);
      }
    }).catch(() => {
      workerSupportedRef.current = false;
      return runStrictOnMainThread(
        pos,
        budgetMs,
        historyHashes,
        maxDepth,
        token,
        onInfo,
        noTimeLimit,
        strictDifficulty,
      );
    });
  }, [cancel, ensureWorker, runStrictOnMainThread]);

  useEffect(() => () => {
    const pending = pendingWorkerRef.current;
    pendingWorkerRef.current = null;
    if (pending) pending.resolve(undefined);

    const worker = workerRef.current;
    workerRef.current = null;
    workerSupportedRef.current = null;
    if (!worker) return;

    try { worker.postMessage({ type: 'cancel' }); } catch {}
    if (workerMessageHandlerRef.current && typeof worker.removeEventListener === 'function') {
      try { worker.removeEventListener('message', workerMessageHandlerRef.current); } catch {}
    } else if ('onmessage' in worker) {
      worker.onmessage = null;
    }
    try { worker.terminate?.(); } catch {}
  }, []);

  return { think, thinkStrict, thinking, lastInfo, cancel };
}
