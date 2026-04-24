// useCodexEngine.ts - async wrapper around the hybrid engine
//
// Hybrid flow:
//   opening -> book
//   tactical / endgame -> alpha-beta
//   midgame -> AlphaZero MCTS

import { useCallback, useEffect, useRef, useState } from 'react';
import { TT } from '../coreClaude/search/tt';
import { CancelToken, iterativeDeepening, SearchInfo } from '../coreClaude/search/alphabeta';
import { Position } from '../coreClaude/position';
import { Move } from '../coreClaude/movegen';
import { preloadAZModel } from '../coreClaude/azNet';
import { azBestMove } from '../coreClaude/azMcts';
import { HybridDifficulty, HybridPlan, hybridBestMove } from '../coreClaude/search/hybrid';
import { Difficulty } from './types';

// Production profile: keep turns responsive on mobile while giving higher
// levels enough budget to avoid shallow tactical blunders.
//
// NOTE:
// - easy/normal/hard/expert use strict minimax from the screen layer.
// - master uses AZ-only route below (no hybrid).
const HYBRID_PROFILE: Record<Difficulty, { hybridDifficulty: HybridDifficulty; budgetMs: number }> = {
  easy: { hybridDifficulty: 'medium', budgetMs: 700 },
  normal: { hybridDifficulty: 'medium', budgetMs: 1200 },
  hard: { hybridDifficulty: 'hard', budgetMs: 2000 },
  expert: { hybridDifficulty: 'hard', budgetMs: 3000 },
  master: { hybridDifficulty: 'hard', budgetMs: 4000 },
};
const STRICT_NO_LIMIT_BUDGET_MS = 24 * 60 * 60 * 1000;
const AZ_ONLY_SIMS_MIN = 140;
const AZ_ONLY_SIMS_MAX = 900;
const AZ_ONLY_SIMS_PER_SECOND = 120;
const USE_ENGINE_WORKER = false;

function simsFromBudgetMs(ms: number, noTimeLimit: boolean): number {
  if (noTimeLimit) return AZ_ONLY_SIMS_MAX;
  const sims = Math.round((Math.max(400, ms) / 1000) * AZ_ONLY_SIMS_PER_SECOND);
  return Math.max(AZ_ONLY_SIMS_MIN, Math.min(AZ_ONLY_SIMS_MAX, sims));
}

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

export function useCodexEngine() {
  const [thinking, setThinking] = useState(false);
  const [lastInfo, setLastInfo] = useState<SearchInfo | null>(null);
  const [lastPlan, setLastPlan] = useState<HybridPlan | null>(null);

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
    preloadAZModel();
  }

  const handleStrictResult = useCallback((
    best: Move | undefined,
    depth: number,
    score: number,
    nodes: number,
    token: CancelToken,
    noTimeLimit: boolean,
    maxDepth: number,
    onInfo?: (info: SearchInfo) => void,
  ) => {
    setThinking(false);
    if (token.cancelled) return undefined;
    setLastPlan({
      mode: 'alphabeta',
      reason: noTimeLimit
        ? `strict mm depth ${maxDepth} (no time limit)`
        : `strict mm depth ${maxDepth}`,
    });
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
  ): Promise<Move | undefined> => {
    return iterativeDeepening(
      pos,
      budgetMs,
      ttRef.current,
      onInfo,
      historyHashes,
      token,
      maxDepth,
    ).then(res => handleStrictResult(
      res.best,
      res.depth,
      res.score,
      res.nodes,
      token,
      noTimeLimit,
      maxDepth,
      onInfo,
    )).catch(() => {
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

    const noTimeLimit = !Number.isFinite(ms) || ms <= 0;
    if (difficulty === 'master') {
      const budgetMs = noTimeLimit ? 0 : Math.max(450, ms);
      const sims = simsFromBudgetMs(budgetMs, noTimeLimit);
      return azBestMove(pos, sims).then(move => {
        setThinking(false);
        if (token.cancelled) return undefined;
        setLastPlan({
          mode: 'az',
          reason: noTimeLimit
            ? `az-only ${sims} sims (no time limit)`
            : `az-only ${sims} sims`,
        });
        const info = move
          ? ({ depth: 0, score: 0, nodes: sims, pv: [move] } as SearchInfo)
          : null;
        setLastInfo(info);
        if (info) onInfo?.(info);
        return move;
      }).catch(() => {
        setThinking(false);
        return undefined;
      });
    }

    const profile = HYBRID_PROFILE[difficulty];
    const budgetMs = noTimeLimit ? 0 : Math.max(350, Math.min(ms, profile.budgetMs));

    return hybridBestMove(
      pos,
      budgetMs,
      ttRef.current,
      historyHashes,
      token,
      profile.hybridDifficulty,
    ).then(res => {
      setThinking(false);
      if (token.cancelled) return undefined;
      setLastPlan(res.plan);
      setLastInfo(res.info);
      if (res.info) onInfo?.(res.info);
      return res.move;
    }).catch(() => {
      setThinking(false);
      return undefined;
    });
  }, [cancel]);

  const thinkStrict = useCallback((
    pos: Position,
    ms = 2000,
    historyHashes: number[] = [],
    maxDepth = 7,
    onInfo?: (info: SearchInfo) => void,
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

  return { think, thinkStrict, thinking, lastInfo, lastPlan, cancel };
}
