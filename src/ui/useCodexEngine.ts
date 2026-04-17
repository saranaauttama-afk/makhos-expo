// useCodexEngine.ts - async wrapper around the hybrid engine
//
// Hybrid flow:
//   opening -> book
//   tactical / endgame -> alpha-beta
//   midgame -> AlphaZero MCTS

import { useCallback, useRef, useState } from 'react';
import { TT } from '../coreClaude/search/tt';
import { CancelToken, SearchInfo } from '../coreClaude/search/alphabeta';
import { Position } from '../coreClaude/position';
import { Move } from '../coreClaude/movegen';
import { preloadAZModel } from '../coreClaude/azNet';
import { HybridDifficulty, HybridPlan, hybridBestMove } from '../coreClaude/search/hybrid';
import { Difficulty } from './types';

const HYBRID_PROFILE: Record<Difficulty, { hybridDifficulty: HybridDifficulty; budgetMs: number }> = {
  easy: { hybridDifficulty: 'medium', budgetMs: 900 },
  normal: { hybridDifficulty: 'medium', budgetMs: 1400 },
  hard: { hybridDifficulty: 'hard', budgetMs: 2200 },
  expert: { hybridDifficulty: 'hard', budgetMs: 3200 },
  master: { hybridDifficulty: 'hard', budgetMs: 4500 },
};

export function useCodexEngine() {
  const [thinking, setThinking] = useState(false);
  const [lastInfo, setLastInfo] = useState<SearchInfo | null>(null);
  const [lastPlan, setLastPlan] = useState<HybridPlan | null>(null);

  const ttRef = useRef(new TT());
  const cancelRef = useRef<CancelToken | null>(null);
  const preloaded = useRef(false);

  if (!preloaded.current) {
    preloaded.current = true;
    preloadAZModel();
  }

  const cancel = useCallback(() => {
    if (cancelRef.current) cancelRef.current.cancelled = true;
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

    const profile = HYBRID_PROFILE[difficulty];
    const budgetMs = Math.max(350, Math.min(ms, profile.budgetMs));

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

  return { think, thinking, lastInfo, lastPlan, cancel };
}
