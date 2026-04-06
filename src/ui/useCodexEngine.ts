// useCodexEngine.ts — async wrapper around the hybrid engine
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
import { HybridPlan, hybridBestMove } from '../coreClaude/search/hybrid';
import { Difficulty } from './types';

export function useCodexEngine() {
  const [thinking, setThinking]   = useState(false);
  const [lastInfo, setLastInfo]   = useState<SearchInfo | null>(null);
  const [lastPlan, setLastPlan]   = useState<HybridPlan | null>(null);

  const ttRef     = useRef(new TT());
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
    difficulty: Difficulty = 'medium',
  ): Promise<Move | undefined> => {
    cancel();

    const token: CancelToken = { cancelled: false };
    cancelRef.current = token;
    setThinking(true);

    return hybridBestMove(
      pos,
      ms,
      ttRef.current,
      historyHashes,
      token,
      difficulty,
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
