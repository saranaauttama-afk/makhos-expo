// src/ui/useCodexEngine.ts
import { useCallback, useRef, useState } from 'react';
import { TT } from '../coreCodex/search/tt';
import { iterativeDeepening, SearchInfo } from '../coreCodex/search/alphabeta';
import { Position } from '../coreCodex/position';

export function useCodexEngine() {
  const [thinking, setThinking] = useState(false);
  const [lastInfo, setLastInfo] = useState<SearchInfo | null>(null);
  const ttRef = useRef(new TT());

  const think = useCallback((pos: Position, ms = 600, historyHashes: number[] = [], onInfo?: (info: SearchInfo) => void) => {
    setThinking(true);
    const res = iterativeDeepening(pos, ms, ttRef.current, (info) => {
      setLastInfo(info);
      onInfo?.(info);
    }, historyHashes);
    setThinking(false);
    return res.best;
  }, []);

  return { think, thinking, lastInfo };
}
