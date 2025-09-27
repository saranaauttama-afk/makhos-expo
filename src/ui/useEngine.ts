// src/ui/useEngine.ts
import { useCallback, useRef, useState } from 'react';
import { TT } from '../core/search/tt';
import { iterativeDeepening, SearchInfo } from '../core/search/alphabeta';
import { Position } from '../core/position';

export function useEngine() {
  const [thinking, setThinking] = useState(false);
  const [lastInfo, setLastInfo] = useState<SearchInfo | null>(null);
  const ttRef = useRef(new TT());

  const think = useCallback((pos: Position, ms = 600, onInfo?: (info: SearchInfo) => void) => {
    setThinking(true);
    const res = iterativeDeepening(pos, ms, ttRef.current, (info) => {
      setLastInfo(info);
      onInfo?.(info);
    });
    setThinking(false);
    return res.best;
  }, []);

  return { think, thinking, lastInfo };
}
