// useCodexEngine.ts — async wrapper around the search engine
//
// Runs iterativeDeepening on the main thread, yielding between depth iterations
// via await setTimeout(0) to keep the UI responsive.
// (Web Worker support was removed: Metro bundles CommonJS, which rejects
//  import.meta syntax at parse time — a Worker-safe solution requires a
//  separate ESM entry point, which is out of scope for now.)

import { useCallback, useRef, useState } from 'react';
import { TT } from '../coreCodex/search/tt';
import { CancelToken, iterativeDeepening, SearchInfo } from '../coreCodex/search/alphabeta';
import { Position } from '../coreCodex/position';
import { Move } from '../coreCodex/movegen';

export function useCodexEngine() {
  const [thinking, setThinking]   = useState(false);
  const [lastInfo, setLastInfo]   = useState<SearchInfo | null>(null);

  const ttRef     = useRef(new TT());
  const cancelRef = useRef<CancelToken | null>(null);

  const cancel = useCallback(() => {
    if (cancelRef.current) cancelRef.current.cancelled = true;
  }, []);

  const think = useCallback((
    pos: Position,
    ms = 1800,
    historyHashes: number[] = [],
    onInfo?: (info: SearchInfo) => void,
  ): Promise<Move | undefined> => {
    cancel();

    const token: CancelToken = { cancelled: false };
    cancelRef.current = token;
    setThinking(true);

    return iterativeDeepening(
      pos, ms, ttRef.current,
      (info) => {
        if (token.cancelled) return;
        setLastInfo(info);
        onInfo?.(info);
      },
      historyHashes,
      token,
    ).then(res => {
      setThinking(false);
      return token.cancelled ? undefined : res.best;
    }).catch(() => {
      setThinking(false);
      return undefined;
    });
  }, [cancel]);

  return { think, thinking, lastInfo, cancel };
}
