// useCodexEngine.ts — async wrapper around the search engine
//
// Runs iterativeDeepening on the main thread, yielding between depth iterations
// via await setTimeout(0) to keep the UI responsive.
// (Web Worker support was removed: Metro bundles CommonJS, which rejects
//  import.meta syntax at parse time — a Worker-safe solution requires a
//  separate ESM entry point, which is out of scope for now.)

import { useCallback, useRef, useState } from 'react';
import { TT } from '../coreClaude/search/tt';
import { CancelToken, iterativeDeepening, SearchInfo } from '../coreClaude/search/alphabeta';
import { Position } from '../coreClaude/position';
import { Move } from '../coreClaude/movegen';
import { lookupOpeningBook } from '../coreClaude/search/openingBook';
import { Difficulty } from './types';

// Probability of skipping the book to add opening variety.
// Easy: high skip → more random; Hard: low skip → mostly follows book.
const BOOK_SKIP_RATE: Record<Difficulty, number> = { easy: 0.7, medium: 0.35, hard: 0.15 };

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
    difficulty: Difficulty = 'medium',
  ): Promise<Move | undefined> => {
    cancel();

    const token: CancelToken = { cancelled: false };
    cancelRef.current = token;
    setThinking(true);

    // Opening book — instant reply for known opening positions.
    // Randomly skip to create variety (higher skip rate for easier levels).
    const bookHit = lookupOpeningBook(pos);
    const skipBook = Math.random() < BOOK_SKIP_RATE[difficulty];
    if (bookHit && !skipBook) {
      return new Promise<Move | undefined>(resolve => {
        setTimeout(() => {
          setThinking(false);
          resolve(token.cancelled ? undefined : bookHit.move);
        }, 120);
      });
    }

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
