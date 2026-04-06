// useAZEngine.ts — React hook wrapping AlphaZero MCTS for UI use
import { useCallback, useRef, useState } from 'react';
import { azBestMove } from '../coreClaude/azMcts';
import { preloadAZModel } from '../coreClaude/azNet';
import { Position } from '../coreClaude/position';
import { Move } from '../coreClaude/movegen';

const N_SIMS = 200; // MCTS simulations per move (increase for stronger play)

export function useAZEngine() {
  const [thinking, setThinking] = useState(false);
  const cancelledRef = useRef(false);

  // Preload model on first call
  const preloaded = useRef(false);
  if (!preloaded.current) {
    preloaded.current = true;
    preloadAZModel();
  }

  const think = useCallback(async (pos: Position): Promise<Move | null> => {
    cancelledRef.current = false;
    setThinking(true);
    try {
      const move = await azBestMove(pos, N_SIMS);
      if (cancelledRef.current) return null;
      return move ?? null;
    } finally {
      if (!cancelledRef.current) setThinking(false);
    }
  }, []);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    setThinking(false);
  }, []);

  return { think, thinking, cancel };
}
