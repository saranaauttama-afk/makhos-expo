// engineWorker.ts — Search engine running in a Worker thread
//
// Runs in a separate JS thread (Web Worker on web; RN Worker on Android/iOS
// with New Architecture).  No React, no UI — pure computation only.
//
// Protocol (main → worker):
//   { type: 'search', gen, pos, timeMs, historyHashes }
//   { type: 'cancel' }
//
// Protocol (worker → main):
//   { type: 'info',   gen, depth, score, nodes, pv }
//   { type: 'result', gen, best, score, nodes, depth }

import { iterativeDeepening, CancelToken } from '../coreClaude/search/alphabeta';
import { TT } from '../coreClaude/search/tt';
import { Position } from '../coreClaude/position';
import { Move } from '../coreClaude/movegen';

// One TT instance lives for the entire worker lifetime (reused across searches)
const tt = new TT();
let currentToken: CancelToken = { cancelled: false };

// Use addEventListener for compatibility across web and RN Worker environments
addEventListener('message', async (e: MessageEvent) => {
  const msg = e.data as
    | { type: 'cancel' }
    | { type: 'search'; gen: number; pos: Position; timeMs: number; historyHashes: number[] };

  if (msg.type === 'cancel') {
    // Cancel the in-flight search; result handler will see token.cancelled = true
    currentToken.cancelled = true;
    return;
  }

  if (msg.type === 'search') {
    // Cancel any previous search before starting a new one
    currentToken.cancelled = true;
    const token: CancelToken = { cancelled: false };
    currentToken = token;

    const { gen, pos, timeMs, historyHashes } = msg;

    const result = await iterativeDeepening(
      pos,
      timeMs,
      tt,
      // onInfo: forward depth/score/nodes to the main thread as it searches
      (info) => {
        if (token.cancelled) return;
        postMessage({ type: 'info', gen, ...info });
      },
      historyHashes,
      token,
    );

    // Always send a result — main thread ignores stale generations via gen check
    postMessage({
      type: 'result',
      gen,
      best:  result.best   as Move | undefined,
      score: result.score,
      nodes: result.nodes,
      depth: result.depth,
    });
  }
});
