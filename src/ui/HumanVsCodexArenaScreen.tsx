import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Alert, Button, Platform, Text, View } from 'react-native';
import { applyMove, generateMoves, Move } from '../coreCodex/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreCodex/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreCodex/search/repetition';
import { hashPosition } from '../coreCodex/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';

const HUMAN_SIDE  = 1 as const;
const AI_SIDE     = -1 as const;
const AI_THINK_MS = Platform.OS === 'web' ? 600 : 1200;
const ALGORITHM_NAME = 'Codex v2';

function posKey(p: Position) {
  return [p.side, p.p1Men, p.p1Kings, p.p2Men, p.p2Kings, p.halfmoveClock].join(':');
}

export default function HumanVsCodexArenaScreen() {
  const [pos, setPos]               = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel]               = useState<number | null>(null);
  const { think, thinking, lastInfo, cancel } = useCodexEngine();
  const pendingRef  = useRef<string | null>(null);
  const endgameRef  = useRef<string | null>(null);

  const myMoves  = useMemo(() => generateMoves(pos), [pos]);
  const isDraw   = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash  = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const canHumanMove = pos.side === HUMAN_SIDE && !thinking && !isDraw && !isThreefold && myMoves.length > 0;

  function commitPosition(next: Position) {
    setPos(next);
    setHashHistory(prev => [...prev, hashPosition(next)]);
  }

  // Game over alerts
  useEffect(() => {
    if (isDraw || isThreefold) {
      const k = posKey(pos) + (isThreefold ? ':rep' : ':draw');
      if (endgameRef.current !== k) {
        endgameRef.current = k;
        Alert.alert('Game over', isThreefold ? 'Draw by repetition' : 'Draw by inactivity');
      }
      return;
    }
    if (!myMoves.length) {
      const k = posKey(pos) + ':nomoves';
      if (endgameRef.current !== k) {
        endgameRef.current = k;
        Alert.alert('Game over', pos.side === HUMAN_SIDE ? 'You have no moves — AI wins!' : 'AI has no moves — You win!');
      }
      return;
    }
    endgameRef.current = null;
  }, [isDraw, isThreefold, myMoves.length, pos]);

  // AI turn
  useEffect(() => {
    if (pos.side !== AI_SIDE || isDraw || isThreefold || !myMoves.length) {
      pendingRef.current = null;
      return;
    }
    const k = posKey(pos);
    if (pendingRef.current === k) return; // already running for this position
    pendingRef.current = k;

    // Capture pos snapshot for the closure — safe because pos only changes via commitPosition.
    // Only one legal move → play instantly, no search needed.
    // Covers forced-capture positions where there is exactly one piece to take.
    if (myMoves.length === 1) {
      commitPosition(applyMove(pos, myMoves[0]));
      setSel(null);
      return;
    }

    const posSnapshot = pos;
    const histSnapshot = hashHistory;

    think(posSnapshot, AI_THINK_MS, histSnapshot).then(best => {
      // Discard result if the position changed while we were searching
      // (e.g. user pressed New Game).
      if (pendingRef.current !== k) return;

      const move = best ?? generateMoves(posSnapshot)[0]; // re-generate to avoid stale closure
      if (move) commitPosition(applyMove(posSnapshot, move));
      setSel(null);
    });

    // NO cleanup return here — intentional.
    //
    // React Strict Mode double-invokes effects: run → cleanup → run.
    // If cleanup resets pendingRef to null, the second invocation sees
    // null ≠ k and starts a SECOND search for the same position.
    // The first (cancelled) search then commits a random fallback move,
    // and setThinking(false) is never called → game freezes forever.
    //
    // The pendingRef guard  `if (pendingRef.current === k) return`  is
    // sufficient: the second invocation finds pendingRef already set to k
    // (by the first) and returns early without starting another search.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pos.side, isDraw, isThreefold, myMoves.length]);
  // NOTE: intentionally minimal deps — we don't want to re-trigger mid-search

  function onTapSquare(i: number) {
    if (!canHumanMove) return;
    if (sel === null) {
      if (myMoves.some(m => m.from === i)) setSel(i);
      return;
    }
    const to = myMoves.find(m => m.from === sel && m.to === i);
    if (to) { commitPosition(applyMove(pos, to)); setSel(null); return; }
    setSel(myMoves.some(m => m.from === i) ? i : null);
  }

  function onNewGame() {
    cancel(); // stop any in-flight search immediately
    pendingRef.current = null; endgameRef.current = null; setSel(null);
    const next = initialPosition();
    setPos(next); setHashHistory([hashPosition(next)]);
  }

  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}→${m.to}`).join(' ');
  const status = isDraw || isThreefold ? 'Draw'
    : !myMoves.length ? 'Game Over'
    : pos.side === HUMAN_SIDE ? 'Your turn'
    : thinking ? `${ALGORITHM_NAME} thinking...` : `${ALGORITHM_NAME}'s turn`;

  return (
    <SafeAreaView style={{ flex:1, alignItems:'center', justifyContent:'center', gap:8 }}>
      <Text style={{ fontSize:20, fontWeight:'700' }}>Makhos (Thai Checkers)</Text>
      <Text style={{ opacity:0.8 }}>You (P1 ↑) vs {ALGORITHM_NAME} (P2 ↓)</Text>
      <Text style={{ opacity:0.9, fontWeight:'600' }}>{status}</Text>

      {lastInfo && (
        <Text style={{ fontSize:12, opacity:0.65 }}>
          depth {lastInfo.depth} | score {lastInfo.score} | nodes {lastInfo.nodes}
          {pvText ? `\npv: ${pvText}` : ''}
        </Text>
      )}

      <Board
        pos={pos}
        onTapSquare={onTapSquare}
        fromSquares={sel !== null ? [sel] : []}
        selectedFrom={sel}
        destSquares={sel !== null ? myMoves.filter(m => m.from === sel).map(m => ({ to: m.to, caps: m.captured.length })) : []}
      />

      <View style={{ flexDirection:'row', gap:12, marginTop:10 }}>
        <Button title="New Game" onPress={onNewGame} />
      </View>
    </SafeAreaView>
  );
}
