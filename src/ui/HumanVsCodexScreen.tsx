import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Button, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { applyMove, generateMoves, Move } from '../coreCodex/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreCodex/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreCodex/search/repetition';
import { hashPosition } from '../coreCodex/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';

const HUMAN_SIDE = 1 as const;
const AI_SIDE = -1 as const;
const AI_THINK_MS = 1200;
const ALGORITHM_NAME = 'Codex algor';

function positionKey(p: Position) {
  return [p.side, p.p1Men, p.p1Kings, p.p2Men, p.p2Kings, p.halfmoveClock].join(':');
}

function noMoveMessage(p: Position) {
  return p.side === HUMAN_SIDE ? 'Player 1 has no legal moves' : 'Player 2 (AI) has no legal moves';
}

export default function HumanVsCodexScreen() {
  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel] = useState<number | null>(null);
  const { think, thinking, lastInfo } = useCodexEngine();
  const pendingAIRef = useRef<string | null>(null);
  const endgameRef = useRef<string | null>(null);

  const myMoves = useMemo(() => generateMoves(pos), [pos]);
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const currentHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), currentHash),
    [currentHash, hashHistory],
  );
  const canHumanMove = pos.side === HUMAN_SIDE && !thinking && !isDraw && !isThreefold && myMoves.length > 0;

  function commitPosition(next: Position) {
    setPos(next);
    setHashHistory((prev) => [...prev, hashPosition(next)]);
  }

  useEffect(() => {
    if (isDraw || isThreefold) {
      const key = `${positionKey(pos)}:${isThreefold ? 'repetition' : 'draw'}`;
      if (endgameRef.current !== key) {
        endgameRef.current = key;
        Alert.alert('Game over', isThreefold ? 'Draw by repetition' : 'Draw by inactivity (20 turns without capture)');
      }
      return;
    }

    if (myMoves.length === 0) {
      const key = `${positionKey(pos)}:nomoves`;
      if (endgameRef.current !== key) {
        endgameRef.current = key;
        Alert.alert('Game over', noMoveMessage(pos));
      }
      return;
    }

    endgameRef.current = null;
  }, [isDraw, isThreefold, myMoves.length, pos]);

  useEffect(() => {
    if (pos.side !== AI_SIDE || isDraw || isThreefold || myMoves.length === 0) {
      pendingAIRef.current = null;
      return;
    }

    const key = positionKey(pos);
    if (pendingAIRef.current === key) return;
    pendingAIRef.current = key;

    const timer = setTimeout(() => {
      const best = think(pos, AI_THINK_MS, hashHistory);
      if (!best) return;
      const next = applyMove(pos, best);
      commitPosition(next);
      setSel(null);
    }, 150);

    return () => clearTimeout(timer);
  }, [hashHistory, isDraw, isThreefold, myMoves.length, pos, think]);

  function onTapSquare(i: number) {
    if (!canHumanMove) return;

    const candidates = myMoves.filter((m) => m.from === (sel ?? i));
    const to = candidates.find((m) => m.to === i);

    if (sel === null) {
      if (candidates.length) setSel(i);
      return;
    }

    if (to) {
      commitPosition(applyMove(pos, to));
      setSel(null);
      return;
    }

    setSel(null);
  }

  function onNewGame() {
    pendingAIRef.current = null;
    endgameRef.current = null;
    setSel(null);
    const next = initialPosition();
    setPos(next);
    setHashHistory([hashPosition(next)]);
  }

  const fromSquares = sel !== null ? [sel] : [];
  const destSquares =
    sel !== null
      ? myMoves.filter((m) => m.from === sel).map((m) => ({ to: m.to, caps: m.captured.length }))
      : [];
  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}-${m.to}`).join(' ');
  const statusText = isDraw || isThreefold
    ? 'Draw'
    : myMoves.length === 0
      ? 'Game Over'
      : pos.side === HUMAN_SIDE
        ? 'Turn: P1 (Human)'
        : `Turn: P2 (AI)${thinking ? ' - thinking...' : ''}`;

  return (
    <SafeAreaView style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 }}>
      <Text style={{ fontSize: 20, fontWeight: '700' }}>Makhos (Thai Checkers)</Text>
      <Text style={{ opacity: 0.9 }}>Player 1 = Human | Player 2 = AI ({ALGORITHM_NAME})</Text>
      <Text style={{ opacity: 0.85 }}>{statusText}</Text>
      <Text style={{ fontSize: 12, opacity: 0.72 }}>
        {ALGORITHM_NAME}: iterative deepening + alpha-beta + TT + quiescence
      </Text>
      {lastInfo && (
        <View style={{ alignItems: 'center', gap: 2 }}>
          <Text style={{ fontSize: 12, opacity: 0.72 }}>
            depth {lastInfo.depth} | score {lastInfo.score} | nodes {lastInfo.nodes}
          </Text>
          {pvText ? (
            <Text style={{ fontSize: 11, opacity: 0.6 }}>
              pv {pvText}
            </Text>
          ) : null}
        </View>
      )}

      <Board
        pos={pos}
        onTapSquare={onTapSquare}
        fromSquares={fromSquares}
        selectedFrom={sel}
        destSquares={destSquares}
      />

      <View style={{ flexDirection: 'row', gap: 12, marginTop: 10 }}>
        <Button title="New Game" onPress={onNewGame} />
      </View>
    </SafeAreaView>
  );
}
