import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Alert, Button, Text, View } from 'react-native';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { hashPosition } from '../coreClaude/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { Difficulty, GameConfig } from './types';

const THINK_MS: Record<Difficulty, number> = { easy: 300, medium: 1000, hard: 2000 };
const ALGORITHM_NAME = 'Codex v2';

function posKey(p: Position) {
  return [p.side, p.p1Men, p.p1Kings, p.p2Men, p.p2Kings, p.halfmoveClock].join(':');
}

interface Props {
  config: GameConfig;
  onBack: () => void;
}

export default function HumanVsCodexArenaScreen({ config, onBack }: Props) {
  const { mode, difficulty, humanSide } = config;
  const isHvH   = mode === 'vs-human';
  const aiSide  = (-humanSide) as 1 | -1;
  const thinkMs = THINK_MS[difficulty];

  const [pos, setPos]                 = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel]                 = useState<number | null>(null);

  const { think, thinking, lastInfo, cancel } = useCodexEngine();
  const pendingRef = useRef<string | null>(null);
  const endgameRef = useRef<string | null>(null);

  const myMoves     = useMemo(() => generateMoves(pos), [pos]);
  const isDraw      = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash     = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );

  // In HvH both sides are human; in vs-AI only humanSide can tap
  const canHumanMove =
    (isHvH || pos.side === humanSide) &&
    !thinking && !isDraw && !isThreefold && myMoves.length > 0;

  function commitPosition(next: Position) {
    setPos(next);
    setHashHistory(prev => [...prev, hashPosition(next)]);
  }

  // ── Game over alerts ──────────────────────────────────────────────────────
  useEffect(() => {
    if (isDraw || isThreefold) {
      const k = posKey(pos) + (isThreefold ? ':rep' : ':draw');
      if (endgameRef.current !== k) {
        endgameRef.current = k;
        Alert.alert('จบเกม', isThreefold ? 'เสมอ — เดินซ้ำ 3 ครั้ง' : 'เสมอ — ไม่มีการกินนานเกินไป');
      }
      return;
    }
    if (!myMoves.length) {
      const k = posKey(pos) + ':nomoves';
      if (endgameRef.current !== k) {
        endgameRef.current = k;
        const winner = isHvH
          ? (pos.side === 1 ? 'ผู้เล่น 2 ชนะ! 🎉' : 'ผู้เล่น 1 ชนะ! 🎉')
          : (pos.side === humanSide ? `${ALGORITHM_NAME} ชนะ!` : 'คุณชนะ! 🎉');
        Alert.alert('จบเกม', winner);
      }
      return;
    }
    endgameRef.current = null;
  }, [isDraw, isThreefold, myMoves.length, pos]);

  // ── AI turn (vs-AI only) ──────────────────────────────────────────────────
  useEffect(() => {
    if (isHvH) return; // Human vs Human — no AI
    if (pos.side !== aiSide || isDraw || isThreefold || !myMoves.length) {
      pendingRef.current = null;
      return;
    }
    const k = posKey(pos);
    if (pendingRef.current === k) return; // already running for this position
    pendingRef.current = k;

    // Only one legal move → play instantly, no search needed
    if (myMoves.length === 1) {
      commitPosition(applyMove(pos, myMoves[0]));
      setSel(null);
      return;
    }

    const posSnapshot  = pos;
    const histSnapshot = hashHistory;

    think(posSnapshot, thinkMs, histSnapshot, undefined, difficulty).then(best => {
      if (pendingRef.current !== k) return;
      const move = best ?? generateMoves(posSnapshot)[0];
      if (move) commitPosition(applyMove(posSnapshot, move));
      setSel(null);
    });

    // NO cleanup return — intentional.
    // React Strict Mode double-invokes effects: run → cleanup → run.
    // The pendingRef guard `if (pendingRef.current === k) return` is sufficient
    // to prevent duplicate searches without a cleanup that would race.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pos.side, isDraw, isThreefold, myMoves.length]);

  // ── Human tap handler ─────────────────────────────────────────────────────
  function onTapSquare(i: number) {
    if (!canHumanMove) return;
    if (sel === null) {
      if (myMoves.some(m => m.from === i)) setSel(i);
      return;
    }
    const move = myMoves.find(m => m.from === sel && m.to === i);
    if (move) { commitPosition(applyMove(pos, move)); setSel(null); return; }
    setSel(myMoves.some(m => m.from === i) ? i : null);
  }

  function onNewGame() {
    cancel();
    pendingRef.current = null; endgameRef.current = null; setSel(null);
    const next = initialPosition();
    setPos(next); setHashHistory([hashPosition(next)]);
  }

  // ── Status text ───────────────────────────────────────────────────────────
  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}→${m.to}`).join(' ');

  let statusText: string;
  if (isDraw || isThreefold)  statusText = 'เสมอ';
  else if (!myMoves.length)   statusText = 'จบเกม';
  else if (isHvH)             statusText = pos.side === 1 ? 'ตาผู้เล่น 1 (●)' : 'ตาผู้เล่น 2 (●)';
  else                        statusText = pos.side === humanSide
    ? 'ตาของคุณ'
    : thinking ? `${ALGORITHM_NAME} กำลังคิด...` : `ตาของ ${ALGORITHM_NAME}`;

  const diffLabel     = isHvH ? '' : `  ·  ${difficulty === 'easy' ? 'ง่าย' : difficulty === 'medium' ? 'กลาง' : 'ยาก'}`;
  const opponentLabel = isHvH ? 'ผู้เล่น 2 (P2 ↓)' : `${ALGORITHM_NAME} (P2 ↓)`;

  return (
    <SafeAreaView style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 }}>
      <Text style={{ fontSize: 20, fontWeight: '700' }}>หมากหัว{diffLabel}</Text>
      <Text style={{ opacity: 0.8 }}>ผู้เล่น 1 (P1 ↑)  vs  {opponentLabel}</Text>
      <Text style={{ opacity: 0.9, fontWeight: '600' }}>{statusText}</Text>

      {!isHvH && lastInfo && (
        <Text style={{ fontSize: 12, opacity: 0.6 }}>
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

      <View style={{ flexDirection: 'row', gap: 12, marginTop: 10 }}>
        <Button title="เกมใหม่" onPress={onNewGame} />
        <Button title="← เมนู" onPress={() => { cancel(); onBack(); }} />
      </View>
    </SafeAreaView>
  );
}
