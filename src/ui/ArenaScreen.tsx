import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { hashPosition } from '../coreClaude/search/zobrist';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { azBestMove } from '../coreClaude/azMcts';
import { preloadAZModel } from '../coreClaude/azNet';

type AgentType = 'human' | 'hand' | 'az';

const AGENT_LABELS: Record<AgentType, string> = {
  human: '👤 Human',
  hand:  '🤖 Hand',
  az:    '🧠 AZ',
};

const THINK_MS: Record<AgentType, number> = {
  human: 0,
  hand:  2000,
  az:    10000,
};

interface LogEntry {
  ply:     number;
  side:    1 | -1;
  agent:   AgentType;
  move:    Move;
}

function fmtMove(m: Move): string {
  const cap = m.captured.length > 0 ? `x${m.captured.length}` : '';
  const promo = m.promote ? '★' : '';
  return `${m.from}→${m.to}${cap}${promo}`;
}

function AgentBtn({ type, active, onPress }: { type: AgentType; active: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.agentBtn, active && styles.agentBtnActive]}>
      <Text style={[styles.agentText, active && styles.agentTextActive]}>{AGENT_LABELS[type]}</Text>
    </Pressable>
  );
}

interface Props { onBack: () => void }

export default function ArenaScreen({ onBack }: Props) {
  const [p1Agent, setP1Agent] = useState<AgentType>('hand');
  const [p2Agent, setP2Agent] = useState<AgentType>('az');

  const [pos, setPos]                 = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [log, setLog]                 = useState<LogEntry[]>([]);
  const [autoPlay, setAutoPlay]       = useState(false);
  const [elapsed, setElapsed]         = useState(0);   // ms since think started
  const [gameResult, setGameResult]   = useState<string | null>(null);

  const { think, thinking } = useCodexEngine();

  // Preload AZ model when arena opens
  useEffect(() => { preloadAZModel(); }, []);

  const steppingRef  = useRef(false);
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const thinkStart   = useRef(0);
  const logScrollRef = useRef<ScrollView>(null);

  // ── Derived ─────────────────────────────────────────────────────────────────
  const moves       = useMemo(() => generateMoves(pos), [pos]);
  const isDraw      = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash     = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const gameOver     = isDraw || isThreefold || moves.length === 0;
  const currentAgent = pos.side === 1 ? p1Agent : p2Agent;

  // ── Timer ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (thinking) {
      thinkStart.current = Date.now();
      timerRef.current = setInterval(() => setElapsed(Date.now() - thinkStart.current), 100);
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      setElapsed(0);
    }
    return () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  }, [thinking]);

  // ── Auto-scroll log ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (log.length > 0) {
      setTimeout(() => logScrollRef.current?.scrollToEnd({ animated: true }), 50);
    }
  }, [log.length]);

  // ── Core step ────────────────────────────────────────────────────────────────
  const step = useCallback(async (curPos: Position, curHistory: number[]) => {
    if (steppingRef.current) return;

    const agent = curPos.side === 1 ? p1Agent : p2Agent;
    if (agent === 'human') return;

    steppingRef.current = true;

    let move: Move | undefined;
    if (agent === 'az') {
      move = await azBestMove(curPos);
    } else {
      move = await think(curPos, THINK_MS[agent], curHistory);
    }
    steppingRef.current = false;

    if (!move) return;

    const next = applyMove(curPos, move);
    const nextHash = hashPosition(next);

    setPos(next);
    setHashHistory(prev => [...prev, nextHash]);
    setLog(prev => [...prev, {
      ply:   prev.length + 1,
      side:  curPos.side as 1 | -1,
      agent,
      move,
    }]);

    const nextMoves = generateMoves(next);
    const draw = isDrawByInactivity(next);
    const rep  = isThreefoldRepetition(
      buildRepetitionCounts([...curHistory, nextHash]),
      nextHash,
    );

    if (!nextMoves.length || draw || rep) {
      const label = curPos.side === 1 ? 'P1' : 'P2';
      if (!nextMoves.length) setGameResult(`${label} (${AGENT_LABELS[agent]}) ชนะ!`);
      else                   setGameResult('เสมอ');
      setAutoPlay(false);
    }
  }, [p1Agent, p2Agent, think]);

  // ── Auto-play loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!autoPlay || gameOver || thinking || steppingRef.current) return;
    if (currentAgent === 'human') return;

    const delay = setTimeout(() => {
      step(pos, hashHistory);
    }, 300);

    return () => clearTimeout(delay);
  }, [autoPlay, pos, thinking, gameOver, step]);

  // ── Actions ──────────────────────────────────────────────────────────────────
  function reset() {
    setAutoPlay(false);
    steppingRef.current = false;
    const start = initialPosition();
    setPos(start);
    setHashHistory([hashPosition(start)]);
    setLog([]);
    setGameResult(null);
  }

  function toggleAutoPlay() {
    if (gameOver) return;
    setAutoPlay(v => !v);
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  const sideLabel    = pos.side === 1 ? 'P1' : 'P2';
  const thinkSeconds = (elapsed / 1000).toFixed(1);
  const limitSeconds = (THINK_MS[currentAgent] / 1000).toFixed(0);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>← กลับ</Text>
        </Pressable>
        <Text style={styles.title}>Arena</Text>
        <Pressable onPress={reset} style={styles.resetBtn}>
          <Text style={styles.resetText}>↺ Reset</Text>
        </Pressable>
      </View>

      {/* Agent selection */}
      <View style={styles.agentRow}>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P1 ⬆</Text>
          {(['hand', 'az', 'human'] as AgentType[]).map(t => (
            <AgentBtn key={t} type={t} active={p1Agent === t} onPress={() => setP1Agent(t)} />
          ))}
        </View>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P2 ⬇</Text>
          {(['hand', 'az', 'human'] as AgentType[]).map(t => (
            <AgentBtn key={t} type={t} active={p2Agent === t} onPress={() => setP2Agent(t)} />
          ))}
        </View>
      </View>

      {/* Main area: board + log */}
      <View style={styles.mainRow}>
        {/* Board */}
        <Board
          pos={pos}
          onTapSquare={() => {}}
          fromSquares={[]}
          selectedFrom={null}
          destSquares={[]}
        />

        {/* Move log */}
        <View style={styles.logPanel}>
          <Text style={styles.logTitle}>Move Log</Text>
          <ScrollView ref={logScrollRef} style={styles.logScroll} showsVerticalScrollIndicator={false}>
            {log.map(entry => (
              <View key={entry.ply} style={styles.logEntry}>
                <Text style={[styles.logPly, entry.side === 1 ? styles.logP1 : styles.logP2]}>
                  {entry.ply}.
                </Text>
                <Text style={styles.logAgent}>{entry.side === 1 ? 'P1' : 'P2'}</Text>
                <Text style={styles.logMove}>{fmtMove(entry.move)}</Text>
              </View>
            ))}
          </ScrollView>
        </View>
      </View>

      {/* Status bar */}
      <View style={styles.statusRow}>
        {gameResult ? (
          <Text style={styles.resultText}>{gameResult}</Text>
        ) : thinking ? (
          <Text style={styles.thinkText}>
            {sideLabel} ({AGENT_LABELS[currentAgent]}) คิดอยู่… {thinkSeconds}s / {limitSeconds}s
          </Text>
        ) : (
          <Text style={styles.turnText}>
            ถึงตา {sideLabel} ({AGENT_LABELS[currentAgent]})  •  ply {log.length + 1}
          </Text>
        )}
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable
          onPress={toggleAutoPlay}
          disabled={gameOver}
          style={[styles.playBtn, autoPlay && styles.stopBtn, gameOver && styles.btnDisabled]}
        >
          <Text style={styles.playText}>{autoPlay ? '⏹ Stop' : '▶ Auto Play'}</Text>
        </Pressable>

        {!autoPlay && !gameOver && currentAgent !== 'human' && (
          <Pressable
            onPress={() => step(pos, hashHistory)}
            disabled={thinking || steppingRef.current}
            style={[styles.stepBtn, (thinking || steppingRef.current) && styles.btnDisabled]}
          >
            <Text style={styles.stepText}>Step</Text>
          </Pressable>
        )}
      </View>
    </SafeAreaView>
  );
}

const ACCENT = '#55aa33';
const P1_CLR = '#3388ff';
const P2_CLR = '#ff5533';

const styles = StyleSheet.create({
  container:   { flex: 1, alignItems: 'center', paddingHorizontal: 12, gap: 8 },
  header:      { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '100%', paddingTop: 4 },
  backBtn:     { padding: 8 },
  backText:    { fontSize: 15, color: ACCENT, fontWeight: '600' },
  resetBtn:    { padding: 8 },
  resetText:   { fontSize: 15, color: '#888', fontWeight: '600' },
  title:       { fontSize: 20, fontWeight: '800' },

  agentRow:    { flexDirection: 'row', gap: 10, width: '100%' },
  agentCol:    { flex: 1, gap: 4 },
  sideLabel:   { fontSize: 10, fontWeight: '700', opacity: 0.5, textTransform: 'uppercase', letterSpacing: 1 },
  agentBtn:    { paddingVertical: 6, borderRadius: 8, borderWidth: 2, borderColor: '#ddd', alignItems: 'center' },
  agentBtnActive:  { borderColor: ACCENT, backgroundColor: 'rgba(85,170,51,0.1)' },
  agentText:       { fontSize: 11, fontWeight: '600', color: '#aaa' },
  agentTextActive: { color: ACCENT },

  mainRow:     { flexDirection: 'row', gap: 8, width: '100%', flex: 1 },

  logPanel:    { flex: 1, borderRadius: 10, backgroundColor: '#f4f4f4', padding: 8 },
  logTitle:    { fontSize: 11, fontWeight: '700', opacity: 0.4, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 },
  logScroll:   { flex: 1 },
  logEntry:    { flexDirection: 'row', alignItems: 'center', gap: 4, paddingVertical: 2 },
  logPly:      { fontSize: 10, fontWeight: '700', width: 20 },
  logP1:       { color: P1_CLR },
  logP2:       { color: P2_CLR },
  logAgent:    { fontSize: 10, fontWeight: '600', color: '#888', width: 18 },
  logMove:     { fontSize: 11, fontWeight: '500', color: '#333' },

  statusRow:   { height: 22, justifyContent: 'center' },
  thinkText:   { fontSize: 13, fontWeight: '600', color: '#e8a020' },
  turnText:    { fontSize: 13, opacity: 0.6, fontWeight: '500' },
  resultText:  { fontSize: 15, fontWeight: '800', color: ACCENT },

  controls:    { flexDirection: 'row', gap: 10, paddingBottom: 8 },
  playBtn:     { backgroundColor: ACCENT, borderRadius: 12, paddingVertical: 12, paddingHorizontal: 32 },
  stopBtn:     { backgroundColor: '#cc3333' },
  stepBtn:     { backgroundColor: '#888', borderRadius: 12, paddingVertical: 12, paddingHorizontal: 24 },
  btnDisabled: { opacity: 0.4 },
  playText:    { fontSize: 16, fontWeight: '700', color: '#fff' },
  stepText:    { fontSize: 16, fontWeight: '700', color: '#fff' },
});
