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
import { usePixelGameFx } from './usePixelGameFx';

type AgentType = 'human' | 'hand' | 'az';

const AGENT_LABELS: Record<AgentType, string> = {
  human: 'Human',
  hand: 'AB',
  az: 'AZ',
};

const THINK_MS: Record<AgentType, number> = {
  human: 0,
  hand: 2000,
  az: 10000,
};

interface LogEntry {
  ply: number;
  side: 1 | -1;
  agent: AgentType;
  move: Move;
}

function fmtMove(m: Move): string {
  const cap = m.captured.length > 0 ? `x${m.captured.length}` : '';
  const promo = m.promote ? 'K' : '';
  return `${m.from}->${m.to}${cap}${promo}`;
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

  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [autoPlay, setAutoPlay] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [gameResult, setGameResult] = useState<string | null>(null);
  const [lastMove, setLastMove] = useState<{ from: number; to: number; captured: number; promote: boolean } | null>(null);
  const [shakeFrame, setShakeFrame] = useState(0);
  const [comboFrame, setComboFrame] = useState(0);
  const [comboText, setComboText] = useState('');

  const { think, thinking } = useCodexEngine();
  const { triggerFx } = usePixelGameFx();

  useEffect(() => { preloadAZModel(); }, []);

  const steppingRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const thinkStart = useRef(0);
  const logScrollRef = useRef<ScrollView>(null);

  const moves = useMemo(() => generateMoves(pos), [pos]);
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const gameOver = isDraw || isThreefold || moves.length === 0;
  const currentAgent = pos.side === 1 ? p1Agent : p2Agent;
  const shakeX = [0, -6, 5, -4, 3, -2, 0][Math.min(shakeFrame, 6)];
  const comboOpacity = [0, 0.75, 1, 1, 0.9, 0.7, 0.45, 0.2, 0][Math.min(comboFrame, 8)];
  const comboLift = [24, 18, 14, 10, 6, 2, -2, -6, -10][Math.min(comboFrame, 8)];

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

  useEffect(() => {
    if (log.length > 0) {
      setTimeout(() => logScrollRef.current?.scrollToEnd({ animated: true }), 50);
    }
  }, [log.length]);

  const step = useCallback(async (curPos: Position, curHistory: number[]) => {
    if (steppingRef.current) return;

    const agent = curPos.side === 1 ? p1Agent : p2Agent;
    if (agent === 'human') return;

    steppingRef.current = true;

    let move: Move | undefined;
    if (agent === 'az') {
      try {
        move = await azBestMove(curPos);
      } catch {
        move = await think(curPos, THINK_MS.hand, curHistory);
      }
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
      ply: prev.length + 1,
      side: curPos.side as 1 | -1,
      agent,
      move,
    }]);
    setLastMove({ from: move.from, to: move.to, captured: move.captured.length, promote: move.promote });

    const nextMoves = generateMoves(next);
    const draw = isDrawByInactivity(next);
    const rep = isThreefoldRepetition(
      buildRepetitionCounts([...curHistory, nextHash]),
      nextHash,
    );

    if (!nextMoves.length || draw || rep) {
      const label = curPos.side === 1 ? 'P1' : 'P2';
      if (!nextMoves.length) setGameResult(`${label} (${AGENT_LABELS[agent]}) wins`);
      else setGameResult('Draw');
      setAutoPlay(false);
    }
  }, [p1Agent, p2Agent, think]);

  useEffect(() => {
    if (!autoPlay || gameOver || thinking || steppingRef.current) return;
    if (currentAgent === 'human') return;

    const delay = setTimeout(() => {
      void step(pos, hashHistory);
    }, 300);

    return () => clearTimeout(delay);
  }, [autoPlay, pos, thinking, gameOver, step, currentAgent, hashHistory]);

  useEffect(() => {
    if (!lastMove) {
      setShakeFrame(0);
      return;
    }

    if (lastMove.captured > 1) {
      triggerFx('combo');
      let frame = 0;
      setShakeFrame(0);
      const shakeId = setInterval(() => {
        frame += 1;
        setShakeFrame(frame);
        if (frame >= 6) clearInterval(shakeId);
      }, 42);
      return () => clearInterval(shakeId);
    }

    if (lastMove.promote) triggerFx('promote');
    else if (lastMove.captured > 0) triggerFx('capture');
    else triggerFx('move');

    setShakeFrame(0);
    return undefined;
  }, [lastMove, triggerFx]);

  useEffect(() => {
    if (!lastMove) {
      setComboFrame(0);
      setComboText('');
      return;
    }

    let nextText = '';
    if (lastMove.promote && lastMove.captured > 1) nextText = `ROYAL ${lastMove.captured}X!`;
    else if (lastMove.promote) nextText = 'KING UP!';
    else if (lastMove.captured > 1) nextText = `${lastMove.captured}X COMBO!`;

    if (!nextText) {
      setComboFrame(0);
      setComboText('');
      return;
    }

    setComboText(nextText);
    setComboFrame(0);
    let frame = 0;
    const comboId = setInterval(() => {
      frame += 1;
      setComboFrame(frame);
      if (frame >= 8) clearInterval(comboId);
    }, 90);
    return () => clearInterval(comboId);
  }, [lastMove]);

  useEffect(() => {
    if (gameResult) triggerFx('victory');
  }, [gameResult, triggerFx]);

  function reset() {
    setAutoPlay(false);
    steppingRef.current = false;
    const start = initialPosition();
    setPos(start);
    setHashHistory([hashPosition(start)]);
    setLog([]);
    setGameResult(null);
    setLastMove(null);
  }

  function toggleAutoPlay() {
    if (gameOver) return;
    setAutoPlay(v => !v);
  }

  const sideLabel = pos.side === 1 ? 'P1' : 'P2';
  const thinkSeconds = (elapsed / 1000).toFixed(1);
  const limitSeconds = (THINK_MS[currentAgent] / 1000).toFixed(0);

  return (
    <SafeAreaView style={styles.container}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />

      <View style={styles.header}>
        <Pressable onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>BACK</Text>
        </Pressable>
        <Text style={styles.title}>Arena Lab</Text>
        <Pressable onPress={reset} style={styles.resetBtn}>
          <Text style={styles.resetText}>RESET</Text>
        </Pressable>
      </View>

      <View style={styles.agentRow}>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P1 TOP</Text>
          {(['hand', 'az', 'human'] as AgentType[]).map(t => (
            <AgentBtn key={t} type={t} active={p1Agent === t} onPress={() => setP1Agent(t)} />
          ))}
        </View>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P2 BOT</Text>
          {(['hand', 'az', 'human'] as AgentType[]).map(t => (
            <AgentBtn key={t} type={t} active={p2Agent === t} onPress={() => setP2Agent(t)} />
          ))}
        </View>
      </View>

      <View style={styles.mainRow}>
        <View style={styles.boardStage}>
          {!!comboText && comboFrame < 8 && (
            <View
              style={[
                styles.comboBadge,
                {
                  opacity: comboOpacity,
                  transform: [{ translateY: comboLift }],
                },
              ]}
            >
              <Text style={styles.comboBadgeText}>{comboText}</Text>
            </View>
          )}

          <View style={{ transform: [{ translateX: shakeX }] }}>
            <Board
              pos={pos}
              onTapSquare={() => {}}
              fromSquares={[]}
              selectedFrom={null}
              destSquares={[]}
              lastMove={lastMove}
            />
          </View>
        </View>

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

      <View style={styles.statusRow}>
        {gameResult ? (
          <Text style={styles.resultText}>{gameResult}</Text>
        ) : thinking ? (
          <Text style={styles.thinkText}>
            {sideLabel} ({AGENT_LABELS[currentAgent]}) thinking {thinkSeconds}s / {limitSeconds}s
          </Text>
        ) : (
          <Text style={styles.turnText}>
            Turn {sideLabel} ({AGENT_LABELS[currentAgent]}) - ply {log.length + 1}
          </Text>
        )}
      </View>

      <View style={styles.controls}>
        <Pressable
          onPress={toggleAutoPlay}
          disabled={gameOver}
          style={[styles.playBtn, autoPlay && styles.stopBtn, gameOver && styles.btnDisabled]}
        >
          <Text style={styles.playText}>{autoPlay ? 'STOP' : 'AUTO PLAY'}</Text>
        </Pressable>

        {!autoPlay && !gameOver && currentAgent !== 'human' && (
          <Pressable
            onPress={() => { void step(pos, hashHistory); }}
            disabled={thinking || steppingRef.current}
            style={[styles.stepBtn, (thinking || steppingRef.current) && styles.btnDisabled]}
          >
            <Text style={styles.stepText}>STEP</Text>
          </Pressable>
        )}
      </View>
    </SafeAreaView>
  );
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const MINT = '#b8f3df';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';
const P1_CLR = '#9be7da';
const P2_CLR = '#f2c5c5';

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', paddingHorizontal: 14, paddingTop: 8, paddingBottom: 10, gap: 10, backgroundColor: BG },
  bgAuraLarge: {
    position: 'absolute',
    width: 540,
    height: 540,
    borderRadius: 270,
    backgroundColor: 'rgba(174, 235, 223, 0.17)',
    top: -260,
    left: -90,
  },
  bgAuraSmall: {
    position: 'absolute',
    width: 340,
    height: 340,
    borderRadius: 170,
    backgroundColor: 'rgba(98, 174, 163, 0.26)',
    bottom: -140,
    right: -100,
  },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '100%', padding: 8, borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL },
  backBtn: { minHeight: 30, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK, justifyContent: 'center', paddingHorizontal: 10 },
  backText: { fontSize: 11, color: WHITE, fontWeight: '800', letterSpacing: 0.6 },
  resetBtn: { minHeight: 30, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK, justifyContent: 'center', paddingHorizontal: 10 },
  resetText: { fontSize: 11, color: SOFT, fontWeight: '800', letterSpacing: 0.6 },
  title: { fontSize: 17, fontWeight: '900', color: WHITE, letterSpacing: 0.6 },

  agentRow: { flexDirection: 'row', gap: 10, width: '100%' },
  agentCol: { flex: 1, gap: 5, borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL, padding: 8 },
  sideLabel: { fontSize: 10, fontWeight: '800', color: GOLD, textTransform: 'uppercase', letterSpacing: 0.8 },
  agentBtn: { minHeight: 30, borderRadius: 999, borderWidth: 1, borderColor: LINE, backgroundColor: PANEL_DARK, alignItems: 'center', justifyContent: 'center' },
  agentBtnActive: { borderColor: MINT, backgroundColor: '#2f6b62' },
  agentText: { fontSize: 11, fontWeight: '700', color: SOFT, letterSpacing: 0.4 },
  agentTextActive: { color: WHITE },

  mainRow: { flexDirection: 'row', gap: 10, width: '100%', flex: 1 },
  boardStage: { alignItems: 'center', justifyContent: 'center', overflow: 'hidden', padding: 8, borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL },
  comboBadge: { position: 'absolute', top: 8, zIndex: 4, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 10, backgroundColor: '#2f6b62', borderWidth: 1, borderColor: GOLD },
  comboBadgeText: { fontSize: 11, fontWeight: '900', color: GOLD, letterSpacing: 0.7 },

  logPanel: { flex: 1, borderRadius: 12, borderWidth: 1, borderColor: LINE, backgroundColor: PANEL, padding: 8 },
  logTitle: { fontSize: 11, fontWeight: '800', color: GOLD, textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 4 },
  logScroll: { flex: 1 },
  logEntry: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingVertical: 2 },
  logPly: { fontSize: 10, fontWeight: '700', width: 20 },
  logP1: { color: P1_CLR },
  logP2: { color: P2_CLR },
  logAgent: { fontSize: 10, fontWeight: '700', color: SOFT, width: 18 },
  logMove: { fontSize: 11, fontWeight: '600', color: WHITE },

  statusRow: { width: '100%', minHeight: 28, justifyContent: 'center', borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL, paddingHorizontal: 10 },
  thinkText: { fontSize: 12, fontWeight: '700', color: GOLD },
  turnText: { fontSize: 12, color: SOFT, fontWeight: '600' },
  resultText: { fontSize: 13, fontWeight: '800', color: MINT },

  controls: { width: '100%', flexDirection: 'row', gap: 10, paddingBottom: 4 },
  playBtn: { flex: 1, minHeight: 42, backgroundColor: '#2f6b62', borderWidth: 1, borderColor: MINT, borderRadius: 999, alignItems: 'center', justifyContent: 'center' },
  stopBtn: { backgroundColor: '#7d4a56', borderColor: '#f2c5c5' },
  stepBtn: { minWidth: 92, minHeight: 42, backgroundColor: PANEL_DARK, borderWidth: 1, borderColor: LINE, borderRadius: 999, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16 },
  btnDisabled: { opacity: 0.45 },
  playText: { fontSize: 12, fontWeight: '800', color: WHITE, letterSpacing: 0.5 },
  stepText: { fontSize: 12, fontWeight: '800', color: WHITE, letterSpacing: 0.5 },
});

