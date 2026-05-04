import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { AppLanguage } from '../../App';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { hashPosition } from '../coreClaude/search/zobrist';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { STRICT_LEVEL_POLICY } from '../coreClaude/search/levelPolicy';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { usePixelGameFx } from './usePixelGameFx';
import { Difficulty } from './types';

type ArenaLevel = Difficulty;
const ARENA_LEVELS: { id: ArenaLevel; label: string }[] = [
  { id: 'easy', label: 'Level 1' },
  { id: 'normal', label: 'Level 2' },
  { id: 'hard', label: 'Level 3' },
  { id: 'expert', label: 'Level 4' },
  { id: 'master', label: 'Level 5' },
];
const THINK_MS: Record<ArenaLevel, number> = {
  easy: 0,
  normal: 0,
  hard: 0,
  expert: 0,
  master: 4500,
};
const LEVEL_LABEL: Record<ArenaLevel, string> = {
  easy: 'Level 1',
  normal: 'Level 2',
  hard: 'Level 3',
  expert: 'Level 4',
  master: 'Level 5',
};
const LEVEL_MODE_TAG: Record<ArenaLevel, string> = {
  easy: 'STRICT',
  normal: 'STRICT',
  hard: 'STRICT',
  expert: 'STRICT',
  master: 'GUIDED',
};

interface LogEntry {
  ply: number;
  side: 1 | -1;
  agent: ArenaLevel;
  move: Move;
}

function fmtMove(m: Move): string {
  const cap = m.captured.length > 0 ? `x${m.captured.length}` : '';
  const promo = m.promote ? 'K' : '';
  return `${m.from + 1}->${m.to + 1}${cap}${promo}`;
}

function LevelBtn({ level, active, onPress }: { level: ArenaLevel; active: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.agentBtn, active && styles.agentBtnActive]}>
      <Text style={[styles.agentText, active && styles.agentTextActive]}>{LEVEL_LABEL[level]}</Text>
    </Pressable>
  );
}

interface Props {
  language: AppLanguage;
  onBack: () => void;
}

const COPY = {
  th: {
    back: 'BACK',
    title: 'Arena Lab',
    reset: 'RESET',
    moveLog: 'Move Log',
    draw: 'Draw',
    wins: 'wins',
    stop: 'STOP',
    autoPlay: 'AUTO PLAY',
    step: 'STEP',
    thinking: 'คิด',
    turn: 'ตา',
    ply: 'ply',
    p1: 'P1',
    p2: 'P2',
  },
  en: {
    back: 'BACK',
    title: 'Arena Lab',
    reset: 'RESET',
    moveLog: 'Move Log',
    draw: 'Draw',
    wins: 'wins',
    stop: 'STOP',
    autoPlay: 'AUTO PLAY',
    step: 'STEP',
    thinking: 'thinking',
    turn: 'Turn',
    ply: 'ply',
    p1: 'P1',
    p2: 'P2',
  },
} as const;

export default function ArenaScreen({ language, onBack }: Props) {
  const t = COPY[language];
  const [p1Level, setP1Level] = useState<ArenaLevel>('normal');
  const [p2Level, setP2Level] = useState<ArenaLevel>('master');

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

  const { think, thinkStrict, thinking } = useCodexEngine();
  const { triggerFx } = usePixelGameFx();

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
  const currentLevel = pos.side === 1 ? p1Level : p2Level;
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

    const agent = curPos.side === 1 ? p1Level : p2Level;

    steppingRef.current = true;

    const move = agent === 'master'
      ? await think(curPos, THINK_MS[agent], curHistory, undefined, 'master')
      : await thinkStrict(curPos, THINK_MS[agent], curHistory, STRICT_LEVEL_POLICY[agent].baseDepth, undefined, agent);
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
      if (!nextMoves.length) setGameResult(`${label} (${LEVEL_LABEL[agent]}) ${t.wins}`);
      else setGameResult(t.draw);
      setAutoPlay(false);
    }
  }, [p1Level, p2Level, t.draw, t.wins, think, thinkStrict]);

  useEffect(() => {
    if (!autoPlay || gameOver || thinking || steppingRef.current) return;
    const delay = setTimeout(() => {
      void step(pos, hashHistory);
    }, 300);

    return () => clearTimeout(delay);
  }, [autoPlay, pos, thinking, gameOver, step, hashHistory]);

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
  const limitLabel = THINK_MS[currentLevel] <= 0
    ? '∞'
    : `${(THINK_MS[currentLevel] / 1000).toFixed(0)}s`;

  return (
    <SafeAreaView style={styles.container}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />

      <View style={styles.header}>
        <Pressable onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>{t.back}</Text>
        </Pressable>
        <Text style={styles.title}>{t.title}</Text>
        <Pressable onPress={reset} style={styles.resetBtn}>
          <Text style={styles.resetText}>{t.reset}</Text>
        </Pressable>
      </View>

      <View style={styles.agentRow}>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P1</Text>
          {ARENA_LEVELS.map(level => (
            <LevelBtn key={level.id} level={level.id} active={p1Level === level.id} onPress={() => setP1Level(level.id)} />
          ))}
        </View>
        <View style={styles.agentCol}>
          <Text style={styles.sideLabel}>P2</Text>
          {ARENA_LEVELS.map(level => (
            <LevelBtn key={level.id} level={level.id} active={p2Level === level.id} onPress={() => setP2Level(level.id)} />
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
          <Text style={styles.logTitle}>{t.moveLog}</Text>
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
            {sideLabel} ({LEVEL_LABEL[currentLevel]} {LEVEL_MODE_TAG[currentLevel]}) {t.thinking} {thinkSeconds}s / {limitLabel}
          </Text>
        ) : (
          <Text style={styles.turnText}>
            {t.turn} {sideLabel} ({LEVEL_LABEL[currentLevel]} {LEVEL_MODE_TAG[currentLevel]}) - {t.ply} {log.length + 1}
          </Text>
        )}
      </View>

      <View style={styles.controls}>
        <Pressable
          onPress={toggleAutoPlay}
          disabled={gameOver}
          style={[styles.playBtn, autoPlay && styles.stopBtn, gameOver && styles.btnDisabled]}
        >
          <Text style={styles.playText}>{autoPlay ? t.stop : t.autoPlay}</Text>
        </Pressable>

        {!autoPlay && !gameOver && (
          <Pressable
            onPress={() => { void step(pos, hashHistory); }}
            disabled={thinking || steppingRef.current}
            style={[styles.stepBtn, (thinking || steppingRef.current) && styles.btnDisabled]}
          >
            <Text style={styles.stepText}>{t.step}</Text>
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
  backText: { fontSize: 11, color: WHITE, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.6 },
  resetBtn: { minHeight: 30, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK, justifyContent: 'center', paddingHorizontal: 10 },
  resetText: { fontSize: 11, color: SOFT, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.6 },
  title: { fontSize: 17, fontFamily: 'Kanit_800ExtraBold', color: WHITE, letterSpacing: 0.6 },

  agentRow: { flexDirection: 'row', gap: 10, width: '100%' },
  agentCol: { flex: 1, gap: 5, borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL, padding: 8 },
  sideLabel: { fontSize: 10, fontFamily: 'Kanit_800ExtraBold', color: GOLD, textTransform: 'uppercase', letterSpacing: 0.8 },
  agentBtn: { minHeight: 30, borderRadius: 999, borderWidth: 1, borderColor: LINE, backgroundColor: PANEL_DARK, alignItems: 'center', justifyContent: 'center' },
  agentBtnActive: { borderColor: MINT, backgroundColor: '#2f6b62' },
  agentText: { fontSize: 11, fontFamily: 'Kanit_700Bold', color: SOFT, letterSpacing: 0.4 },
  agentTextActive: { color: WHITE },

  mainRow: { flexDirection: 'row', gap: 10, width: '100%', flex: 1 },
  boardStage: { alignItems: 'center', justifyContent: 'center', overflow: 'hidden', padding: 8, borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL },
  comboBadge: { position: 'absolute', top: 8, zIndex: 4, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 10, backgroundColor: '#2f6b62', borderWidth: 1, borderColor: GOLD },
  comboBadgeText: { fontSize: 11, fontFamily: 'Kanit_800ExtraBold', color: GOLD, letterSpacing: 0.7 },

  logPanel: { flex: 1, borderRadius: 12, borderWidth: 1, borderColor: LINE, backgroundColor: PANEL, padding: 8 },
  logTitle: { fontSize: 11, fontFamily: 'Kanit_800ExtraBold', color: GOLD, textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 4 },
  logScroll: { flex: 1 },
  logEntry: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingVertical: 2 },
  logPly: { fontSize: 10, fontFamily: 'Kanit_700Bold', width: 20 },
  logP1: { color: P1_CLR },
  logP2: { color: P2_CLR },
  logAgent: { fontSize: 10, fontFamily: 'Kanit_700Bold', color: SOFT, width: 18 },
  logMove: { fontSize: 11, fontFamily: 'Kanit_500Medium', color: WHITE },

  statusRow: { width: '100%', minHeight: 28, justifyContent: 'center', borderWidth: 1, borderColor: LINE, borderRadius: 12, backgroundColor: PANEL, paddingHorizontal: 10 },
  thinkText: { fontSize: 12, fontFamily: 'Kanit_700Bold', color: GOLD },
  turnText: { fontSize: 12, color: SOFT, fontFamily: 'Kanit_500Medium' },
  resultText: { fontSize: 13, fontFamily: 'Kanit_800ExtraBold', color: MINT },

  controls: { width: '100%', flexDirection: 'row', gap: 10, paddingBottom: 4 },
  playBtn: { flex: 1, minHeight: 42, backgroundColor: '#2f6b62', borderWidth: 1, borderColor: MINT, borderRadius: 999, alignItems: 'center', justifyContent: 'center' },
  stopBtn: { backgroundColor: '#7d4a56', borderColor: '#f2c5c5' },
  stepBtn: { minWidth: 92, minHeight: 42, backgroundColor: PANEL_DARK, borderWidth: 1, borderColor: LINE, borderRadius: 999, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16 },
  btnDisabled: { opacity: 0.45 },
  playText: { fontSize: 12, fontFamily: 'Kanit_800ExtraBold', color: WHITE, letterSpacing: 0.5 },
  stepText: { fontSize: 12, fontFamily: 'Kanit_800ExtraBold', color: WHITE, letterSpacing: 0.5 },
});

