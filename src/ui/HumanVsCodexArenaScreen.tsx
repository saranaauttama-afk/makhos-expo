import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { hashPosition } from '../coreClaude/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { Difficulty, GameConfig } from './types';
import { usePixelGameFx } from './usePixelGameFx';

const THINK_MS: Record<Difficulty, number> = { easy: 300, medium: 1000, hard: 2000 };
const ALGORITHM_NAME = 'Codex Hybrid';

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_ALT = '#2c1f49';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const CYAN = '#5ec5ff';
const MINT = '#77f7cf';
const PINK = '#ff7dc4';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

function posKey(p: Position) {
  return [p.side, p.p1Men, p.p1Kings, p.p2Men, p.p2Kings, p.halfmoveClock].join(':');
}

interface Props {
  config: GameConfig;
  onBack: () => void;
}

function PixelInfoCard({
  label,
  value,
  tint,
}: {
  label: string;
  value: string;
  tint: string;
}) {
  return (
    <View style={[styles.infoCard, { borderColor: tint }]}>
      <Text style={[styles.infoValue, { color: tint }]}>{value}</Text>
      <Text style={styles.infoLabel}>{label}</Text>
    </View>
  );
}

export default function HumanVsCodexArenaScreen({ config, onBack }: Props) {
  const { mode, difficulty, humanSide } = config;
  const isHvH = mode === 'vs-human';
  const aiSide = (-humanSide) as 1 | -1;
  const thinkMs = THINK_MS[difficulty];

  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel] = useState<number | null>(null);
  const [lastMove, setLastMove] = useState<{ from: number; to: number; captured: number; promote: boolean } | null>(null);
  const [shakeFrame, setShakeFrame] = useState(0);
  const [comboFrame, setComboFrame] = useState(0);
  const [comboText, setComboText] = useState('');

  const { think, thinking, lastInfo, lastPlan, cancel } = useCodexEngine();
  const { triggerFx } = usePixelGameFx();
  const pendingRef = useRef<string | null>(null);
  const endgameRef = useRef<string | null>(null);

  const myMoves = useMemo(() => generateMoves(pos), [pos]);
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );

  const canHumanMove =
    (isHvH || pos.side === humanSide) &&
    !thinking && !isDraw && !isThreefold && myMoves.length > 0;
  const shakeX = [0, -6, 5, -4, 3, -2, 0][Math.min(shakeFrame, 6)];
  const comboOpacity = [0, 0.75, 1, 1, 0.9, 0.7, 0.45, 0.2, 0][Math.min(comboFrame, 8)];
  const comboLift = [24, 18, 14, 10, 6, 2, -2, -6, -10][Math.min(comboFrame, 8)];

  function commitMove(move: Move) {
    const next = applyMove(pos, move);
    setPos(next);
    setHashHistory(prev => [...prev, hashPosition(next)]);
    setLastMove({ from: move.from, to: move.to, captured: move.captured.length, promote: move.promote });
  }

  useEffect(() => {
    if (isDraw || isThreefold) {
      const key = posKey(pos) + (isThreefold ? ':rep' : ':draw');
      if (endgameRef.current !== key) {
        endgameRef.current = key;
        Alert.alert('Game Over', isThreefold ? 'Draw by repetition.' : 'Draw by inactivity.');
      }
      return;
    }
    if (!myMoves.length) {
      const key = posKey(pos) + ':nomoves';
      if (endgameRef.current !== key) {
        endgameRef.current = key;
        const winner = isHvH
          ? (pos.side === 1 ? 'Player 2 wins.' : 'Player 1 wins.')
          : (pos.side === humanSide ? `${ALGORITHM_NAME} wins.` : 'You win.');
        Alert.alert('Game Over', winner);
      }
      return;
    }
    endgameRef.current = null;
  }, [humanSide, isDraw, isHvH, isThreefold, myMoves.length, pos]);

  useEffect(() => {
    if (isHvH) return;
    if (pos.side !== aiSide || isDraw || isThreefold || !myMoves.length) {
      pendingRef.current = null;
      return;
    }
    const key = posKey(pos);
    if (pendingRef.current === key) return;
    pendingRef.current = key;

    if (myMoves.length === 1) {
      commitMove(myMoves[0]);
      setSel(null);
      return;
    }

    const posSnapshot = pos;
    const histSnapshot = hashHistory;

    think(posSnapshot, thinkMs, histSnapshot, undefined, difficulty).then(best => {
      if (pendingRef.current !== key) return;
      const move = best ?? generateMoves(posSnapshot)[0];
      if (move) {
        const next = applyMove(posSnapshot, move);
        setPos(next);
        setHashHistory(prev => [...prev, hashPosition(next)]);
        setLastMove({ from: move.from, to: move.to, captured: move.captured.length, promote: move.promote });
      }
      setSel(null);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pos.side, isDraw, isThreefold, myMoves.length]);

  useEffect(() => {
    if (!lastMove) {
      setShakeFrame(0);
      setComboFrame(0);
      setComboText('');
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
    if (!myMoves.length || isDraw || isThreefold) triggerFx('victory');
  }, [isDraw, isThreefold, myMoves.length, triggerFx]);

  function onTapSquare(i: number) {
    if (!canHumanMove) return;
    if (sel === null) {
      if (myMoves.some(m => m.from === i)) setSel(i);
      return;
    }
    const move = myMoves.find(m => m.from === sel && m.to === i);
    if (move) {
      commitMove(move);
      setSel(null);
      return;
    }
    setSel(myMoves.some(m => m.from === i) ? i : null);
  }

  function onNewGame() {
    cancel();
    pendingRef.current = null;
    endgameRef.current = null;
    setSel(null);
    const next = initialPosition();
    setPos(next);
    setHashHistory([hashPosition(next)]);
    setLastMove(null);
  }

  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}->${m.to}`).join(' ');

  let statusText = 'Waiting';
  if (isDraw || isThreefold) statusText = 'DRAW';
  else if (!myMoves.length) statusText = 'GAME OVER';
  else if (isHvH) statusText = pos.side === 1 ? 'PLAYER 1 TURN' : 'PLAYER 2 TURN';
  else statusText = pos.side === humanSide ? 'YOUR TURN' : (thinking ? 'AI THINKING' : 'AI READY');

  const opponentLabel = isHvH ? 'Player 2 (P2)' : `${ALGORITHM_NAME} (P2)`;
  const planMode = lastPlan?.mode?.toUpperCase() ?? 'IDLE';
  const moveCount = String(myMoves.length).padStart(2, '0');
  const selectCount = sel === null ? '00' : '01';

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <View style={styles.headerTopRow}>
            <Pressable style={styles.backButton} onPress={() => { cancel(); onBack(); }}>
              <Text style={styles.backButtonText}>MENU</Text>
            </Pressable>
            <View style={styles.headerTag}>
              <Text style={styles.headerTagText}>{isHvH ? 'LOCAL' : 'HYBRID AI'}</Text>
            </View>
          </View>

          <Text style={styles.kicker}>PIXEL MATCH HUD</Text>
          <Text style={styles.title}>MAKHOS BOARD</Text>
          <Text style={styles.subtitle}>Player 1 (P1) vs {opponentLabel}</Text>
          <Text style={styles.statusText}>{statusText}</Text>
        </View>

        <View style={styles.infoRow}>
          <PixelInfoCard label="TURN" value={pos.side === 1 ? 'P1' : 'P2'} tint={GOLD} />
          <PixelInfoCard label="STATE" value={thinking ? 'THINK' : 'READY'} tint={thinking ? PINK : MINT} />
          <PixelInfoCard label="MODE" value={planMode} tint={CYAN} />
        </View>

        <View style={styles.infoRow}>
          <PixelInfoCard label="MOVES" value={moveCount} tint={MINT} />
          <PixelInfoCard label="PICK" value={selectCount} tint={PINK} />
          <PixelInfoCard label="CLOCK" value={String(pos.halfmoveClock).padStart(2, '0')} tint={GOLD} />
        </View>

        <View style={styles.boardPanel}>
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

          <View style={[styles.boardFrame, { transform: [{ translateX: shakeX }] }]}>
            <Board
              pos={pos}
              onTapSquare={onTapSquare}
              fromSquares={sel !== null ? [sel] : []}
              selectedFrom={sel}
              destSquares={sel !== null ? myMoves.filter(m => m.from === sel).map(m => ({ to: m.to, caps: m.captured.length })) : []}
              lastMove={lastMove}
            />
          </View>
        </View>

        <View style={styles.telemetryPanel}>
          <Text style={styles.telemetryTitle}>ENGINE TELEMETRY</Text>
          {!isHvH && lastInfo ? (
            <>
              <Text style={styles.telemetryLine}>depth {lastInfo.depth} | score {lastInfo.score} | nodes {lastInfo.nodes}</Text>
              {lastPlan ? <Text style={styles.telemetryLine}>mode {lastPlan.mode} | {lastPlan.reason}</Text> : null}
              {pvText ? <Text style={styles.telemetryLine}>pv {pvText}</Text> : null}
            </>
          ) : (
            <Text style={styles.telemetryLine}>Telemetry appears after the first AI search completes.</Text>
          )}
        </View>

        <View style={styles.actionRow}>
          <Pressable style={styles.primaryButton} onPress={onNewGame}>
            <Text style={styles.primaryButtonText}>NEW GAME</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => { cancel(); onBack(); }}>
            <Text style={styles.secondaryButtonText}>EXIT BOARD</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: BG,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 24,
    gap: 14,
    alignItems: 'center',
  },
  headerPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    gap: 8,
  },
  headerTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  backButton: {
    minWidth: 88,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    alignItems: 'center',
  },
  backButtonText: {
    color: WHITE,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
  },
  headerTag: {
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderWidth: 2,
    borderColor: CYAN,
    backgroundColor: PANEL_DARK,
  },
  headerTagText: {
    color: CYAN,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.9,
  },
  kicker: {
    color: GOLD,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.3,
  },
  title: {
    color: WHITE,
    fontSize: 24,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  subtitle: {
    color: SOFT,
    fontSize: 12,
    lineHeight: 18,
  },
  statusText: {
    color: MINT,
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.7,
  },
  infoRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 10,
  },
  infoCard: {
    flex: 1,
    minHeight: 82,
    backgroundColor: PANEL_ALT,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  infoValue: {
    fontSize: 20,
    fontWeight: '900',
    letterSpacing: 1,
  },
  infoLabel: {
    color: SOFT,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  boardPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    alignItems: 'center',
    overflow: 'hidden',
  },
  boardFrame: {
    padding: 10,
    backgroundColor: PANEL_DARK,
    borderWidth: 3,
    borderColor: GOLD,
  },
  comboBadge: {
    position: 'absolute',
    top: 8,
    zIndex: 4,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderWidth: 2,
    borderColor: GOLD,
    backgroundColor: '#ff4fa3',
  },
  comboBadgeText: {
    color: WHITE,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  telemetryPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    gap: 6,
  },
  telemetryTitle: {
    color: PINK,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1,
  },
  telemetryLine: {
    color: WHITE,
    fontSize: 12,
    lineHeight: 18,
  },
  actionRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 10,
  },
  primaryButton: {
    flex: 1,
    minHeight: 54,
    backgroundColor: PANEL_ALT,
    borderWidth: 3,
    borderColor: GOLD,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: {
    color: WHITE,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 54,
    backgroundColor: PANEL,
    borderWidth: 2,
    borderColor: LINE,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: SOFT,
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
});
