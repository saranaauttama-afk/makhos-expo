import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { bitCount } from '../coreClaude/bitboards';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { hashPosition } from '../coreClaude/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { Difficulty, GameConfig } from './types';
import { usePixelGameFx } from './usePixelGameFx';

const THINK_MS: Record<Difficulty, number> = { easy: 300, medium: 1000, hard: 2000 };
const MOVE_ANIM_GUARD_MS = 560;
const INITIAL_PIECES_PER_SIDE = 8;

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

function formatLastMoveCompact(move: { from: number; to: number; captured: number; promote: boolean } | null) {
  if (!move) return '-';
  const cap = move.captured > 0 ? ` x${move.captured}` : '';
  const promo = move.promote ? ' K' : '';
  return `${move.from}->${move.to}${cap}${promo}`;
}

function formatTurnSeconds(ms: number) {
  return `${(ms / 1000).toFixed(1)}s`;
}

interface Props {
  config: GameConfig;
  onBack: () => void;
}

type MoveHint = { from: number; to: number; captured: number; promote: boolean };
type AvatarKind = 'human' | 'human-sad' | 'bot-easy' | 'bot-medium' | 'bot-hard';

const AVATAR_PATTERNS_16: Record<AvatarKind, string[]> = {
  human: [
    '..hhhhhhhhhhhh..',
    '.hhhhhhhhhhhhhh.',
    '.hhsssssssssshh.',
    'hhsssssssssssshh',
    'hhsssse..esssshh',
    'hhssss....sssshh',
    'hhsssse..esssshh',
    'hhsssssssssssshh',
    'hhssssmmmmsssshh',
    'hhssssmmmmsssshh',
    'hhsssssssssssshh',
    '.hhsssssssssshh.',
    '.hhssmmmmmmsshh.',
    '..hhsssssssshh..',
    '...hhhhhhhhhh...',
    '................',
  ],
  'human-sad': [
    '..hhhhhhhhhhhh..',
    '.hhhhhhhhhhhhhh.',
    '.hhsssssssssshh.',
    'hhsssssssssssshh',
    'hhsssse..esssshh',
    'hhssss....sssshh',
    'hhsssse..esssshh',
    'hhsssssssssssshh',
    'hhssssmmmmsssshh',
    'hhssssmmmmsssshh',
    'hhsssssssssssshh',
    '.hhsssssssssshh.',
    '.hhssm....msshh.',
    '..hhssmmmmsshh..',
    '...hhhhhhhhhh...',
    '................',
  ],
  'bot-easy': [
    '.......bb.......',
    '.......bb.......',
    '..bbbbbbbbbbbb..',
    '.bbccccccccccbb.',
    'bbccgg....ggccbb',
    'bbccg......gccbb',
    'bbcc........ccbb',
    'bbcc..bbbb..ccbb',
    'bbcc........ccbb',
    'bbccgg....ggccbb',
    'bbcc........ccbb',
    '.bbccccccccccbb.',
    '..bbbbbbbbbbbb..',
    '......bbbb......',
    '......b..b......',
    '................',
  ],
  'bot-medium': [
    '.......bb.......',
    '.......bb.......',
    '..bbbbbbbbbbbb..',
    '.bbccccccccccbb.',
    'bbccyy....yyccbb',
    'bbccy......yccbb',
    'bbcc........ccbb',
    'bbcc..byyb..ccbb',
    'bbcc........ccbb',
    'bbcc..yyyy..ccbb',
    'bbcc........ccbb',
    '.bbccccccccccbb.',
    '..bbbbbbbbbbbb..',
    '.....bbbbbb.....',
    '......b..b......',
    '................',
  ],
  'bot-hard': [
    '......rr..rr....',
    '.......rrrr.....',
    '..rrrrrrrrrrrr..',
    '.rrccccccccccrr.',
    'rrccxx....xxccrr',
    'rrccx......xccrr',
    'rrcc........ccrr',
    'rrcc..rrrr..ccrr',
    'rrcc........ccrr',
    'rrcc..xxxx..ccrr',
    'rrcc........ccrr',
    '.rrccccccccccrr.',
    '..rrrrrrrrrrrr..',
    '.....rrrrrr.....',
    '......r..r......',
    '................',
  ],
};

const AVATAR_COLORS: Record<string, string> = {
  h: '#5f3d1a',
  s: '#ffd7a1',
  e: '#2b1b12',
  m: '#d88d6a',
  b: '#9ec7ff',
  c: '#2e3f64',
  g: '#77f7cf',
  y: '#ffe17d',
  x: '#ff7dc4',
  r: '#ff5f85',
};

function avatarKindByDifficulty(difficulty: Difficulty): AvatarKind {
  if (difficulty === 'hard') return 'bot-hard';
  if (difficulty === 'medium') return 'bot-medium';
  return 'bot-easy';
}

function PixelAvatar({
  kind,
  tint,
  active,
}: {
  kind: AvatarKind;
  tint: string;
  active: boolean;
}) {
  const pattern = AVATAR_PATTERNS_16[kind];
  return (
    <View style={[styles.avatarShell, { borderColor: active ? tint : LINE }]}>
      <View style={[styles.avatarGrid, { opacity: active ? 1 : 0.78 }]}>
        {pattern.flatMap((row, r) =>
          row.split('').map((cell, c) => (
            <View
              key={`${kind}-${r}-${c}`}
              style={[
                styles.avatarPixel,
                { backgroundColor: cell === '.' ? 'transparent' : (AVATAR_COLORS[cell] ?? 'transparent') },
              ]}
            />
          )),
        )}
      </View>
    </View>
  );
}

function aiLevelTag(difficulty: Difficulty) {
  if (difficulty === 'hard') return 'AI-H';
  if (difficulty === 'medium') return 'AI-M';
  return 'AI-E';
}

function PieceMeter({
  pieceCount,
  active,
  tint,
}: {
  pieceCount: number;
  active: boolean;
  tint: string;
}) {
  const ratio = Math.max(0, Math.min(1, pieceCount / INITIAL_PIECES_PER_SIDE));
  const fillFlex = ratio <= 0 ? 0 : ratio;
  const restFlex = 1 - ratio;
  return (
    <View style={styles.pieceMeterTrack}>
      <View
        style={[
          styles.pieceMeterFill,
          {
            flex: fillFlex,
            backgroundColor: active ? tint : '#8478a8',
          },
        ]}
      />
      <View style={[styles.pieceMeterRest, { flex: restFlex }]} />
    </View>
  );
}

function TurnSeatChip({
  lane,
  avatarKind,
  role,
  pieceCount,
  captured,
  turnTimer,
  lastMoveText,
  active,
  tint,
  showTelemetryToggle = false,
  telemetryEnabled = false,
  onToggleTelemetry,
}: {
  lane: string;
  avatarKind: AvatarKind;
  role: string;
  pieceCount: number;
  captured: number;
  turnTimer: string;
  lastMoveText?: string | null;
  active: boolean;
  tint: string;
  showTelemetryToggle?: boolean;
  telemetryEnabled?: boolean;
  onToggleTelemetry?: () => void;
}) {
  return (
    <View
      style={[
        styles.turnSeatChip,
        {
          borderColor: active ? tint : LINE,
          backgroundColor: active ? PANEL_ALT : PANEL_DARK,
          opacity: active ? 1 : 0.68,
          transform: [{ scale: active ? 1 : 0.98 }],
        },
      ]}
    >
      <View style={styles.turnSeatInner}>
        <PixelAvatar kind={avatarKind} tint={tint} active={active} />
        <View style={styles.turnSeatCopy}>
          <Text style={[styles.turnSeatText, { color: active ? WHITE : SOFT }]}>
            {lane}
          </Text>
          <View style={styles.seatBoxRow}>
            <View style={[styles.seatInfoBox, { borderColor: active ? tint : LINE }]}>
              <Text style={[styles.turnSeatRole, { color: active ? tint : SOFT }]}>
                {role}
              </Text>
            </View>
            <View style={[styles.seatInfoBox, { borderColor: active ? tint : LINE }]}>
              <Text style={styles.seatInfoText}>CAP x{captured}</Text>
            </View>
            <View style={[styles.seatInfoBox, { borderColor: active ? tint : LINE }]}>
              <Text style={styles.seatInfoText}>T {turnTimer}</Text>
            </View>
            <View style={[styles.seatInfoBox, styles.seatInfoBoxWide, { borderColor: active ? GOLD : LINE }]}>
              <Text style={styles.seatInfoText}>
                {lastMoveText ? `LAST ${lastMoveText}` : 'LAST -'}
              </Text>
            </View>
          </View>
          <View style={styles.pieceMeterRow}>
            <Text style={styles.pieceMeterLabel}>UNITS {pieceCount}/{INITIAL_PIECES_PER_SIDE}</Text>
            <PieceMeter pieceCount={pieceCount} active={active} tint={tint} />
          </View>
        </View>
        {showTelemetryToggle && onToggleTelemetry ? (
          <Pressable
            onPress={onToggleTelemetry}
            style={[
              styles.telemetryToggleBtn,
              telemetryEnabled ? styles.telemetryToggleBtnOn : styles.telemetryToggleBtnOff,
            ]}
          >
            <Text style={styles.telemetryToggleText}>
              {telemetryEnabled ? 'TEL ON' : 'TEL OFF'}
            </Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

export default function HumanVsCodexArenaScreen({ config, onBack }: Props) {
  const { mode, difficulty, humanSide } = config;
  const isHvH = mode === 'vs-human';
  const aiSide = (-humanSide) as 1 | -1;
  const thinkMs = THINK_MS[difficulty];

  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [posHistory, setPosHistory] = useState<Position[]>(() => [initialPosition()]);
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel] = useState<number | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveHint[]>([]);
  const [lastMove, setLastMove] = useState<MoveHint | null>(null);
  const [shakeFrame, setShakeFrame] = useState(0);
  const [comboFrame, setComboFrame] = useState(0);
  const [comboText, setComboText] = useState('');
  const [showTelemetry, setShowTelemetry] = useState(false);
  const [premiumUndo, setPremiumUndo] = useState(false);
  const [turnElapsedMs, setTurnElapsedMs] = useState(0);
  const [isAnimLocked, setIsAnimLocked] = useState(false);
  const turnStartRef = useRef<number>(Date.now());

  const { think, thinking, lastInfo, lastPlan, cancel } = useCodexEngine();
  const { triggerFx } = usePixelGameFx();
  const pendingRef = useRef<string | null>(null);
  const aiCommitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animUnlockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thinkStartTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animLockUntilRef = useRef<number>(0);

  const myMoves = useMemo(() => generateMoves(pos), [pos]);
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const gameResult = useMemo(() => {
    if (isDraw || isThreefold) return { label: 'DRAW', tone: 'draw' as const };
    if (myMoves.length > 0) return null;

    const winnerSide = (pos.side === 1 ? -1 : 1) as 1 | -1;
    if (isHvH) return { label: winnerSide === 1 ? 'P1 WIN' : 'P2 WIN', tone: 'win' as const };
    return winnerSide === humanSide
      ? { label: 'YOU WIN', tone: 'win' as const, avatarKind: 'human' as AvatarKind }
      : { label: 'YOU LOSE', tone: 'loss' as const, avatarKind: 'human-sad' as AvatarKind };
  }, [humanSide, isDraw, isHvH, isThreefold, myMoves.length, pos.side]);

  const canHumanMove =
    (isHvH || pos.side === humanSide) &&
    !thinking && !isAnimLocked && !isDraw && !isThreefold && myMoves.length > 0;
  const shakeX = [0, -6, 5, -4, 3, -2, 0][Math.min(shakeFrame, 6)];
  const comboOpacity = [0, 0.75, 1, 1, 0.9, 0.7, 0.45, 0.2, 0][Math.min(comboFrame, 8)];
  const comboLift = [24, 18, 14, 10, 6, 2, -2, -6, -10][Math.min(comboFrame, 8)];

  function commitMove(basePos: Position, move: Move) {
    const next = applyMove(basePos, move);
    const hint: MoveHint = { from: move.from, to: move.to, captured: move.captured.length, promote: move.promote };
    lockMoveAnimationWindow();
    setPos(next);
    setPosHistory(prev => [...prev, next]);
    setHashHistory(prev => [...prev, hashPosition(next)]);
    setMoveHistory(prev => [...prev, hint]);
    setLastMove(hint);
  }

  function clearPendingAICommit() {
    if (aiCommitTimerRef.current) {
      clearTimeout(aiCommitTimerRef.current);
      aiCommitTimerRef.current = null;
    }
  }

  function clearAnimUnlockTimer() {
    if (animUnlockTimerRef.current) {
      clearTimeout(animUnlockTimerRef.current);
      animUnlockTimerRef.current = null;
    }
  }

  function clearThinkStartTimer() {
    if (thinkStartTimerRef.current) {
      clearTimeout(thinkStartTimerRef.current);
      thinkStartTimerRef.current = null;
    }
  }

  function lockMoveAnimationWindow() {
    animLockUntilRef.current = Date.now() + MOVE_ANIM_GUARD_MS;
    setIsAnimLocked(true);
    clearAnimUnlockTimer();
    animUnlockTimerRef.current = setTimeout(() => {
      animUnlockTimerRef.current = null;
      if (Date.now() >= animLockUntilRef.current) setIsAnimLocked(false);
    }, MOVE_ANIM_GUARD_MS);
  }

  function scheduleAICommit(basePos: Position, move: Move, turnKey: string) {
    clearPendingAICommit();
    setSel(null);

    const delayMs = Math.max(0, animLockUntilRef.current - Date.now());
    if (delayMs <= 0) {
      if (pendingRef.current === turnKey) commitMove(basePos, move);
      return;
    }

    aiCommitTimerRef.current = setTimeout(() => {
      aiCommitTimerRef.current = null;
      if (pendingRef.current === turnKey) commitMove(basePos, move);
    }, delayMs);
  }

  useEffect(() => {
    if (isHvH) {
      clearThinkStartTimer();
      return;
    }
    if (pos.side !== aiSide || isDraw || isThreefold || !myMoves.length) {
      pendingRef.current = null;
      clearPendingAICommit();
      clearThinkStartTimer();
      return;
    }
    const key = posKey(pos);
    if (pendingRef.current === key) return;
    pendingRef.current = key;

    if (myMoves.length === 1) {
      scheduleAICommit(pos, myMoves[0], key);
      return;
    }

    const posSnapshot = pos;
    const histSnapshot = hashHistory;
    const runThink = () => {
      thinkStartTimerRef.current = null;
      if (pendingRef.current !== key) return;
      think(posSnapshot, thinkMs, histSnapshot, undefined, difficulty).then(best => {
        if (pendingRef.current !== key) return;
        const move = best ?? generateMoves(posSnapshot)[0];
        if (move) {
          scheduleAICommit(posSnapshot, move, key);
        }
      });
    };

    clearThinkStartTimer();
    const startDelayMs = Math.max(0, animLockUntilRef.current - Date.now());
    if (startDelayMs > 0) {
      thinkStartTimerRef.current = setTimeout(runThink, startDelayMs);
    } else {
      runThink();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pos.side, isDraw, isThreefold, myMoves.length]);

  useEffect(() => () => {
    clearPendingAICommit();
    clearAnimUnlockTimer();
    clearThinkStartTimer();
  }, []);

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

  useEffect(() => {
    turnStartRef.current = Date.now();
    setTurnElapsedMs(0);
  }, [pos.side]);

  useEffect(() => {
    if (!myMoves.length || isDraw || isThreefold) return;
    const id = setInterval(() => {
      if (!isAnimLocked) setTurnElapsedMs(Date.now() - turnStartRef.current);
    }, 220);
    return () => clearInterval(id);
  }, [isAnimLocked, isDraw, isThreefold, myMoves.length, pos.side]);

  function onTapSquare(i: number) {
    if (!canHumanMove) return;
    if (sel === null) {
      if (myMoves.some(m => m.from === i)) setSel(i);
      return;
    }
    const move = myMoves.find(m => m.from === sel && m.to === i);
    if (move) {
      commitMove(pos, move);
      setSel(null);
      return;
    }
    setSel(myMoves.some(m => m.from === i) ? i : null);
  }

  function runUndo() {
    const rewindPlies = isHvH ? 1 : 2;
    const maxUndo = posHistory.length - 1;
    const actual = Math.min(rewindPlies, maxUndo);
    if (actual <= 0) return;

    cancel();
    pendingRef.current = null;
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);

    const nextPosHistory = posHistory.slice(0, posHistory.length - actual);
    const restored = nextPosHistory[nextPosHistory.length - 1];
    const nextHashHistory = hashHistory.slice(0, Math.max(1, hashHistory.length - actual));
    const nextMoveHistory = moveHistory.slice(0, Math.max(0, moveHistory.length - actual));

    setPos(restored);
    setPosHistory(nextPosHistory);
    setHashHistory(nextHashHistory);
    setMoveHistory(nextMoveHistory);
    setLastMove(nextMoveHistory.length ? nextMoveHistory[nextMoveHistory.length - 1] : null);
    setSel(null);
  }

  function onPressUndo() {
    if (thinking) return;
    if (posHistory.length <= 1) return;

    const undoLockedByMode = !isHvH && difficulty === 'hard';
    if (undoLockedByMode) {
      Alert.alert('Undo Locked', 'Hard mode locks undo to keep the challenge fair.');
      return;
    }

    if (premiumUndo) {
      runUndo();
      return;
    }

    Alert.alert(
      'Unlock Undo',
      'Watch an ad to get 1 undo or unlock premium for unlimited undo.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Go Premium',
          onPress: () => {
            setPremiumUndo(true);
            runUndo();
          },
        },
        {
          text: 'Watch Ad',
          onPress: () => {
            runUndo();
          },
        },
      ],
    );
  }

  function onNewGame() {
    cancel();
    pendingRef.current = null;
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);
    setSel(null);
    const next = initialPosition();
    setPos(next);
    setPosHistory([next]);
    setHashHistory([hashPosition(next)]);
    setMoveHistory([]);
    setLastMove(null);
  }

  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}->${m.to}`).join(' ');

  const p1IsHuman = isHvH || humanSide === 1;
  const p2IsHuman = isHvH || humanSide === -1;
  const p1Role = p1IsHuman ? 'HUMAN' : aiLevelTag(difficulty);
  const p2Role = p2IsHuman ? 'HUMAN' : aiLevelTag(difficulty);
  const p1AvatarKind: AvatarKind = p1IsHuman ? 'human' : avatarKindByDifficulty(difficulty);
  const p2AvatarKind: AvatarKind = p2IsHuman ? 'human' : avatarKindByDifficulty(difficulty);
  const p1IsBot = !p1IsHuman;
  const p2IsBot = !p2IsHuman;
  const p1PieceCount = bitCount((pos.p1Men | pos.p1Kings) >>> 0);
  const p2PieceCount = bitCount((pos.p2Men | pos.p2Kings) >>> 0);
  const p1Captured = Math.max(0, INITIAL_PIECES_PER_SIDE - p2PieceCount);
  const p2Captured = Math.max(0, INITIAL_PIECES_PER_SIDE - p1PieceCount);
  const lastMovedSide: 1 | -1 = pos.side === 1 ? -1 : 1;
  const lastMoveText = formatLastMoveCompact(lastMove);
  const p1Last = lastMovedSide === 1 && lastMove ? lastMoveText : null;
  const p2Last = lastMovedSide === -1 && lastMove ? lastMoveText : null;
  const liveTurn = formatTurnSeconds(turnElapsedMs);
  const p1Turn = pos.side === 1 ? liveTurn : '--';
  const p2Turn = pos.side === -1 ? liveTurn : '--';
  const undoLockedByMode = !isHvH && difficulty === 'hard';
  const canUndoNow = !thinking && posHistory.length > 1;
  const undoBtnText = undoLockedByMode ? '🔒 UNDO' : 'UNDO';
  const modeLabel = isHvH ? 'LOCAL DUEL' : `YOU VS ${aiLevelTag(difficulty)}`;
  const stateLabel = gameResult ? gameResult.label : pos.side === 1 ? 'P1 TURN' : 'P2 TURN';
  const recentMovesText = moveHistory.length
    ? moveHistory.slice(-4).map(m => formatLastMoveCompact(m)).join(' | ')
    : '-';
  const forcedFrom = useMemo(() => {
    if (!canHumanMove || sel !== null) return null;
    const fromSet = new Set(myMoves.map(m => m.from));
    if (fromSet.size !== 1) return null;
    const [onlyFrom] = Array.from(fromSet);
    return onlyFrom ?? null;
  }, [canHumanMove, myMoves, sel]);
  const suggestedFrom = sel === null ? forcedFrom : null;

  function onExitBoard() {
    cancel();
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);
    onBack();
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.edgeSpacer} />

        <View style={styles.matchHeaderPanel}>
          <View style={styles.matchHeaderCopy}>
            <Text style={styles.matchHeaderMode}>{modeLabel}</Text>
            <Text style={styles.matchHeaderState}>{stateLabel}</Text>
          </View>
          <Pressable style={styles.matchHeaderExitBtn} onPress={onExitBoard}>
            <Text style={styles.matchHeaderExitText}>EXIT</Text>
          </Pressable>
        </View>

        <TurnSeatChip
          lane="P2"
          avatarKind={p2AvatarKind}
          role={p2Role}
          pieceCount={p2PieceCount}
          captured={p2Captured}
          turnTimer={p2Turn}
          lastMoveText={p2Last}
          active={pos.side === -1}
          tint={PINK}
          showTelemetryToggle={p2IsBot}
          telemetryEnabled={showTelemetry}
          onToggleTelemetry={() => setShowTelemetry(v => !v)}
        />

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
              fromSquares={suggestedFrom !== null ? [suggestedFrom] : []}
              selectedFrom={sel}
              destSquares={sel !== null ? myMoves.filter(m => m.from === sel).map(m => ({ to: m.to, caps: m.captured.length })) : []}
              lastMove={lastMove}
            />
            {gameResult ? (
              <View style={styles.resultOverlay}>
                <View
                  style={[
                    styles.resultOverlayBox,
                    gameResult.tone === 'win'
                      ? styles.resultOverlayWin
                      : gameResult.tone === 'loss'
                        ? styles.resultOverlayLoss
                        : styles.resultOverlayDraw,
                  ]}
                >
                  <Text style={styles.resultOverlayTitle}>{gameResult.label}</Text>
                  {'avatarKind' in gameResult && gameResult.avatarKind ? (
                    <View style={styles.resultOverlayAvatarRow}>
                      <View style={styles.resultOverlayAvatarWrap}>
                        <PixelAvatar
                          kind={gameResult.avatarKind}
                          tint={gameResult.tone === 'win' ? MINT : PINK}
                          active
                        />
                        {gameResult.tone === 'loss' ? (
                          <View pointerEvents="none" style={styles.lossCrossOverlay}>
                            <View style={[styles.lossCrossLine, styles.lossCrossLineA]} />
                            <View style={[styles.lossCrossLine, styles.lossCrossLineB]} />
                          </View>
                        ) : null}
                      </View>
                    </View>
                  ) : null}
                </View>
              </View>
            ) : null}
          </View>

        </View>

        <View style={styles.recentMovesPanel}>
          <Text style={styles.recentMovesTitle}>RECENT MOVES</Text>
          <Text style={styles.recentMovesText}>{recentMovesText}</Text>
        </View>

        <TurnSeatChip
          lane="P1"
          avatarKind={p1AvatarKind}
          role={p1Role}
          pieceCount={p1PieceCount}
          captured={p1Captured}
          turnTimer={p1Turn}
          lastMoveText={p1Last}
          active={pos.side === 1}
          tint={CYAN}
          showTelemetryToggle={p1IsBot}
          telemetryEnabled={showTelemetry}
          onToggleTelemetry={() => setShowTelemetry(v => !v)}
        />

        {showTelemetry ? (
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
        ) : null}

        <View style={styles.actionRow}>
          <Pressable
            style={[
              styles.secondaryButton,
              undoLockedByMode && styles.undoBtnLockedLook,
              !canUndoNow && styles.btnDisabled,
            ]}
            onPress={onPressUndo}
            disabled={!canUndoNow}
          >
            <Text style={styles.secondaryButtonText}>{undoBtnText}</Text>
          </Pressable>
          <Pressable style={styles.primaryButton} onPress={onNewGame}>
            <Text style={styles.primaryButtonText}>NEW GAME</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={onExitBoard}>
            <Text style={styles.secondaryButtonText}>EXIT BOARD</Text>
          </Pressable>
        </View>

        <View style={styles.edgeSpacer} />
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
    flexGrow: 1,
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 16,
    gap: 10,
    alignItems: 'center',
  },
  edgeSpacer: {
    flexGrow: 1,
    minHeight: 0,
  },
  matchHeaderPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    minHeight: 44,
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  matchHeaderCopy: {
    flex: 1,
    gap: 2,
  },
  matchHeaderMode: {
    color: GOLD,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.9,
  },
  matchHeaderState: {
    color: WHITE,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 0.8,
  },
  matchHeaderExitBtn: {
    minWidth: 58,
    minHeight: 30,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  matchHeaderExitText: {
    color: WHITE,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  turnSeatChip: {
    width: '100%',
    minHeight: 58,
    borderWidth: 3,
    justifyContent: 'center',
    alignItems: 'stretch',
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  turnSeatInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  turnSeatCopy: {
    flex: 1,
    justifyContent: 'center',
    gap: 2,
  },
  seatBoxRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
  seatInfoBox: {
    minHeight: 16,
    paddingHorizontal: 4,
    borderWidth: 1,
    backgroundColor: PANEL_DARK,
    justifyContent: 'center',
    alignItems: 'center',
  },
  seatInfoBoxWide: {
    flex: 1,
    alignItems: 'flex-start',
  },
  seatInfoText: {
    color: WHITE,
    fontSize: 7,
    fontWeight: '900',
    letterSpacing: 0.3,
  },
  pieceMeterRow: {
    marginTop: 3,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  pieceMeterLabel: {
    width: 50,
    color: SOFT,
    fontSize: 7,
    fontWeight: '900',
    letterSpacing: 0.3,
  },
  pieceMeterTrack: {
    flex: 1,
    height: 8,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: '#161022',
    flexDirection: 'row',
  },
  pieceMeterFill: {
    height: '100%',
  },
  pieceMeterRest: {
    height: '100%',
  },
  telemetryToggleBtn: {
    minWidth: 62,
    minHeight: 26,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  telemetryToggleBtnOn: {
    borderColor: MINT,
    backgroundColor: '#19392e',
  },
  telemetryToggleBtnOff: {
    borderColor: LINE,
    backgroundColor: '#1b1330',
  },
  telemetryToggleText: {
    color: WHITE,
    fontSize: 9,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  turnSeatText: {
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.8,
    lineHeight: 13,
  },
  turnSeatRole: {
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1,
  },
  avatarShell: {
    width: 40,
    height: 40,
    borderWidth: 2,
    backgroundColor: PANEL_DARK,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarGrid: {
    width: 32,
    height: 32,
    flexDirection: 'row',
    flexWrap: 'wrap',
    backgroundColor: PANEL_DARK,
  },
  avatarPixel: {
    width: 2,
    height: 2,
  },
  boardPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 10,
    alignItems: 'center',
    overflow: 'hidden',
  },
  boardFrame: {
    padding: 10,
    backgroundColor: PANEL_DARK,
    borderWidth: 3,
    borderColor: GOLD,
    position: 'relative',
  },
  resultOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultOverlayBox: {
    minWidth: 140,
    minHeight: 72,
    borderWidth: 3,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: '#130d1e',
  },
  resultOverlayWin: {
    borderColor: MINT,
  },
  resultOverlayLoss: {
    borderColor: PINK,
  },
  resultOverlayDraw: {
    borderColor: GOLD,
  },
  resultOverlayTitle: {
    color: WHITE,
    fontSize: 22,
    fontWeight: '900',
    letterSpacing: 1.4,
  },
  resultOverlayAvatarRow: {
    marginTop: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  resultOverlayAvatarWrap: {
    width: 40,
    height: 40,
    position: 'relative',
  },
  lossCrossOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  lossCrossLine: {
    position: 'absolute',
    width: 58,
    height: 4,
    backgroundColor: '#000000',
    opacity: 0.95,
  },
  lossCrossLineA: {
    transform: [{ rotate: '45deg' }],
  },
  lossCrossLineB: {
    transform: [{ rotate: '-45deg' }],
  },
  comboBadge: {
    position: 'absolute',
    top: 8,
    zIndex: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 2,
    borderColor: GOLD,
    backgroundColor: '#ff4fa3',
  },
  comboBadgeText: {
    color: WHITE,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  recentMovesPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 2,
    borderColor: LINE,
    paddingHorizontal: 10,
    paddingVertical: 7,
    gap: 3,
  },
  recentMovesTitle: {
    color: GOLD,
    fontSize: 9,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  recentMovesText: {
    color: SOFT,
    fontSize: 10,
    fontWeight: '700',
    lineHeight: 14,
  },
  telemetryPanel: {
    width: '100%',
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 12,
    gap: 4,
  },
  telemetryTitle: {
    color: PINK,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
  },
  telemetryLine: {
    color: WHITE,
    fontSize: 11,
    lineHeight: 16,
  },
  actionRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 8,
  },
  primaryButton: {
    flex: 1,
    minHeight: 46,
    backgroundColor: PANEL_ALT,
    borderWidth: 3,
    borderColor: GOLD,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: {
    color: WHITE,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 46,
    backgroundColor: PANEL,
    borderWidth: 2,
    borderColor: LINE,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: SOFT,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  undoBtnLockedLook: {
    opacity: 0.7,
  },
  btnDisabled: {
    opacity: 0.45,
  },
});
