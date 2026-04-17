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

const THINK_MS: Record<Difficulty, number> = {
  easy: 900,
  normal: 1400,
  hard: 2200,
  expert: 3200,
  master: 4500,
};
const MOVE_ANIM_GUARD_MS = 560;
const INITIAL_PIECES_PER_SIDE = 8;

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_ALT = '#2b5f59';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const MINT = '#b8f3df';
const PINK = '#f2c5c5';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

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
  if (difficulty === 'expert' || difficulty === 'master') return 'bot-hard';
  if (difficulty === 'normal' || difficulty === 'hard') return 'bot-medium';
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
  return difficulty.toUpperCase();
}

function TurnSeatChip({
  lane,
  avatarKind,
  role,
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
            {lane} · {role}
          </Text>
          <Text style={[styles.turnSeatRole, { color: active ? tint : SOFT }]}>
            CAP x{captured} · TURN {turnTimer}{lastMoveText ? ` · LAST ${lastMoveText}` : ''}
          </Text>
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

    const undoLockedByMode = !isHvH && (difficulty === 'expert' || difficulty === 'master');
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
  const undoLockedByMode = !isHvH && (difficulty === 'expert' || difficulty === 'master');
  const canUndoNow = !thinking && posHistory.length > 1;
  const canHintNow = canHumanMove && !thinking && myMoves.length > 0;
  const undoBtnText = undoLockedByMode ? 'LOCK' : 'UNDO';

  function onExitBoard() {
    cancel();
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);
    onBack();
  }

  function onPressHint() {
    if (!canHintNow) return;
    if (myMoves.length === 1) {
      setSel(myMoves[0].from);
      return;
    }
    const posSnapshot = pos;
    const histSnapshot = hashHistory;
    think(posSnapshot, Math.min(1200, thinkMs), histSnapshot, undefined, difficulty).then(best => {
      if (!best) return;
      setSel(best.from);
      Alert.alert('Hint', `Try ${best.from} -> ${best.to}`);
    });
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.edgeSpacer} />

        <TurnSeatChip
          lane="P2"
          avatarKind={p2AvatarKind}
          role={p2Role}
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
              fromSquares={[]}
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
                  <Text style={styles.resultOverlaySub}>
                    {gameResult.tone === 'win'
                      ? 'Great run. Push to the next level.'
                      : gameResult.tone === 'loss'
                        ? 'Try Hint or Undo, then run it back.'
                        : 'Even match. One more round?'}
                  </Text>
                </View>
              </View>
            ) : null}
          </View>

        </View>

        <TurnSeatChip
          lane="P1"
          avatarKind={p1AvatarKind}
          role={p1Role}
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
              styles.roundButton,
              !canHintNow && styles.btnDisabled,
            ]}
            onPress={onPressHint}
            disabled={!canHintNow}
          >
            <Text style={styles.roundButtonTitle}>HINT</Text>
          </Pressable>
          <Pressable
            style={[
              styles.roundButton,
              undoLockedByMode && styles.undoBtnLockedLook,
              !canUndoNow && styles.btnDisabled,
            ]}
            onPress={onPressUndo}
            disabled={!canUndoNow}
          >
            <Text style={styles.roundButtonTitle}>{undoBtnText}</Text>
          </Pressable>
        </View>

        <View style={styles.bottomPillRow}>
          <Pressable style={styles.primaryButton} onPress={onNewGame}>
            <Text style={styles.primaryButtonText}>NEW GAME</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={onExitBoard}>
            <Text style={styles.secondaryButtonText}>EXIT</Text>
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
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 18,
    paddingTop: 12,
    paddingBottom: 22,
    gap: 12,
    alignItems: 'center',
  },
  edgeSpacer: {
    flexGrow: 1,
    minHeight: 0,
  },
  turnSeatChip: {
    width: '100%',
    minHeight: 54,
    borderWidth: 1,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'stretch',
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  turnSeatInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  turnSeatCopy: {
    flex: 1,
    justifyContent: 'center',
    gap: 1,
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
  telemetryToggleBtn: {
    minWidth: 56,
    minHeight: 24,
    borderWidth: 1,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  telemetryToggleBtnOn: {
    borderColor: MINT,
    backgroundColor: '#2f6b62',
  },
  telemetryToggleBtnOff: {
    borderColor: LINE,
    backgroundColor: '#285650',
  },
  telemetryToggleText: {
    color: WHITE,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.3,
  },
  turnSeatText: {
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.3,
    lineHeight: 15,
  },
  turnSeatRole: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  avatarShell: {
    width: 42,
    height: 42,
    borderWidth: 1.5,
    borderRadius: 9,
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
    backgroundColor: 'transparent',
    borderWidth: 0,
    padding: 4,
    alignItems: 'center',
    overflow: 'hidden',
  },
  boardFrame: {
    padding: 8,
    backgroundColor: '#3e6d66',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.22)',
    position: 'relative',
    shadowColor: '#000',
    shadowOpacity: 0.34,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 9 },
    elevation: 9,
  },
  resultOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(20, 46, 43, 0.56)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultOverlayBox: {
    minWidth: 210,
    minHeight: 96,
    borderWidth: 1,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'rgba(27,69,64,0.94)',
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
    fontSize: 28,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  resultOverlaySub: {
    marginTop: 6,
    color: SOFT,
    fontSize: 11,
    textAlign: 'center',
    lineHeight: 16,
  },
  comboBadge: {
    position: 'absolute',
    top: 8,
    zIndex: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    borderRadius: 10,
    backgroundColor: 'rgba(28, 79, 74, 0.92)',
  },
  comboBadgeText: {
    color: GOLD,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  telemetryPanel: {
    width: '100%',
    backgroundColor: 'rgba(27,69,64,0.7)',
    borderWidth: 1,
    borderRadius: 12,
    borderColor: LINE,
    padding: 10,
    gap: 4,
  },
  telemetryTitle: {
    color: GOLD,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  telemetryLine: {
    color: WHITE,
    fontSize: 10,
    lineHeight: 15,
  },
  actionRow: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 14,
    marginTop: 2,
  },
  roundButton: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: '#f2f3ef',
    borderWidth: 1,
    borderColor: 'rgba(53,87,82,0.4)',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 5 },
    elevation: 6,
  },
  roundButtonTitle: {
    color: '#4e5d5a',
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 0.6,
  },
  bottomPillRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 8,
    marginTop: 10,
  },
  primaryButton: {
    flex: 1,
    minHeight: 40,
    backgroundColor: 'rgba(22,64,59,0.9)',
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: {
    color: WHITE,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 40,
    backgroundColor: 'rgba(22,64,59,0.7)',
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: SOFT,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.4,
  },
  undoBtnLockedLook: {
    opacity: 0.7,
  },
  btnDisabled: {
    opacity: 0.45,
  },
});
