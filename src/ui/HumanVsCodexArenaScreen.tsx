import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { AppLanguage } from '../../App';
import { bitCount } from '../coreClaude/bitboards';
import { applyMove, generateMoves, Move } from '../coreClaude/movegen';
import { initialPosition, isDrawByInactivity, Position } from '../coreClaude/position';
import { buildRepetitionCounts, isThreefoldRepetition } from '../coreClaude/search/repetition';
import { hashPosition } from '../coreClaude/search/zobrist';
import { Board } from './Board';
import { useCodexEngine } from './useCodexEngine';
import { Difficulty, GameConfig, MonetizationState } from './types';
import { usePixelGameFx } from './usePixelGameFx';
import SpendOrWatchAdModal from './components/SpendOrWatchAdModal';
import { HINT_COST, LOSE_REWARD, makeSpendPreview, MatchOutcome, SpendKind, SpendSource, UNDO_COST, WIN_REWARD } from './walletStore';

type StrictDifficulty = Exclude<Difficulty, 'master'>;
const STRICT_MM_DEPTH: Record<StrictDifficulty, number> = {
  easy: 5,
  normal: 7,
  hard: 9,
  expert: 11,
};
const STRICT_FAST_MS: Record<StrictDifficulty, number> = {
  easy: 600,
  normal: 1400,
  hard: 2800,
  expert: 5000,
};
const AZ_ONLY_FAST_MS = 2800;
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

const COPY = {
  th: {
    draw: 'เสมอ',
    p1Win: 'P1 ชนะ',
    p2Win: 'P2 ชนะ',
    youWin: 'YOU WIN',
    youLose: 'YOU LOSE',
    human: 'HUMAN',
    cap: 'CAP',
    turn: 'TURN',
    last: 'LAST',
    forced: 'FORCED',
    telOn: 'TEL ON',
    telOff: 'TEL OFF',
    rewardWin: `Match reward +${WIN_REWARD} coins`,
    rewardLose: `Match reward +${LOSE_REWARD} coins`,
    rewardDraw: 'Match reward +0 coins',
    winSub: 'เล่นได้ดีมาก ลุยระดับต่อไปได้เลย',
    loseSub: 'ลองใช้ Hint หรือ Undo แล้วสู้ใหม่',
    drawSub: 'สูสีมาก เล่นอีกตาไหม',
    forcedCapture: 'ถูกบังคับกิน: เลือกตัวหมากที่ถูกไฮไลท์',
    forcedMove: 'ถูกบังคับเดิน: เดินได้เฉพาะตัวที่ถูกไฮไลท์',
    telemetryTitle: 'ENGINE TELEMETRY',
    telemetryPending: 'จะแสดงหลัง AI คิดจบอย่างน้อย 1 ครั้ง',
    hint: 'Hint',
    hintTry: (from: number, to: number) => `ลองเดิน ${from} -> ${to}`,
    actionUnavailableTitle: 'ยังใช้งานไม่ได้',
    actionUnavailableBody: 'ลองอีกครั้ง',
    notEnoughCoinsTitle: 'เหรียญไม่พอ',
    notEnoughCoinsBody: 'ดูโฆษณาฟรี หรือเล่นแมตช์เพื่อรับเหรียญเพิ่ม',
    hintButton: `HINT • ${HINT_COST}`,
    undoButton: `UNDO • ${UNDO_COST}`,
    walletLine: (coins: number, hintCredits: number, undoCredits: number) =>
      `เหรียญ ${coins} | Hint ${hintCredits} | Undo ${undoCredits}`,
    modalHintTitle: 'ใช้ Hint',
    modalUndoTitle: 'ใช้ Undo',
    loadingAd: 'กำลังโหลดโฆษณา...',
    hintCost: `Hint ใช้ ${HINT_COST} เหรียญ หรือดูโฆษณาแทน`,
    undoCost: `Undo ใช้ ${UNDO_COST} เหรียญ หรือดูโฆษณาแทน`,
    spendCoins: (cost: number) => `ใช้ ${cost} เหรียญ`,
    watchAdFree: 'ดูโฆษณาใช้ฟรี',
    cancel: 'ยกเลิก',
    newGame: 'NEW GAME',
    exit: 'EXIT',
  },
  en: {
    draw: 'DRAW',
    p1Win: 'P1 WIN',
    p2Win: 'P2 WIN',
    youWin: 'YOU WIN',
    youLose: 'YOU LOSE',
    human: 'HUMAN',
    cap: 'CAP',
    turn: 'TURN',
    last: 'LAST',
    forced: 'FORCED',
    telOn: 'TEL ON',
    telOff: 'TEL OFF',
    rewardWin: `Match reward +${WIN_REWARD} coins`,
    rewardLose: `Match reward +${LOSE_REWARD} coins`,
    rewardDraw: 'Match reward +0 coins',
    winSub: 'Great run. Push to the next level.',
    loseSub: 'Try Hint or Undo, then run it back.',
    drawSub: 'Even match. One more round?',
    forcedCapture: 'forced capture: pick a highlighted piece.',
    forcedMove: 'forced move: only highlighted piece can move.',
    telemetryTitle: 'ENGINE TELEMETRY',
    telemetryPending: 'Telemetry appears after the first AI search completes.',
    hint: 'Hint',
    hintTry: (from: number, to: number) => `Try ${from} -> ${to}`,
    actionUnavailableTitle: 'Action unavailable',
    actionUnavailableBody: 'Please try again.',
    notEnoughCoinsTitle: 'Not enough coins',
    notEnoughCoinsBody: 'Watch ad for free, or earn more coins from matches.',
    hintButton: `HINT • ${HINT_COST}`,
    undoButton: `UNDO • ${UNDO_COST}`,
    walletLine: (coins: number, hintCredits: number, undoCredits: number) =>
      `Coins ${coins} | Hint credits ${hintCredits} | Undo credits ${undoCredits}`,
    modalHintTitle: 'Use Hint',
    modalUndoTitle: 'Use Undo',
    loadingAd: 'Loading rewarded ad...',
    hintCost: `Hint uses ${HINT_COST} coins unless you watch an ad.`,
    undoCost: `Undo uses ${UNDO_COST} coins unless you watch an ad.`,
    spendCoins: (cost: number) => `Spend ${cost} coins`,
    watchAdFree: 'Watch Ad for free',
    cancel: 'Cancel',
    newGame: 'NEW GAME',
    exit: 'EXIT',
  },
} as const;
type ScreenCopy = (typeof COPY)[keyof typeof COPY];

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
  language: AppLanguage;
  config: GameConfig;
  monetization: MonetizationState;
  onConsumeSpend: (kind: SpendKind) => SpendSource;
  onWatchRewarded: (kind: SpendKind) => Promise<boolean>;
  onMatchComplete: (outcome: MatchOutcome) => void | Promise<void>;
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

function isStrictDifficulty(difficulty: Difficulty): difficulty is StrictDifficulty {
  return difficulty !== 'master';
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
  if (!isStrictDifficulty(difficulty)) return 'AZ';
  return `MM${STRICT_MM_DEPTH[difficulty]}`;
}

function TurnSeatChipLegacy({
  lane,
  avatarKind,
  role,
  captured,
  turnTimer,
  lastMoveText,
  active,
  forced = false,
  tint,
  copy,
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
  forced?: boolean;
  tint: string;
  copy: ScreenCopy;
  showTelemetryToggle?: boolean;
  telemetryEnabled?: boolean;
  onToggleTelemetry?: () => void;
}) {
  return (
    <View
      style={[
        styles.turnSeatChip,
        {
          borderColor: active ? (forced ? GOLD : tint) : LINE,
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
        {forced ? (
          <View style={styles.forcedBadge}>
            <Text style={styles.forcedBadgeText}>{copy.forced}</Text>
          </View>
        ) : null}
        {showTelemetryToggle && onToggleTelemetry ? (
          <Pressable
            onPress={onToggleTelemetry}
            style={[
              styles.telemetryToggleBtn,
              telemetryEnabled ? styles.telemetryToggleBtnOn : styles.telemetryToggleBtnOff,
            ]}
          >
            <Text style={styles.telemetryToggleText}>
              {telemetryEnabled ? copy.telOn : copy.telOff}
            </Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

function TurnSeatChip({
  lane,
  avatarKind,
  role,
  captured,
  turnTimer,
  lastMoveText,
  active,
  forced = false,
  tint,
  copy,
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
  forced?: boolean;
  tint: string;
  copy: ScreenCopy;
  showTelemetryToggle?: boolean;
  telemetryEnabled?: boolean;
  onToggleTelemetry?: () => void;
}) {
  return (
    <View
      style={[
        styles.turnSeatChip,
        {
          borderColor: active ? (forced ? GOLD : tint) : LINE,
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
            {copy.cap} x{captured} · {copy.turn} {turnTimer}{lastMoveText ? ` · ${copy.last} ${lastMoveText}` : ''}
          </Text>
        </View>
        {forced ? (
          <View style={styles.forcedBadge}>
            <Text style={styles.forcedBadgeText}>{copy.forced}</Text>
          </View>
        ) : null}
        {showTelemetryToggle && onToggleTelemetry ? (
          <Pressable
            onPress={onToggleTelemetry}
            style={[
              styles.telemetryToggleBtn,
              telemetryEnabled ? styles.telemetryToggleBtnOn : styles.telemetryToggleBtnOff,
            ]}
          >
            <Text style={styles.telemetryToggleText}>
              {telemetryEnabled ? copy.telOn : copy.telOff}
            </Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

export default function HumanVsCodexArenaScreen({
  language,
  config,
  monetization,
  onConsumeSpend,
  onWatchRewarded,
  onMatchComplete,
  onBack,
}: Props) {
  const t = COPY[language];
  const { mode, difficulty, humanSide, unlimitedThink = false } = config;
  const isHvH = mode === 'vs-human';
  const aiSide = (-humanSide) as 1 | -1;
  const aiThinkMs = !isStrictDifficulty(difficulty)
    ? (unlimitedThink ? 0 : AZ_ONLY_FAST_MS)
    : (unlimitedThink ? 0 : STRICT_FAST_MS[difficulty]);

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
  const [turnElapsedMs, setTurnElapsedMs] = useState(0);
  const [isAnimLocked, setIsAnimLocked] = useState(false);
  const [spendModalKind, setSpendModalKind] = useState<SpendKind | null>(null);
  const [rewardLoadingKind, setRewardLoadingKind] = useState<SpendKind | null>(null);
  const turnStartRef = useRef<number>(Date.now());
  const reportedResultKeyRef = useRef<string | null>(null);

  const { think, thinkStrict, thinking, lastInfo, lastPlan, cancel } = useCodexEngine();
  const { triggerFx } = usePixelGameFx();
  const pendingRef = useRef<string | null>(null);
  const aiCommitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animUnlockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thinkStartTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animLockUntilRef = useRef<number>(0);

  const myMoves = useMemo(() => generateMoves(pos), [pos]);
  const forcedFromSquares = useMemo(() => {
    if (!myMoves.length) return [];
    const fromSquares = Array.from(new Set(myMoves.map(m => m.from)));
    const forcedCapture = myMoves[0].captured.length > 0;
    if (forcedCapture || fromSquares.length === 1) return fromSquares;
    return [];
  }, [myMoves]);
  const isForcedTurn = forcedFromSquares.length > 0;
  const isForcedCaptureTurn = isForcedTurn && myMoves.length > 0 && myMoves[0].captured.length > 0;
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const gameResult = useMemo(() => {
    if (isDraw || isThreefold) return { label: t.draw, tone: 'draw' as const };
    if (myMoves.length > 0) return null;

    const winnerSide = (pos.side === 1 ? -1 : 1) as 1 | -1;
    if (isHvH) return { label: winnerSide === 1 ? t.p1Win : t.p2Win, tone: 'win' as const };
    return winnerSide === humanSide
      ? { label: t.youWin, tone: 'win' as const, avatarKind: 'human' as AvatarKind }
      : { label: t.youLose, tone: 'loss' as const, avatarKind: 'human-sad' as AvatarKind };
  }, [humanSide, isDraw, isHvH, isThreefold, myMoves.length, pos.side, t.draw, t.p1Win, t.p2Win, t.youLose, t.youWin]);

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

  function computeAIMove(posSnapshot: Position, histSnapshot: number[]) {
    if (isStrictDifficulty(difficulty)) {
      return thinkStrict(
        posSnapshot,
        aiThinkMs,
        histSnapshot,
        STRICT_MM_DEPTH[difficulty],
        undefined,
      );
    }
    return think(posSnapshot, aiThinkMs, histSnapshot, undefined, difficulty);
  }

  function computeHintMove(posSnapshot: Position, histSnapshot: number[]) {
    if (isStrictDifficulty(difficulty)) {
      const hintBudget = Math.min(1200, STRICT_FAST_MS[difficulty]);
      const hintDepth = Math.min(STRICT_MM_DEPTH[difficulty], 7);
      return thinkStrict(posSnapshot, hintBudget, histSnapshot, hintDepth, undefined);
    }
    return think(posSnapshot, Math.min(1200, AZ_ONLY_FAST_MS), histSnapshot, undefined, difficulty);
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
      computeAIMove(posSnapshot, histSnapshot).then(best => {
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
    if (!canUndoNow || thinking) return;
    if (undoPreview.availableCredits > 0) {
      const source = onConsumeSpend('undo');
      if (source !== 'none') runUndo();
      return;
    }
    setSpendModalKind('undo');
  }

  function runHintNow() {
    if (myMoves.length === 1) {
      setSel(myMoves[0].from);
      return;
    }
    const posSnapshot = pos;
    const histSnapshot = hashHistory;
    computeHintMove(posSnapshot, histSnapshot).then(best => {
      if (!best) return;
      setSel(best.from);
      Alert.alert(t.hint, t.hintTry(best.from, best.to));
    });
  }

  function onPressHint() {
    if (!canHintNow) return;
    if (hintPreview.availableCredits > 0) {
      const source = onConsumeSpend('hint');
      if (source !== 'none') runHintNow();
      return;
    }
    setSpendModalKind('hint');
  }

  async function onSpendModalWatchAd() {
    if (!spendModalKind) return;
    setRewardLoadingKind(spendModalKind);
    const ok = await onWatchRewarded(spendModalKind);
    setRewardLoadingKind(null);
    if (!ok) return;

    const source = onConsumeSpend(spendModalKind);
    if (source === 'none') {
      Alert.alert(t.actionUnavailableTitle, t.actionUnavailableBody);
      return;
    }

    const kind = spendModalKind;
    setSpendModalKind(null);
    if (kind === 'hint') runHintNow();
    else runUndo();
  }

  function onSpendModalSpendCoins() {
    if (!spendModalKind) return;
    const source = onConsumeSpend(spendModalKind);
    if (source === 'none') {
      Alert.alert(t.notEnoughCoinsTitle, t.notEnoughCoinsBody);
      return;
    }
    const kind = spendModalKind;
    setSpendModalKind(null);
    if (kind === 'hint') runHintNow();
    else runUndo();
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
    setSpendModalKind(null);
    reportedResultKeyRef.current = null;
  }

  const pvText = lastInfo?.pv.map((m: Move) => `${m.from}->${m.to}`).join(' ');

  const p1IsHuman = isHvH || humanSide === 1;
  const p2IsHuman = isHvH || humanSide === -1;
  const p1Role = p1IsHuman ? t.human : aiLevelTag(difficulty);
  const p2Role = p2IsHuman ? t.human : aiLevelTag(difficulty);
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
  const canUndoNow = !thinking && posHistory.length > 1;
  const canHintNow = canHumanMove && !thinking && myMoves.length > 0;
  const hintPreview = makeSpendPreview(monetization, 'hint');
  const undoPreview = makeSpendPreview(monetization, 'undo');
  const rewardLine = gameResult
    ? gameResult.tone === 'win'
      ? t.rewardWin
      : gameResult.tone === 'loss'
        ? t.rewardLose
        : t.rewardDraw
    : null;
  const hintBtnText = t.hintButton;
  const undoBtnText = t.undoButton;

  useEffect(() => {
    if (!gameResult) return;
    const outcome: MatchOutcome =
      gameResult.tone === 'win' ? 'win' : gameResult.tone === 'loss' ? 'loss' : 'draw';
    const resultKey = `${outcome}:${moveHistory.length}:${pos.side}`;
    if (reportedResultKeyRef.current === resultKey) return;
    reportedResultKeyRef.current = resultKey;
    void onMatchComplete(outcome);
  }, [gameResult, moveHistory.length, onMatchComplete, pos.side]);

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
          forced={pos.side === -1 && isForcedTurn}
          tint={PINK}
          copy={t}
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
              fromSquares={isForcedTurn ? forcedFromSquares : []}
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
                      ? t.winSub
                      : gameResult.tone === 'loss'
                        ? t.loseSub
                        : t.drawSub}
                  </Text>
                  {rewardLine ? <Text style={styles.resultOverlayReward}>{rewardLine}</Text> : null}
                </View>
              </View>
            ) : null}
          </View>

        </View>
        {isForcedTurn && !gameResult ? (
          <Text style={styles.forcedTurnHelper}>
            {isForcedCaptureTurn
              ? `${pos.side === 1 ? 'P1' : 'P2'} ${t.forcedCapture}`
              : `${pos.side === 1 ? 'P1' : 'P2'} ${t.forcedMove}`}
          </Text>
        ) : null}

        <TurnSeatChip
          lane="P1"
          avatarKind={p1AvatarKind}
          role={p1Role}
          captured={p1Captured}
          turnTimer={p1Turn}
          lastMoveText={p1Last}
          active={pos.side === 1}
          forced={pos.side === 1 && isForcedTurn}
          tint={CYAN}
          copy={t}
          showTelemetryToggle={p1IsBot}
          telemetryEnabled={showTelemetry}
          onToggleTelemetry={() => setShowTelemetry(v => !v)}
        />

        {showTelemetry ? (
          <View style={styles.telemetryPanel}>
            <Text style={styles.telemetryTitle}>{t.telemetryTitle}</Text>
            {!isHvH && lastInfo ? (
              <>
                <Text style={styles.telemetryLine}>depth {lastInfo.depth} | score {lastInfo.score} | nodes {lastInfo.nodes}</Text>
                {lastPlan ? <Text style={styles.telemetryLine}>mode {lastPlan.mode} | {lastPlan.reason}</Text> : null}
                {pvText ? <Text style={styles.telemetryLine}>pv {pvText}</Text> : null}
              </>
            ) : (
              <Text style={styles.telemetryLine}>{t.telemetryPending}</Text>
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
            <Text style={styles.roundButtonTitle}>{hintBtnText}</Text>
          </Pressable>
          <Pressable
            style={[
              styles.roundButton,
              !canUndoNow && styles.btnDisabled,
            ]}
            onPress={onPressUndo}
            disabled={!canUndoNow}
          >
            <Text style={styles.roundButtonTitle}>{undoBtnText}</Text>
          </Pressable>
        </View>
        <Text style={styles.actionHelperText}>
          {t.walletLine(monetization.coins, monetization.hintCredits, monetization.undoCredits)}
        </Text>

        <SpendOrWatchAdModal
          visible={spendModalKind !== null}
          title={spendModalKind === 'hint' ? t.modalHintTitle : t.modalUndoTitle}
          helperText={
            rewardLoadingKind
              ? t.loadingAd
              : spendModalKind === 'hint'
                ? t.hintCost
                : t.undoCost
          }
          cost={spendModalKind === 'hint' ? HINT_COST : UNDO_COST}
          canSpendCoins={spendModalKind === 'hint' ? hintPreview.canSpendCoins : undoPreview.canSpendCoins}
          spendLabel={t.spendCoins(spendModalKind === 'hint' ? HINT_COST : UNDO_COST)}
          watchAdLabel={t.watchAdFree}
          cancelLabel={t.cancel}
          onSpendCoins={onSpendModalSpendCoins}
          onWatchAd={() => { void onSpendModalWatchAd(); }}
          onCancel={() => {
            if (!rewardLoadingKind) setSpendModalKind(null);
          }}
        />

        <View style={styles.bottomPillRow}>
          <Pressable style={styles.primaryButton} onPress={onNewGame}>
            <Text style={styles.primaryButtonText}>{t.newGame}</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={onExitBoard}>
            <Text style={styles.secondaryButtonText}>{t.exit}</Text>
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
  forcedBadge: {
    minWidth: 56,
    minHeight: 24,
    borderWidth: 1,
    borderRadius: 999,
    borderColor: 'rgba(255,240,197,0.95)',
    backgroundColor: 'rgba(126,88,22,0.78)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 7,
  },
  forcedBadgeText: {
    color: '#fff4d2',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.4,
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
  forcedTurnHelper: {
    color: GOLD,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.25,
    marginTop: -2,
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
  resultOverlayReward: {
    marginTop: 6,
    color: GOLD,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.35,
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
  actionHelperText: {
    color: SOFT,
    fontSize: 11,
    marginTop: 2,
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
  btnDisabled: {
    opacity: 0.45,
  },
});
