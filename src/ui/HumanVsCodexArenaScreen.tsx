import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ActivityIndicator, Alert, Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { AppLanguage } from '../../App';
import { B1, bitCount } from '../coreClaude/bitboards';
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
  easy: 900,
  normal: 1400,
  hard: 2800,
  expert: 5000,
};
const STRICT_DEPTH_CAP: Record<StrictDifficulty, number> = {
  easy: 7,
  normal: 9,
  hard: 11,
  expert: 13,
};
const STRICT_BUDGET_CAP_MS: Record<StrictDifficulty, number> = {
  easy: 1700,
  normal: 2500,
  hard: 4200,
  expert: 6500,
};
const AZ_ONLY_FAST_MS = 2800;
const MOVE_ANIM_GUARD_MS = 560;
const INITIAL_PIECES_PER_SIDE = 8;
const HINT_PRO_STRICT_MIN_MS = 2600;
const HINT_PRO_STRICT_MAX_MS = 5000;
const HINT_PRO_AZ_MS = 3600;

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
    p1Win: 'ผู้เล่น 1 ชนะ',
    p2Win: 'ผู้เล่น 2 ชนะ',
    player1: 'ผู้เล่น 1',
    player2: 'ผู้เล่น 2',
    youWin: 'คุณชนะ',
    youLose: 'คุณแพ้',
    human: 'ผู้เล่น',
    cap: 'กิน',
    turn: 'เวลา',
    last: 'ล่าสุด',
    forced: 'บังคับ',
    telOn: 'ข้อมูล AI เปิด',
    telOff: 'ข้อมูล AI ปิด',
    rewardWin: `รางวัลแมตช์ +${WIN_REWARD} เหรียญ`,
    rewardLose: `รางวัลแมตช์ +${LOSE_REWARD} เหรียญ`,
    rewardDraw: 'รางวัลแมตช์ +0 เหรียญ',
    winSub: 'เล่นได้ดีมาก ลุยระดับต่อไปได้เลย',
    loseSub: 'ลองใช้ Hint หรือ Undo แล้วสู้ใหม่',
    drawSub: 'สูสีมาก เล่นอีกตาไหม',
    forcedCapture: 'ถูกบังคับกิน: เลือกตัวหมากที่ถูกไฮไลท์',
    forcedMove: 'ถูกบังคับเดิน: เดินได้เฉพาะตัวที่ถูกไฮไลท์',
    telemetryTitle: 'ข้อมูลการคิดของ AI',
    telemetryPending: 'จะแสดงหลัง AI คิดจบอย่างน้อย 1 ครั้ง',
    hint: 'แนะนำ',
    hintTry: (from: number, to: number) => `ลองเดิน ${from + 1} -> ${to + 1}`,
    actionUnavailableTitle: 'ยังใช้งานไม่ได้',
    actionUnavailableBody: 'ลองอีกครั้ง',
    notEnoughCoinsTitle: 'เหรียญไม่พอ',
    notEnoughCoinsBody: 'ดูโฆษณาฟรี หรือเล่นแมตช์เพื่อรับเหรียญเพิ่ม',
    hintButton: 'แนะนำ',
    undoButton: 'ย้อนตา',
    walletLine: (coins: number, hintCredits: number, undoCredits: number) =>
      `เหรียญ ${coins} | แนะนำ ${hintCredits} | ย้อนตา ${undoCredits}`,
    modalHintTitle: 'ใช้คำแนะนำ',
    modalUndoTitle: 'ใช้ย้อนตา',
    loadingAd: 'กำลังโหลดโฆษณา...',
    hintCost: `คำแนะนำใช้ ${HINT_COST} เหรียญ หรือดูโฆษณาแทน`,
    undoCost: `ย้อนตาใช้ ${UNDO_COST} เหรียญ หรือดูโฆษณาแทน`,
    spendCoins: (cost: number) => `ใช้ ${cost} เหรียญ`,
    watchAdFree: 'ดูโฆษณาเพื่อใช้ฟรี',
    cancel: 'ยกเลิก',
    newGame: 'เริ่มเกมใหม่',
    exit: 'ออก',
  },
  en: {
    draw: 'DRAW',
    p1Win: 'PLAYER 1 WIN',
    p2Win: 'PLAYER 2 WIN',
    player1: 'Player 1',
    player2: 'Player 2',
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
    hintTry: (from: number, to: number) => `Try ${from + 1} -> ${to + 1}`,
    actionUnavailableTitle: 'Action unavailable',
    actionUnavailableBody: 'Please try again.',
    notEnoughCoinsTitle: 'Not enough coins',
    notEnoughCoinsBody: 'Watch ad for free, or earn more coins from matches.',
    hintButton: 'Hint',
    undoButton: 'Undo',
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
  return `${move.from + 1}->${move.to + 1}${cap}${promo}`;
}

function formatTurnSeconds(ms: number) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatHintRoute(move: Move) {
  return [move.from, ...movePath(move)].map(sq => sq + 1).join(' -> ');
}

function formatHintMessage(language: AppLanguage, move: Move) {
  const route = formatHintRoute(move);
  const caps = move.captured.length;
  if (language === 'th') {
    if (caps > 0) {
      const promo = move.promote ? ' + โปรโมต' : '';
      return `แนะนำ ${route} (กิน ${caps}${promo})`;
    }
    return move.promote ? `แนะนำ ${route} (โปรโมต)` : `แนะนำ ${route}`;
  }
  if (caps > 0) {
    const promo = move.promote ? ' + promote' : '';
    return `Try ${route} (capture ${caps}${promo})`;
  }
  return move.promote ? `Try ${route} (promote)` : `Try ${route}`;
}

function movePath(move: Move) {
  return move.path && move.path.length > 0 ? move.path : [move.to];
}

function pathStartsWith(path: number[], prefix: number[]) {
  return prefix.every((sq, idx) => path[idx] === sq);
}

function previewCaptureSteps(basePos: Position, move: Move, steps: number[]): Position {
  let preview: Position = { ...basePos };
  const myMen = basePos.side === 1 ? 'p1Men' : 'p2Men';
  const myKings = basePos.side === 1 ? 'p1Kings' : 'p2Kings';
  const opMen = basePos.side === 1 ? 'p2Men' : 'p1Men';
  const opKings = basePos.side === 1 ? 'p2Kings' : 'p1Kings';

  let cur = move.from;
  for (let i = 0; i < steps.length; i += 1) {
    const to = steps[i];
    const fromBit = B1(cur);
    const toBit = B1(to);
    const capturedBit = B1(move.captured[i]);
    const movingKing = ((preview as any)[myKings] & fromBit) !== 0;

    if (movingKing) (preview as any)[myKings] = (((preview as any)[myKings] & ~fromBit) | toBit) >>> 0;
    else (preview as any)[myMen] = (((preview as any)[myMen] & ~fromBit) | toBit) >>> 0;

    if ((preview as any)[opMen] & capturedBit) {
      (preview as any)[opMen] = ((preview as any)[opMen] & ~capturedBit) >>> 0;
    } else {
      (preview as any)[opKings] = ((preview as any)[opKings] & ~capturedBit) >>> 0;
    }
    cur = to;
  }

  return preview;
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
type CaptureStepState = {
  from: number;
  steps: number[];
  previewPos: Position;
};
type AvatarKind = 'human' | 'human-sad' | 'bot-easy' | 'bot-medium' | 'bot-hard';
type MatchResultTone = 'win' | 'loss' | 'draw';
type MatchResult = {
  label: string;
  tone: MatchResultTone;
  outcome: MatchOutcome;
  avatarKind?: AvatarKind;
};
type HintDialogPhase = 'thinking' | 'ready' | 'error';

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

function countTotalPieces(pos: Position): number {
  return bitCount(pos.p1Men | pos.p1Kings | pos.p2Men | pos.p2Kings);
}

function pickAdaptiveStrictDepth(difficulty: StrictDifficulty, pos: Position): number {
  const base = STRICT_MM_DEPTH[difficulty];
  const cap = STRICT_DEPTH_CAP[difficulty];
  const moves = generateMoves(pos);
  if (moves.length <= 1) return Math.min(cap, base + 1);

  const forcedCapture = moves[0].captured.length > 0;
  const hasMultiCapture = forcedCapture && moves.some(m => m.captured.length >= 2);
  const total = countTotalPieces(pos);

  let bonus = 0;
  if (forcedCapture) bonus += 1;
  if (hasMultiCapture) bonus += 1;
  if (total <= 10) bonus += 1;
  if (total <= 7) bonus += 1;

  return Math.min(cap, base + bonus);
}

function pickAdaptiveStrictBudgetMs(difficulty: StrictDifficulty, pos: Position): number {
  const base = STRICT_FAST_MS[difficulty];
  const cap = STRICT_BUDGET_CAP_MS[difficulty];
  const moves = generateMoves(pos);
  if (moves.length <= 1) return Math.min(cap, base + 100);

  const forcedCapture = moves[0].captured.length > 0;
  const hasMultiCapture = forcedCapture && moves.some(m => m.captured.length >= 2);
  const total = countTotalPieces(pos);
  const lowMobility = moves.length <= 3;

  let extra = 0;
  if (forcedCapture) extra += 220;
  if (hasMultiCapture) extra += 280;
  if (lowMobility) extra += 180;
  if (total <= 10) extra += 220;
  if (total <= 7) extra += 280;

  return Math.min(cap, base + extra);
}

function immediateCaptureRiskForMove(pos: Position, move: Move): number {
  const child = applyMove(pos, move);
  const oppMoves = generateMoves(child);
  if (!oppMoves.length) return -5_000;
  if (oppMoves[0].captured.length === 0) return 0;

  let maxCap = 0;
  let hangingMovedPieceMax = 0;
  for (const reply of oppMoves) {
    const cap = reply.captured.length;
    if (cap > maxCap) maxCap = cap;
    if (reply.captured.includes(move.to) && cap > hangingMovedPieceMax) {
      hangingMovedPieceMax = cap;
    }
  }

  let risk = maxCap * 140;
  if (maxCap >= 2) risk += 170;
  if (maxCap >= 3) risk += 220;
  if (hangingMovedPieceMax > 0) {
    risk += 260 + hangingMovedPieceMax * 200;
    if (hangingMovedPieceMax >= 2) risk += 220;
  }
  if (move.captured.length === 0) risk += 40;
  return risk;
}

function pickSaferFallbackMove(pos: Position, suggested: Move | undefined): Move | undefined {
  const legal = generateMoves(pos);
  if (!legal.length) return undefined;
  if (!suggested) return legal[0];

  const matched = legal.find(m =>
    m.from === suggested.from &&
    m.to === suggested.to &&
    m.promote === suggested.promote &&
    m.captured.length === suggested.captured.length,
  ) ?? suggested;

  const currentRisk = immediateCaptureRiskForMove(pos, matched);
  if (currentRisk < 220) return matched;

  const analyzed = legal.map(m => ({
    move: m,
    risk: immediateCaptureRiskForMove(pos, m),
  }));
  const safest = analyzed
    .sort((a, b) => (a.risk - b.risk) || (b.move.captured.length - a.move.captured.length))[0];
  if (!safest) return matched;
  if (safest.risk + 90 > currentRisk) return matched;

  return safest.move;
}

const DIFFICULTY_LABEL: Record<AppLanguage, Record<Difficulty, string>> = {
  th: {
    easy: 'ระดับ 1',
    normal: 'ระดับ 2',
    hard: 'ระดับ 3',
    expert: 'ระดับ 4',
    master: 'ระดับ 5',
  },
  en: {
    easy: 'Level 1',
    normal: 'Level 2',
    hard: 'Level 3',
    expert: 'Level 4',
    master: 'Level 5',
  },
};

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

function aiLevelTag(language: AppLanguage, difficulty: Difficulty) {
  return DIFFICULTY_LABEL[language][difficulty];
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
  rotate180 = false,
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
  rotate180?: boolean;
}) {
  return (
    <View
      style={[
        styles.turnSeatChip,
        {
          borderColor: active ? (forced ? GOLD : tint) : LINE,
          backgroundColor: active ? PANEL_ALT : PANEL_DARK,
          opacity: active ? 1 : 0.68,
          transform: [{ scale: active ? 1 : 0.98 }, ...(rotate180 ? [{ rotate: '180deg' as const }] : [])],
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
  const { mode, difficulty, humanSide } = config;
  const isHvH = mode === 'vs-human';
  const aiSide = (-humanSide) as 1 | -1;
  const aiThinkMs = !isStrictDifficulty(difficulty)
    ? AZ_ONLY_FAST_MS
    : STRICT_FAST_MS[difficulty];

  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [posHistory, setPosHistory] = useState<Position[]>(() => [initialPosition()]);
  const [hashHistory, setHashHistory] = useState<number[]>(() => [hashPosition(initialPosition())]);
  const [sel, setSel] = useState<number | null>(null);
  const [captureStep, setCaptureStep] = useState<CaptureStepState | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveHint[]>([]);
  const [lastMove, setLastMove] = useState<MoveHint | null>(null);
  const [shakeFrame, setShakeFrame] = useState(0);
  const [comboFrame, setComboFrame] = useState(0);
  const [comboText, setComboText] = useState('');
  const [turnElapsedMs, setTurnElapsedMs] = useState(0);
  const [isAnimLocked, setIsAnimLocked] = useState(false);
  const [spendModalKind, setSpendModalKind] = useState<SpendKind | null>(null);
  const [rewardLoadingKind, setRewardLoadingKind] = useState<SpendKind | null>(null);
  const [hintDialogVisible, setHintDialogVisible] = useState(false);
  const [hintDialogPhase, setHintDialogPhase] = useState<HintDialogPhase>('thinking');
  const [hintDialogText, setHintDialogText] = useState('');
  const [hintThinkingMs, setHintThinkingMs] = useState(0);
  const [surrenderDialogVisible, setSurrenderDialogVisible] = useState(false);
  const [manualResult, setManualResult] = useState<MatchResult | null>(null);
  const turnStartRef = useRef<number>(Date.now());
  const reportedResultKeyRef = useRef<string | null>(null);

  const { think, thinkStrict, thinking, cancel } = useCodexEngine();
  const { triggerFx } = usePixelGameFx({
    soundEnabled: monetization.soundEnabled,
    vibrationEnabled: monetization.vibrationEnabled,
  });
  const pendingRef = useRef<string | null>(null);
  const aiCommitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animUnlockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thinkStartTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animLockUntilRef = useRef<number>(0);
  const hintRequestSeqRef = useRef(0);

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
  const displayPos = captureStep?.previewPos ?? pos;
  const selectedDestSquares = useMemo(() => {
    if (sel === null) return [];
    const activeSteps = captureStep?.from === sel ? captureStep.steps : [];
    const nextBySquare = new Map<number, number>();

    for (const move of myMoves) {
      if (move.from !== sel) continue;
      const path = movePath(move);
      if (!pathStartsWith(path, activeSteps)) continue;
      const next = path[activeSteps.length];
      if (next === undefined) continue;
      nextBySquare.set(next, Math.max(nextBySquare.get(next) ?? 0, move.captured.length));
    }

    return Array.from(nextBySquare.entries()).map(([to, caps]) => ({ to, caps }));
  }, [captureStep, myMoves, sel]);
  const isDraw = useMemo(() => isDrawByInactivity(pos), [pos]);
  const curHash = useMemo(() => hashPosition(pos), [pos]);
  const isThreefold = useMemo(
    () => isThreefoldRepetition(buildRepetitionCounts(hashHistory), curHash),
    [curHash, hashHistory],
  );
  const gameResult = useMemo<MatchResult | null>(() => {
    if (manualResult) return manualResult;
    if (isDraw || isThreefold) return { label: t.draw, tone: 'draw', outcome: 'draw' };
    if (myMoves.length > 0) return null;

    const winnerSide = (pos.side === 1 ? -1 : 1) as 1 | -1;
    if (isHvH) return { label: winnerSide === 1 ? t.p1Win : t.p2Win, tone: 'win', outcome: 'win' };
    return winnerSide === humanSide
      ? { label: t.youWin, tone: 'win', outcome: 'win', avatarKind: 'human' as AvatarKind }
      : { label: t.youLose, tone: 'loss', outcome: 'loss', avatarKind: 'human-sad' as AvatarKind };
  }, [humanSide, isDraw, isHvH, isThreefold, manualResult, myMoves.length, pos.side, t.draw, t.p1Win, t.p2Win, t.youLose, t.youWin]);

  const canHumanMove =
    (isHvH || pos.side === humanSide) &&
    !thinking && !isAnimLocked && !isDraw && !isThreefold && !gameResult && myMoves.length > 0;
  const shakeX = [0, -6, 5, -4, 3, -2, 0][Math.min(shakeFrame, 6)];
  const comboOpacity = [0, 0.75, 1, 1, 0.9, 0.7, 0.45, 0.2, 0][Math.min(comboFrame, 8)];
  const comboLift = [24, 18, 14, 10, 6, 2, -2, -6, -10][Math.min(comboFrame, 8)];

  function commitMove(basePos: Position, move: Move, visualHint?: MoveHint) {
    const next = applyMove(basePos, move);
    const hint: MoveHint = { from: move.from, to: move.to, captured: move.captured.length, promote: move.promote };
    lockMoveAnimationWindow();
    setPos(next);
    setPosHistory(prev => [...prev, next]);
    setHashHistory(prev => [...prev, hashPosition(next)]);
    setMoveHistory(prev => [...prev, hint]);
    setLastMove(visualHint ?? hint);
    setCaptureStep(null);
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
    // Tactical finisher: if there is a legal move that leaves opponent with no move,
    // take it immediately instead of relying on low-budget search.
    const immediateWin = generateMoves(posSnapshot).find(move => {
      const next = applyMove(posSnapshot, move);
      return generateMoves(next).length === 0;
    });
    if (immediateWin) return Promise.resolve(immediateWin);

    if (isStrictDifficulty(difficulty)) {
      const adaptiveMs = pickAdaptiveStrictBudgetMs(difficulty, posSnapshot);
      const adaptiveDepth = pickAdaptiveStrictDepth(difficulty, posSnapshot);
      return thinkStrict(
        posSnapshot,
        adaptiveMs,
        histSnapshot,
        adaptiveDepth,
        undefined,
      ).then(best => pickSaferFallbackMove(posSnapshot, best));
    }
    return think(posSnapshot, aiThinkMs, histSnapshot, undefined, difficulty)
      .then(best => pickSaferFallbackMove(posSnapshot, best));
  }

  function computeHintMove(posSnapshot: Position, histSnapshot: number[]) {
    if (isStrictDifficulty(difficulty)) {
      const hintBudget = Math.min(HINT_PRO_STRICT_MAX_MS, Math.max(HINT_PRO_STRICT_MIN_MS, STRICT_FAST_MS[difficulty] + 1000));
      const hintDepth = Math.min(13, STRICT_MM_DEPTH[difficulty] + 2);
      return thinkStrict(posSnapshot, hintBudget, histSnapshot, hintDepth, undefined);
    }
    return thinkStrict(posSnapshot, HINT_PRO_AZ_MS, histSnapshot, 11, undefined);
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
    if (gameResult || !myMoves.length || isDraw || isThreefold) return;
    const id = setInterval(() => {
      if (!isAnimLocked) setTurnElapsedMs(Date.now() - turnStartRef.current);
    }, 220);
    return () => clearInterval(id);
  }, [gameResult, isAnimLocked, isDraw, isThreefold, myMoves.length, pos.side]);

  useEffect(() => {
    if (!hintDialogVisible || hintDialogPhase !== 'thinking') return;
    const startedAt = Date.now();
    setHintThinkingMs(0);
    const id = setInterval(() => {
      setHintThinkingMs(Date.now() - startedAt);
    }, 120);
    return () => clearInterval(id);
  }, [hintDialogPhase, hintDialogVisible]);

  function onTapSquare(i: number) {
    if (!canHumanMove) return;
    if (sel === null) {
      if (myMoves.some(m => m.from === i)) {
        setSel(i);
        setCaptureStep(null);
      }
      return;
    }

    const activeSteps = captureStep?.from === sel ? captureStep.steps : [];
    const stepMatches = myMoves.filter(m => {
      if (m.from !== sel) return false;
      const path = movePath(m);
      return pathStartsWith(path, activeSteps) && path[activeSteps.length] === i;
    });
    if (stepMatches.length > 0) {
      const nextSteps = [...activeSteps, i];
      const completed = stepMatches.find(m => movePath(m).length === nextSteps.length);
      if (completed) {
        const visualFrom = activeSteps.length > 0 ? activeSteps[activeSteps.length - 1] : completed.from;
        commitMove(pos, completed, {
          from: visualFrom,
          to: completed.to,
          captured: completed.captured.length,
          promote: completed.promote,
        });
        setSel(null);
        return;
      }
      const previewMove = stepMatches[0];
      setCaptureStep({
        from: sel,
        steps: nextSteps,
        previewPos: previewCaptureSteps(pos, previewMove, nextSteps),
      });
      return;
    }

    const move = myMoves.find(m => m.from === sel && m.to === i);
    if (move && movePath(move).length <= 1) {
      commitMove(pos, move);
      setSel(null);
      return;
    }
    if (myMoves.some(m => m.from === i)) {
      setSel(i);
      setCaptureStep(null);
    } else {
      setSel(null);
      setCaptureStep(null);
    }
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
    setCaptureStep(null);
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
    const requestSeq = ++hintRequestSeqRef.current;
    setHintDialogVisible(true);
    setHintDialogPhase('thinking');
    setHintDialogText('');
    setHintThinkingMs(0);
    const posSnapshot = pos;
    const histSnapshot = hashHistory;
    const fallbackMove = myMoves[0];
    if (myMoves.length === 1 && fallbackMove) {
      setSel(fallbackMove.from);
      setCaptureStep(null);
      if (hintRequestSeqRef.current !== requestSeq) return;
      setHintDialogText(formatHintMessage(language, fallbackMove));
      setHintDialogPhase('ready');
      return;
    }
    computeHintMove(posSnapshot, histSnapshot).then(best => {
      if (hintRequestSeqRef.current !== requestSeq) return;
      const suggested = best ?? fallbackMove;
      if (!suggested) {
        setHintDialogPhase('error');
        setHintDialogText(language === 'th' ? 'ไม่พบตาเดินที่เหมาะสม ลองใหม่อีกครั้ง' : 'No suggestion available. Please try again.');
        return;
      }
      setSel(suggested.from);
      setCaptureStep(null);
      setHintDialogText(formatHintMessage(language, suggested));
      setHintDialogPhase('ready');
    }).catch(() => {
      if (hintRequestSeqRef.current !== requestSeq) return;
      setHintDialogPhase('error');
      setHintDialogText(language === 'th' ? 'AI คิดไม่สำเร็จ ลองใหม่อีกครั้ง' : 'Hint search failed. Please try again.');
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
    setCaptureStep(null);
    const next = initialPosition();
    setPos(next);
    setPosHistory([next]);
    setHashHistory([hashPosition(next)]);
    setMoveHistory([]);
    setLastMove(null);
    setManualResult(null);
    hintRequestSeqRef.current += 1;
    setHintDialogVisible(false);
    setHintDialogPhase('thinking');
    setHintDialogText('');
    setHintThinkingMs(0);
    setSpendModalKind(null);
    setRewardLoadingKind(null);
    setSurrenderDialogVisible(false);
    reportedResultKeyRef.current = null;
  }


  const p1IsHuman = isHvH || humanSide === 1;
  const p2IsHuman = isHvH || humanSide === -1;
  const p1Role = p1IsHuman ? t.human : aiLevelTag(language, difficulty);
  const p2Role = p2IsHuman ? t.human : aiLevelTag(language, difficulty);
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
  const canSurrenderNow = !isHvH && !gameResult;
  const hintPreview = makeSpendPreview(monetization, 'hint');
  const undoPreview = makeSpendPreview(monetization, 'undo');
  const rewardLine = gameResult
    ? gameResult.outcome === 'win'
      ? t.rewardWin
      : gameResult.outcome === 'loss'
        ? t.rewardLose
        : t.rewardDraw
    : null;
  const hintBtnText = t.hintButton;
  const undoBtnText = t.undoButton;
  const surrenderBtnText = language === 'th' ? 'ยอมแพ้' : 'Surrender';

  useEffect(() => {
    if (!gameResult) return;
    const outcome: MatchOutcome = gameResult.outcome;
    const resultKey = `${outcome}:${moveHistory.length}:${pos.side}`;
    if (reportedResultKeyRef.current === resultKey) return;
    reportedResultKeyRef.current = resultKey;
    void onMatchComplete(outcome);
  }, [gameResult, moveHistory.length, onMatchComplete, pos.side]);

  function performSurrender() {
    if (!canSurrenderNow) return;
    setSurrenderDialogVisible(false);
    hintRequestSeqRef.current += 1;
    setHintDialogVisible(false);
    setHintDialogPhase('thinking');
    setHintDialogText('');
    setHintThinkingMs(0);
    cancel();
    pendingRef.current = null;
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);
    setSel(null);
    setCaptureStep(null);
    setSpendModalKind(null);
    setRewardLoadingKind(null);
    setManualResult({
      label: t.youLose,
      tone: 'loss',
      outcome: 'surrender',
      avatarKind: 'human-sad',
    });
  }

  function onPressSurrender() {
    if (!canSurrenderNow) return;
    setSurrenderDialogVisible(true);
  }

  function onExitBoard() {
    cancel();
    clearPendingAICommit();
    clearThinkStartTimer();
    clearAnimUnlockTimer();
    animLockUntilRef.current = 0;
    setIsAnimLocked(false);
    setCaptureStep(null);
    hintRequestSeqRef.current += 1;
    setHintDialogVisible(false);
    setHintDialogPhase('thinking');
    setHintDialogText('');
    setHintThinkingMs(0);
    onBack();
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.edgeSpacer} />

        <TurnSeatChip
          lane={t.player2}
          avatarKind={p2AvatarKind}
          role={p2Role}
          captured={p2Captured}
          turnTimer={p2Turn}
          lastMoveText={p2Last}
          active={pos.side === -1}
          forced={pos.side === -1 && isForcedTurn}
          tint={PINK}
          copy={t}
          rotate180={isHvH}
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
              pos={displayPos}
              onTapSquare={onTapSquare}
              fromSquares={captureStep ? [] : (isForcedTurn ? forcedFromSquares : [])}
              selectedFrom={captureStep ? captureStep.steps[captureStep.steps.length - 1] : sel}
              destSquares={selectedDestSquares}
              lastMove={lastMove}
              rotateNumbers180={isHvH && pos.side === -1}
              rotatePieces180={isHvH && pos.side === -1}
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
              ? `${pos.side === 1 ? t.player1 : t.player2} ${t.forcedCapture}`
              : `${pos.side === 1 ? t.player1 : t.player2} ${t.forcedMove}`}
          </Text>
        ) : null}

        <TurnSeatChip
          lane={t.player1}
          avatarKind={p1AvatarKind}
          role={p1Role}
          captured={p1Captured}
          turnTimer={p1Turn}
          lastMoveText={p1Last}
          active={pos.side === 1}
          forced={pos.side === 1 && isForcedTurn}
          tint={CYAN}
          copy={t}
        />

        {!isHvH ? (
          <>
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
              <Pressable
                style={[
                  styles.roundButton,
                  styles.surrenderButton,
                  !canSurrenderNow && styles.btnDisabled,
                ]}
                onPress={onPressSurrender}
                disabled={!canSurrenderNow}
              >
                <Text style={styles.roundButtonTitle}>{surrenderBtnText}</Text>
              </Pressable>
            </View>
            <Text style={styles.actionHelperText}>
              {t.walletLine(monetization.coins, monetization.hintCredits, monetization.undoCredits)}
            </Text>
          </>
        ) : null}

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
        <Modal
          visible={hintDialogVisible}
          animationType="fade"
          transparent
          onRequestClose={() => {
            if (hintDialogPhase !== 'thinking') setHintDialogVisible(false);
          }}
        >
          <View style={styles.hintBackdrop}>
            <Pressable
              style={StyleSheet.absoluteFill}
              onPress={() => {
                if (hintDialogPhase !== 'thinking') setHintDialogVisible(false);
              }}
            />
            <View style={styles.hintSheet}>
              <Text style={styles.hintTitle}>{t.hint}</Text>
              {hintDialogPhase === 'thinking' ? (
                <View style={styles.hintThinkingWrap}>
                  <ActivityIndicator size="small" color={CYAN} />
                  <Text style={styles.hintThinkingText}>
                    {language === 'th' ? 'AI กำลังคิดคำแนะนำ...' : 'AI is thinking...'}
                  </Text>
                  <Text style={styles.hintTimerText}>
                    {(hintThinkingMs / 1000).toFixed(1)}s {Math.floor(hintThinkingMs / 400) % 2 === 0 ? '⌛' : '⏳'}
                  </Text>
                </View>
              ) : (
                <Text style={styles.hintResultText}>{hintDialogText}</Text>
              )}
              <View style={styles.hintActionRow}>
                {hintDialogPhase === 'thinking' ? (
                  <Pressable
                    style={[styles.hintActionBtn, styles.hintCancelBtn]}
                    onPress={() => {
                      hintRequestSeqRef.current += 1;
                      setHintDialogVisible(false);
                      setHintDialogPhase('thinking');
                      setHintDialogText('');
                      setHintThinkingMs(0);
                    }}
                  >
                    <Text style={styles.hintCancelText}>{language === 'th' ? 'ซ่อน' : 'Hide'}</Text>
                  </Pressable>
                ) : (
                  <Pressable
                    style={[styles.hintActionBtn, styles.hintOkBtn]}
                    onPress={() => setHintDialogVisible(false)}
                  >
                    <Text style={styles.hintOkText}>{language === 'th' ? 'ตกลง' : 'OK'}</Text>
                  </Pressable>
                )}
              </View>
            </View>
          </View>
        </Modal>
        <Modal
          visible={surrenderDialogVisible}
          animationType="fade"
          transparent
          onRequestClose={() => setSurrenderDialogVisible(false)}
        >
          <View style={styles.confirmBackdrop}>
            <Pressable style={StyleSheet.absoluteFill} onPress={() => setSurrenderDialogVisible(false)} />
            <View style={styles.confirmSheet}>
              <Text style={styles.confirmTitle}>
                {language === 'th' ? 'ยอมแพ้เกมนี้?' : 'Surrender this game?'}
              </Text>
              <Text style={styles.confirmHelper}>
                {language === 'th'
                  ? 'จะนับแพ้ทันที และรางวัลเหรียญเป็น 0'
                  : 'This counts as a loss and gives 0 coins.'}
              </Text>
              <View style={styles.confirmRow}>
                <Pressable
                  style={[styles.confirmBtn, styles.confirmCancelBtn]}
                  onPress={() => setSurrenderDialogVisible(false)}
                >
                  <Text style={styles.confirmCancelText}>{t.cancel}</Text>
                </Pressable>
                <Pressable
                  style={[styles.confirmBtn, styles.confirmDangerBtn]}
                  onPress={performSurrender}
                >
                  <Text style={styles.confirmDangerText}>
                    {language === 'th' ? 'ยอมแพ้' : 'Surrender'}
                  </Text>
                </Pressable>
              </View>
            </View>
          </View>
        </Modal>

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
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  turnSeatText: {
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.3,
    lineHeight: 15,
  },
  turnSeatRole: {
    fontSize: 11,
    fontFamily: 'Kanit_700Bold',
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
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1.1,
  },
  resultOverlaySub: {
    marginTop: 6,
    color: SOFT,
    fontSize: 11,
    textAlign: 'center',
    lineHeight: 16,
    fontFamily: 'Kanit_500Medium',
  },
  resultOverlayReward: {
    marginTop: 6,
    color: GOLD,
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.7,
  },
  telemetryLine: {
    color: WHITE,
    fontSize: 10,
    lineHeight: 15,
    fontFamily: 'Kanit_500Medium',
  },
  actionRow: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginTop: 2,
  },
  roundButton: {
    width: 66,
    height: 48,
    borderRadius: 12,
    backgroundColor: 'rgba(23, 66, 61, 0.9)',
    borderWidth: 1,
    borderColor: LINE,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.14,
    shadowRadius: 5,
    shadowOffset: { width: 0, height: 3 },
    elevation: 2,
  },
  surrenderButton: {
    borderColor: 'rgba(242, 124, 124, 0.55)',
    backgroundColor: 'rgba(88, 41, 41, 0.88)',
  },
  roundButtonTitle: {
    color: WHITE,
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.35,
  },
  actionHelperText: {
    color: SOFT,
    fontSize: 11,
    marginTop: 2,
    fontFamily: 'Kanit_500Medium',
  },
  hintBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(12, 33, 31, 0.45)',
    justifyContent: 'center',
    padding: 18,
  },
  hintSheet: {
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 14,
    backgroundColor: 'rgba(27, 69, 64, 0.96)',
    padding: 12,
    gap: 10,
  },
  hintTitle: {
    color: WHITE,
    fontSize: 15,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  hintThinkingWrap: {
    minHeight: 74,
    borderWidth: 1,
    borderColor: 'rgba(155, 231, 218, 0.32)',
    borderRadius: 10,
    backgroundColor: 'rgba(33, 75, 70, 0.85)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 8,
    paddingVertical: 10,
  },
  hintThinkingText: {
    color: CYAN,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    textAlign: 'center',
  },
  hintTimerText: {
    color: SOFT,
    fontSize: 11,
    fontFamily: 'Kanit_700Bold',
    letterSpacing: 0.25,
  },
  hintResultText: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 17,
    fontFamily: 'Kanit_500Medium',
  },
  hintActionRow: {
    flexDirection: 'row',
  },
  hintActionBtn: {
    flex: 1,
    minHeight: 38,
    borderWidth: 1,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  hintCancelBtn: {
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
  },
  hintOkBtn: {
    borderColor: 'rgba(155, 231, 218, 0.6)',
    backgroundColor: PANEL_DARK,
  },
  hintCancelText: {
    color: SOFT,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  hintOkText: {
    color: CYAN,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  confirmBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(12, 33, 31, 0.45)',
    justifyContent: 'center',
    padding: 18,
  },
  confirmSheet: {
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 14,
    backgroundColor: 'rgba(27, 69, 64, 0.96)',
    padding: 12,
    gap: 10,
  },
  confirmTitle: {
    color: WHITE,
    fontSize: 15,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  confirmHelper: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 16,
    fontFamily: 'Kanit_500Medium',
  },
  confirmRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 2,
  },
  confirmBtn: {
    flex: 1,
    minHeight: 38,
    borderWidth: 1,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  confirmCancelBtn: {
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
  },
  confirmDangerBtn: {
    borderColor: 'rgba(242, 124, 124, 0.6)',
    backgroundColor: 'rgba(88, 41, 41, 0.9)',
  },
  confirmCancelText: {
    color: SOFT,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  confirmDangerText: {
    color: PINK,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
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
    fontFamily: 'Kanit_800ExtraBold',
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
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.4,
  },
  btnDisabled: {
    opacity: 0.45,
  },
});

