import React, { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { Difficulty, GameConfig, GameMode, MonetizationState } from './types';

interface Props {
  language: AppLanguage;
  initialConfig: GameConfig;
  monetization: MonetizationState;
  onBack: () => void;
  onPlay: (config: GameConfig) => void;
  onOpenAccount: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const MINT = '#b8f3df';
const CYAN = '#9be7da';
const PINK = '#f2c5c5';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

type SelectableDifficulty = Exclude<Difficulty, 'master'>;
const LEVELS: Array<{ id: SelectableDifficulty; tint: string }> = [
  { id: 'easy', tint: MINT },
  { id: 'normal', tint: CYAN },
  { id: 'hard', tint: GOLD },
  { id: 'expert', tint: PINK },
  { id: 'alpha', tint: '#ff6b9d' }, // Hot pink for NN-powered Alpha
];

function clampSelectableDifficulty(value: Difficulty): SelectableDifficulty {
  return value === 'master' ? 'expert' : value;
}

const COPY = {
  th: {
    back: 'กลับ',
    title: 'ตั้งค่าเกม',
    subtitle: 'เลือกโหมด ฝั่งเล่น และระดับ',
    mode: 'โหมด',
    side: 'ฝั่งเล่น',
    sideP1: 'ผู้เล่น 1',
    sideP2: 'ผู้เล่น 2',
    level: 'ระดับ',
    modeAi: 'เล่นกับ AI',
    modeHuman: 'เล่น 2 คน',
    selected: 'เลือกแล้ว',
    selectedLabel: 'ระดับที่เลือก',
    start: 'เริ่มเกม',
    adPlan: 'โหมดโฆษณา',
    noAdsActive: 'เปิด No Ads แล้ว - จะมีเฉพาะโฆษณาแบบสมัครใจ',
    freeMode: 'โหมดฟรี - มีโฆษณาคั่นหลังจบบางแมตช์',
    rewardCredits: 'เครดิตรางวัล',
    hint: 'แนะนำ',
    undo: 'ย้อนตา',
    account: 'บัญชี',
    levelLabels: {
      easy: 'ระดับ 1',
      normal: 'ระดับ 2',
      hard: 'ระดับ 3',
      expert: 'ระดับ 4',
      master: 'ระดับ 5',
      alpha: 'Alpha NN',
    },
    levelHints: {
      easy: 'สบายๆ',
      normal: 'สมดุล',
      hard: 'วางแผน',
      expert: 'เข้มข้น',
      master: 'ค้นหานำทาง',
      alpha: 'โมเดล Neural Network',
    },
  },
  en: {
    back: 'BACK',
    title: 'SETUP',
    subtitle: 'mode, side, level',
    mode: 'MODE',
    side: 'SIDE',
    sideP1: 'PLAYER 1',
    sideP2: 'PLAYER 2',
    level: 'LEVEL',
    modeAi: 'VS AI',
    modeHuman: 'VS HUMAN',
    selected: 'SELECTED',
    selectedLabel: 'Selected',
    start: 'START MATCH',
    adPlan: 'AD PLAN',
    noAdsActive: 'No Ads active - only optional rewarded ads',
    freeMode: 'Free mode - interstitial ads after some matches',
    rewardCredits: 'Reward credits',
    hint: 'Hint',
    undo: 'Undo',
    account: 'ACCOUNT',
    levelLabels: {
      easy: 'Level 1',
      normal: 'Level 2',
      hard: 'Level 3',
      expert: 'Level 4',
      master: 'Level 5',
      alpha: 'Alpha NN',
    },
    levelHints: {
      easy: 'Friendly',
      normal: 'Balanced',
      hard: 'Tactical',
      expert: 'Sharp',
      master: 'Guided search',
      alpha: 'Neural Network Model',
    },
  },
} as const;

function Chip({
  title,
  active,
  tint,
  onPress,
}: {
  title: string;
  active: boolean;
  tint: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, active && { borderColor: tint }]}>
      <Text style={[styles.chipText, active && { color: tint }]}>{title}</Text>
    </Pressable>
  );
}

export default function SetupScreen({ language, initialConfig, monetization, onBack, onPlay, onOpenAccount }: Props) {
  const [mode, setMode] = useState<GameMode>(initialConfig.mode);
  const [difficulty, setDifficulty] = useState<SelectableDifficulty>(clampSelectableDifficulty(initialConfig.difficulty));
  const [humanSide, setHumanSide] = useState<1 | -1>(initialConfig.humanSide);
  const t = COPY[language];

  const levelHint = useMemo(
    () => t.levelHints[difficulty],
    [difficulty, t.levelHints],
  );
  const levelTint = useMemo(
    () => LEVELS.find(x => x.id === difficulty)?.tint ?? CYAN,
    [difficulty],
  );
  const monetizationLine = monetization.noAdsUnlocked
    ? t.noAdsActive
    : t.freeMode;

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backText}>{t.back}</Text>
          </Pressable>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>{t.mode}</Text>
          <View style={styles.row}>
            <Chip title={t.modeAi} active={mode === 'vs-ai'} tint={GOLD} onPress={() => setMode('vs-ai')} />
            <Chip title={t.modeHuman} active={mode === 'vs-human'} tint={MINT} onPress={() => setMode('vs-human')} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>{t.side}</Text>
          <View style={styles.row}>
            <Chip title={t.sideP1} active={humanSide === 1} tint={CYAN} onPress={() => setHumanSide(1)} />
            <Chip title={t.sideP2} active={humanSide === -1} tint={PINK} onPress={() => setHumanSide(-1)} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>{t.level}</Text>
          <View style={styles.levelGrid}>
            {LEVELS.map(level => (
              <Pressable
                key={level.id}
                onPress={() => setDifficulty(level.id)}
                style={[
                  styles.levelCard,
                  difficulty === level.id && styles.levelCardActive,
                  difficulty === level.id && { borderColor: level.tint, backgroundColor: '#2a6159' },
                ]}
              >
                <Text style={[styles.levelTitle, difficulty === level.id && { color: level.tint }]}>
                  {t.levelLabels[level.id]}
                </Text>
                <Text style={styles.levelHint}>{t.levelHints[level.id]}</Text>
                {difficulty === level.id ? <Text style={[styles.levelSelectedTag, { color: level.tint }]}>{t.selected}</Text> : null}
              </Pressable>
            ))}
          </View>
          <Text style={[styles.levelCurrent, { color: levelTint }]}>
            {t.selectedLabel}: {t.levelLabels[difficulty]} - {levelHint}
          </Text>
        </View>

        <Pressable
          style={[styles.playButton, { borderColor: levelTint }]}
          onPress={() => onPlay({ mode, difficulty, humanSide, unlimitedThink: false })}
        >
          <Text style={styles.playText}>{t.start}</Text>
        </Pressable>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>{t.adPlan}</Text>
          <Text style={styles.helperText}>{monetizationLine}</Text>
          <Text style={styles.helperText}>
            {t.rewardCredits}: {t.hint} {monetization.hintCredits} | {t.undo} {monetization.undoCredits}
          </Text>
        </View>

        <Pressable style={styles.secondaryButton} onPress={onOpenAccount}>
          <Text style={styles.secondaryText}>{t.account}</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: BG },
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
  scrollContent: { paddingHorizontal: 18, paddingTop: 12, paddingBottom: 22, gap: 10 },
  header: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 14,
    padding: 10,
    gap: 3,
  },
  backButton: {
    alignSelf: 'flex-start',
    minHeight: 30,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    borderRadius: 999,
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  backText: { color: WHITE, fontSize: 10, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.8 },
  title: { color: WHITE, fontSize: 22, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 11, fontFamily: 'Kanit_500Medium' },
  panel: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    padding: 10,
    gap: 8,
  },
  sectionLabel: { color: WHITE, fontSize: 11, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.8 },
  row: { flexDirection: 'row', gap: 8 },
  chip: {
    flex: 1,
    minHeight: 40,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  chipText: { color: WHITE, fontSize: 11, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.7 },
  levelGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  levelCard: {
    width: '48%',
    minHeight: 52,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 6,
    justifyContent: 'center',
    gap: 1,
  },
  levelCardActive: {
    shadowColor: '#000',
    shadowOpacity: 0.24,
    shadowRadius: 7,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  levelTitle: { color: WHITE, fontSize: 13, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.8 },
  levelHint: { color: SOFT, fontSize: 10, fontFamily: 'Kanit_500Medium' },
  levelSelectedTag: { fontSize: 9, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.7 },
  levelCurrent: { fontSize: 11, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.4 },
  playButton: {
    minHeight: 48,
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: '#2b5f59',
    justifyContent: 'center',
    alignItems: 'center',
  },
  playText: { color: WHITE, fontSize: 13, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.8 },
  secondaryButton: {
    flex: 1,
    minHeight: 40,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL,
    borderRadius: 999,
    justifyContent: 'center',
    alignItems: 'center',
  },
  secondaryText: { color: SOFT, fontSize: 11, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.6 },
  helperText: { color: SOFT, fontSize: 11, lineHeight: 16, fontFamily: 'Kanit_500Medium' },
});

