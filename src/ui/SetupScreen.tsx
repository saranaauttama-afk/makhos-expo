import React, { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Difficulty, GameConfig, GameMode, MonetizationState } from './types';

interface Props {
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

const LEVELS: Array<{ id: Difficulty; label: string; hint: string; tint: string }> = [
  { id: 'easy', label: 'EASY', hint: 'Friendly', tint: MINT },
  { id: 'normal', label: 'NORMAL', hint: 'Balanced', tint: CYAN },
  { id: 'hard', label: 'HARD', hint: 'Tactical', tint: GOLD },
  { id: 'expert', label: 'EXPERT', hint: 'Sharp', tint: PINK },
  { id: 'master', label: 'MASTER', hint: 'Maximum', tint: '#ff8a5e' },
];

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

export default function SetupScreen({ initialConfig, monetization, onBack, onPlay, onOpenAccount }: Props) {
  const [mode, setMode] = useState<GameMode>(initialConfig.mode);
  const [difficulty, setDifficulty] = useState<Difficulty>(initialConfig.difficulty);
  const [humanSide, setHumanSide] = useState<1 | -1>(initialConfig.humanSide);

  const levelHint = useMemo(
    () => LEVELS.find(x => x.id === difficulty)?.hint ?? '',
    [difficulty],
  );
  const levelTint = useMemo(
    () => LEVELS.find(x => x.id === difficulty)?.tint ?? CYAN,
    [difficulty],
  );
  const monetizationLine = monetization.noAds
    ? 'No Ads active - only optional rewarded ads'
    : 'Free mode - interstitial ads after some matches';

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backText}>BACK</Text>
          </Pressable>
          <Text style={styles.title}>SETUP</Text>
          <Text style={styles.subtitle}>mode, side, level</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>MODE</Text>
          <View style={styles.row}>
            <Chip title="VS AI" active={mode === 'vs-ai'} tint={GOLD} onPress={() => setMode('vs-ai')} />
            <Chip title="VS HUMAN" active={mode === 'vs-human'} tint={MINT} onPress={() => setMode('vs-human')} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>SIDE</Text>
          <View style={styles.row}>
            <Chip title="P1" active={humanSide === 1} tint={CYAN} onPress={() => setHumanSide(1)} />
            <Chip title="P2" active={humanSide === -1} tint={PINK} onPress={() => setHumanSide(-1)} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>LEVEL</Text>
          <View style={styles.levelGrid}>
            {LEVELS.map(level => (
              <Pressable
                key={level.id}
                onPress={() => setDifficulty(level.id)}
                style={[styles.levelCard, difficulty === level.id && { borderColor: level.tint }]}
              >
                <Text style={[styles.levelTitle, difficulty === level.id && { color: level.tint }]}>{level.label}</Text>
                <Text style={styles.levelHint}>{level.hint}</Text>
              </Pressable>
            ))}
          </View>
          <Text style={[styles.levelCurrent, { color: levelTint }]}>Selected: {difficulty.toUpperCase()} - {levelHint}</Text>
        </View>

        <Pressable style={[styles.playButton, { borderColor: levelTint }]} onPress={() => onPlay({ mode, difficulty, humanSide })}>
          <Text style={styles.playText}>START MATCH</Text>
        </Pressable>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>AD PLAN</Text>
          <Text style={styles.helperText}>{monetizationLine}</Text>
          <Text style={styles.helperText}>Reward credits: Hint {monetization.rewardedHints} | Undo {monetization.rewardedUndos}</Text>
        </View>

        <Pressable style={styles.secondaryButton} onPress={onOpenAccount}>
          <Text style={styles.secondaryText}>ACCOUNT</Text>
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
  backText: { color: WHITE, fontSize: 10, fontWeight: '900', letterSpacing: 0.8 },
  title: { color: WHITE, fontSize: 22, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 11 },
  panel: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    padding: 10,
    gap: 8,
  },
  sectionLabel: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 0.8 },
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
  chipText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 0.7 },
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
  },
  levelTitle: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.8 },
  levelHint: { color: SOFT, fontSize: 10 },
  levelCurrent: { fontSize: 11, fontWeight: '900', letterSpacing: 0.4 },
  playButton: {
    minHeight: 48,
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: '#2b5f59',
    justifyContent: 'center',
    alignItems: 'center',
  },
  playText: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.8 },
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
  secondaryText: { color: SOFT, fontSize: 11, fontWeight: '800', letterSpacing: 0.6 },
  helperText: { color: SOFT, fontSize: 11, lineHeight: 16 },
});
