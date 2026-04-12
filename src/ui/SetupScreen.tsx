import React, { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Difficulty, GameConfig, GameMode } from './types';

interface Props {
  initialConfig: GameConfig;
  onBack: () => void;
  onPlay: (config: GameConfig) => void;
  onPreviewResult: () => void;
  onOpenShop: () => void;
}

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_ALT = '#2c1f49';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const MINT = '#77f7cf';
const CYAN = '#5ec5ff';
const PINK = '#ff7dc4';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

function PixelChip({
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
    <Pressable onPress={onPress} style={[styles.chip, active && { borderColor: tint, backgroundColor: PANEL_ALT }]}>
      <Text style={[styles.chipText, active && { color: tint }]}>{title}</Text>
    </Pressable>
  );
}

function LadderCard({
  title,
  tint,
  copy,
}: {
  title: string;
  tint: string;
  copy: string;
}) {
  return (
    <View style={[styles.ladderCard, { borderColor: tint }]}>
      <Text style={[styles.ladderTitle, { color: tint }]}>{title}</Text>
      <Text style={styles.ladderCopy}>{copy}</Text>
    </View>
  );
}

export default function SetupScreen({ initialConfig, onBack, onPlay, onPreviewResult, onOpenShop }: Props) {
  const [mode, setMode] = useState<GameMode>(initialConfig.mode);
  const [difficulty, setDifficulty] = useState<Difficulty>(initialConfig.difficulty);
  const [humanSide, setHumanSide] = useState<1 | -1>(initialConfig.humanSide);

  const premiumLabel = useMemo(() => {
    if (difficulty === 'hard') return 'Strong AI tier may be gated by rewarded ads or premium unlock.';
    if (difficulty === 'medium') return 'Balanced AI tier for everyday play.';
    return 'Friendly quick-play tier with fast turns.';
  }, [difficulty]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>BACK</Text>
          </Pressable>

          <Text style={styles.kicker}>MATCH SETUP</Text>
          <Text style={styles.title}>PLAYER LOADOUT</Text>
          <Text style={styles.subtitle}>Skeleton setup screen for mode, side, and monetized AI tiers.</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>MODE</Text>
          <View style={styles.row}>
            <PixelChip title="VS AI" active={mode === 'vs-ai'} tint={GOLD} onPress={() => setMode('vs-ai')} />
            <PixelChip title="VS HUMAN" active={mode === 'vs-human'} tint={MINT} onPress={() => setMode('vs-human')} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>SIDE SELECT</Text>
          <View style={styles.row}>
            <PixelChip title="P1 START" active={humanSide === 1} tint={CYAN} onPress={() => setHumanSide(1)} />
            <PixelChip title="P2 REACT" active={humanSide === -1} tint={PINK} onPress={() => setHumanSide(-1)} />
          </View>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>AI TIER</Text>
          <View style={styles.rowWrap}>
            <PixelChip title="EASY" active={difficulty === 'easy'} tint={MINT} onPress={() => setDifficulty('easy')} />
            <PixelChip title="MEDIUM" active={difficulty === 'medium'} tint={GOLD} onPress={() => setDifficulty('medium')} />
            <PixelChip title="HARD" active={difficulty === 'hard'} tint={PINK} onPress={() => setDifficulty('hard')} />
          </View>
          <Text style={styles.metaCopy}>{premiumLabel}</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>PRODUCT LADDER</Text>
          <View style={styles.stack}>
            <LadderCard title="FREE PLAY" tint={MINT} copy="Quick matches, lighter AI, ads shown outside the board flow." />
            <LadderCard title="REWARDED MATCH" tint={CYAN} copy="Watch one ad to unlock a stronger AI battle for a single run." />
            <LadderCard title="PREMIUM PASS" tint={PINK} copy="Remove ads and unlock stronger AI tiers permanently." />
          </View>
        </View>

        <View style={styles.footer}>
          <Pressable style={[styles.primaryButton, { borderColor: difficulty === 'hard' ? PINK : difficulty === 'medium' ? GOLD : MINT }]} onPress={() => onPlay({ mode, difficulty, humanSide })}>
            <Text style={styles.primaryButtonText}>START MATCH</Text>
          </Pressable>

          <View style={styles.secondaryRow}>
            <Pressable style={styles.secondaryButton} onPress={onPreviewResult}>
              <Text style={styles.secondaryButtonText}>RESULT PREVIEW</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={onOpenShop}>
              <Text style={styles.secondaryButtonText}>SHOP</Text>
            </Pressable>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: BG },
  scrollContent: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 24, gap: 16 },
  headerPanel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    gap: 10,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
  },
  backButtonText: { color: WHITE, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 11, fontWeight: '800', letterSpacing: 1.3 },
  title: { color: WHITE, fontSize: 30, fontWeight: '900', letterSpacing: 1.1 },
  subtitle: { color: SOFT, fontSize: 13, lineHeight: 19 },
  panel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    gap: 12,
  },
  sectionTitle: { color: WHITE, fontSize: 16, fontWeight: '900', letterSpacing: 1.1 },
  row: { flexDirection: 'row', gap: 10 },
  rowWrap: { flexDirection: 'row', gap: 10, flexWrap: 'wrap' },
  chip: {
    flex: 1,
    minHeight: 52,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  chipText: { color: WHITE, fontSize: 13, fontWeight: '800', letterSpacing: 1 },
  metaCopy: { color: SOFT, fontSize: 12, lineHeight: 18 },
  stack: { gap: 10 },
  ladderCard: {
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    padding: 12,
    gap: 6,
  },
  ladderTitle: { fontSize: 14, fontWeight: '900', letterSpacing: 1 },
  ladderCopy: { color: SOFT, fontSize: 12, lineHeight: 18 },
  footer: { gap: 10 },
  primaryButton: {
    minHeight: 58,
    borderWidth: 3,
    backgroundColor: PANEL_ALT,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: { color: WHITE, fontSize: 16, fontWeight: '900', letterSpacing: 1 },
  secondaryRow: { flexDirection: 'row', gap: 10 },
  secondaryButton: {
    flex: 1,
    minHeight: 50,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: { color: SOFT, fontSize: 12, fontWeight: '800', letterSpacing: 0.9 },
});
