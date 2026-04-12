import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Difficulty, GameConfig, GameMode } from './types';

interface Props {
  onStart: (config: GameConfig) => void;
  onArena: () => void;
}

type UiDifficulty = 'beginner' | 'easy' | 'medium' | 'hard' | 'master';

const SURFACE = '#f6f1e8';
const CARD = '#fffaf2';
const INK = '#201713';
const MUTED = '#7c6e64';
const LINE = '#dac9b8';
const ACCENT = '#c46834';
const ACCENT_SOFT = '#f4dccd';
const DARK = '#3a2a22';

const DIFFICULTIES: Array<{
  id: UiDifficulty;
  title: string;
  subtitle: string;
  badge: string;
  engine: Difficulty;
}> = [
  { id: 'beginner', title: 'Beginner', subtitle: 'Relaxed turns and lighter pressure', badge: 'B', engine: 'easy' },
  { id: 'easy', title: 'Easy', subtitle: 'Friendly matches for casual play', badge: 'E', engine: 'easy' },
  { id: 'medium', title: 'Medium', subtitle: 'Balanced default challenge', badge: 'M', engine: 'medium' },
  { id: 'hard', title: 'Hard', subtitle: 'Sharper tactics and longer thinking', badge: 'H', engine: 'hard' },
  { id: 'master', title: 'Master', subtitle: 'Strongest practical mobile mode', badge: 'X', engine: 'hard' },
];

function ModePill({
  title,
  subtitle,
  active,
  onPress,
}: {
  title: string;
  subtitle: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.modePill, active && styles.modePillActive]}>
      <Text style={[styles.modeTitle, active && styles.modeTitleActive]}>{title}</Text>
      <Text style={[styles.modeSubtitle, active && styles.modeSubtitleActive]}>{subtitle}</Text>
    </Pressable>
  );
}

function DifficultyCard({
  item,
  active,
  onPress,
}: {
  item: (typeof DIFFICULTIES)[number];
  active: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.diffCard, active && styles.diffCardActive]}>
      <View style={[styles.diffThumb, active && styles.diffThumbActive]}>
        <View style={styles.diffThumbInner}>
          <Text style={[styles.diffBadge, active && styles.diffBadgeActive]}>{item.badge}</Text>
        </View>
      </View>

      <View style={styles.diffTextBlock}>
        <Text style={[styles.diffTitle, active && styles.diffTitleActive]}>{item.title}</Text>
        <Text style={styles.diffSubtitle}>{item.subtitle}</Text>
      </View>

      <View style={[styles.diffCheck, active && styles.diffCheckActive]}>
        <Text style={[styles.diffCheckText, active && styles.diffCheckTextActive]}>{active ? 'OK' : ''}</Text>
      </View>
    </Pressable>
  );
}

function UtilityChip({ title }: { title: string }) {
  return (
    <View style={styles.utilityChip}>
      <Text style={styles.utilityChipText}>{title}</Text>
    </View>
  );
}

export default function HomeScreen({ onStart, onArena }: Props) {
  const [mode, setMode] = useState<GameMode>('vs-ai');
  const [uiDifficulty, setUiDifficulty] = useState<UiDifficulty>('medium');

  const selectedDifficulty = DIFFICULTIES.find(item => item.id === uiDifficulty) ?? DIFFICULTIES[2];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        bounces={false}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroCard}>
          <View style={styles.heroTopRow}>
            <View style={styles.heroMarkWrap}>
              <View style={styles.heroMarkOuter}>
                <View style={styles.heroMarkInner}>
                  <Text style={styles.heroMarkText}>MK</Text>
                </View>
              </View>
            </View>

            <View style={styles.heroUtilityCol}>
              <UtilityChip title="Shop" />
              <UtilityChip title="Settings" />
            </View>
          </View>

          <View style={styles.heroTextBlock}>
            <Text style={styles.kicker}>THAI CHECKERS</Text>
            <Text style={styles.title}>Makhos</Text>
            <Text style={styles.subtitle}>
              Skeleton home screen for the new 5-level game flow.
            </Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Mode</Text>
          <View style={styles.modeRow}>
            <ModePill
              title="VS AI"
              subtitle="Solo challenge"
              active={mode === 'vs-ai'}
              onPress={() => setMode('vs-ai')}
            />
            <ModePill
              title="VS HUMAN"
              subtitle="Local board play"
              active={mode === 'vs-human'}
              onPress={() => setMode('vs-human')}
            />
          </View>
        </View>

        {mode === 'vs-ai' ? (
          <View style={styles.section}>
            <View style={styles.sectionHeaderRow}>
              <Text style={styles.sectionLabel}>Difficulty</Text>
              <Text style={styles.sectionMeta}>5 visual levels mapped to 3 engine tiers for now</Text>
            </View>

            <View style={styles.diffList}>
              {DIFFICULTIES.map(item => (
                <DifficultyCard
                  key={item.id}
                  item={item}
                  active={uiDifficulty === item.id}
                  onPress={() => setUiDifficulty(item.id)}
                />
              ))}
            </View>
          </View>
        ) : (
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Local Match</Text>
            <View style={styles.localCard}>
              <View style={styles.localArt} />
              <View style={styles.localTextBlock}>
                <Text style={styles.localTitle}>Pass-and-play skeleton</Text>
                <Text style={styles.localSubtitle}>
                  Same device, no AI, clean setup for two players.
                </Text>
              </View>
            </View>
          </View>
        )}

        <View style={styles.bottomBlock}>
          <Pressable
            style={styles.startBtn}
            onPress={() => onStart({ mode, difficulty: selectedDifficulty.engine, humanSide: 1 })}
          >
            <Text style={styles.startText}>
              {mode === 'vs-ai' ? `Start ${selectedDifficulty.title}` : 'Start Local Match'}
            </Text>
          </Pressable>

          <Pressable style={styles.secondaryBtn} onPress={onArena}>
            <Text style={styles.secondaryBtnText}>Open Arena / Labs</Text>
          </Pressable>

          <Text style={styles.footnote}>
            Visual note: image slots are placeholders only. Final art can replace every thumbnail block later.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: SURFACE,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 24,
    gap: 24,
  },
  heroCard: {
    backgroundColor: DARK,
    borderRadius: 28,
    padding: 20,
    gap: 20,
  },
  heroTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  heroMarkWrap: {
    width: 96,
    height: 96,
  },
  heroMarkOuter: {
    width: 88,
    height: 88,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  heroMarkInner: {
    width: 64,
    height: 64,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: ACCENT,
  },
  heroMarkText: {
    color: '#fffaf2',
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  heroUtilityCol: {
    alignItems: 'flex-end',
    gap: 10,
  },
  utilityChip: {
    minWidth: 88,
    height: 40,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.14)',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  utilityChipText: {
    color: '#f9eadf',
    fontSize: 13,
    fontWeight: '700',
  },
  heroTextBlock: {
    gap: 8,
  },
  kicker: {
    color: '#f0d1c0',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.8,
  },
  title: {
    color: '#fffaf2',
    fontSize: 40,
    fontWeight: '800',
    lineHeight: 44,
  },
  subtitle: {
    color: '#dacbc2',
    fontSize: 15,
    lineHeight: 22,
    maxWidth: 280,
  },
  section: {
    gap: 12,
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    gap: 12,
  },
  sectionLabel: {
    color: MUTED,
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.4,
    textTransform: 'uppercase',
  },
  sectionMeta: {
    color: MUTED,
    fontSize: 11,
    lineHeight: 16,
    flex: 1,
    textAlign: 'right',
  },
  modeRow: {
    flexDirection: 'row',
    gap: 10,
  },
  modePill: {
    flex: 1,
    minHeight: 92,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 14,
    backgroundColor: CARD,
    borderWidth: 1,
    borderColor: LINE,
    gap: 6,
  },
  modePillActive: {
    borderColor: ACCENT,
    backgroundColor: ACCENT_SOFT,
  },
  modeTitle: {
    color: INK,
    fontSize: 16,
    fontWeight: '800',
  },
  modeTitleActive: {
    color: '#8b3e18',
  },
  modeSubtitle: {
    color: MUTED,
    fontSize: 13,
    lineHeight: 18,
  },
  modeSubtitleActive: {
    color: '#9b5d36',
  },
  diffList: {
    gap: 12,
  },
  diffCard: {
    minHeight: 92,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: CARD,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  diffCardActive: {
    borderColor: ACCENT,
    backgroundColor: '#fff1e6',
    shadowColor: '#8b3e18',
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 2,
  },
  diffThumb: {
    width: 64,
    height: 64,
    borderRadius: 20,
    backgroundColor: '#efe4d8',
    alignItems: 'center',
    justifyContent: 'center',
  },
  diffThumbActive: {
    backgroundColor: '#f1cdb6',
  },
  diffThumbInner: {
    width: 52,
    height: 52,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(58,42,34,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.55)',
  },
  diffBadge: {
    color: MUTED,
    fontSize: 20,
    fontWeight: '800',
  },
  diffBadgeActive: {
    color: '#8b3e18',
  },
  diffTextBlock: {
    flex: 1,
    gap: 4,
  },
  diffTitle: {
    color: INK,
    fontSize: 17,
    fontWeight: '800',
  },
  diffTitleActive: {
    color: '#8b3e18',
  },
  diffSubtitle: {
    color: MUTED,
    fontSize: 13,
    lineHeight: 18,
  },
  diffCheck: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: LINE,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
  diffCheckActive: {
    borderColor: ACCENT,
    backgroundColor: ACCENT,
  },
  diffCheckText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '800',
  },
  diffCheckTextActive: {
    color: '#fff',
  },
  localCard: {
    minHeight: 112,
    borderRadius: 24,
    backgroundColor: CARD,
    borderWidth: 1,
    borderColor: LINE,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  localArt: {
    width: 72,
    height: 72,
    borderRadius: 24,
    backgroundColor: '#eadccf',
  },
  localTextBlock: {
    flex: 1,
    gap: 4,
  },
  localTitle: {
    color: INK,
    fontSize: 16,
    fontWeight: '800',
  },
  localSubtitle: {
    color: MUTED,
    fontSize: 13,
    lineHeight: 18,
  },
  bottomBlock: {
    gap: 12,
    paddingTop: 4,
  },
  startBtn: {
    height: 56,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: ACCENT,
  },
  startText: {
    color: '#fffaf2',
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  secondaryBtn: {
    height: 48,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: CARD,
  },
  secondaryBtnText: {
    color: MUTED,
    fontSize: 14,
    fontWeight: '700',
  },
  footnote: {
    color: MUTED,
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
    paddingHorizontal: 10,
  },
});
