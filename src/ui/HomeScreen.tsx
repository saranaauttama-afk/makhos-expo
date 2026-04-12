import React, { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Difficulty, GameConfig, GameMode } from './types';

interface Props {
  onStart: (config: GameConfig) => void;
  onArena: () => void;
  onConcept: () => void;
  onShop: () => void;
  onSettings: () => void;
  onResultPreview: () => void;
}

type UiDifficulty = 'beginner' | 'easy' | 'medium' | 'hard' | 'master';

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_ALT = '#2c1f49';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const PIXEL_GOLD = '#f3c969';
const PIXEL_MINT = '#77f7cf';
const PIXEL_CYAN = '#5ec5ff';
const PIXEL_PINK = '#ff7dc4';
const PIXEL_RED = '#ff6b6b';
const PIXEL_WHITE = '#f7f2ff';
const PIXEL_SOFT = '#b9abd8';
const PIXEL_SHADOW = '#09050f';

const DIFFICULTIES: Array<{
  id: UiDifficulty;
  title: string;
  tag: string;
  subtitle: string;
  palette: string;
  engine: Difficulty;
}> = [
  { id: 'beginner', title: 'BEGINNER', tag: 'CASUAL', subtitle: 'Soft pressure and extra breathing room.', palette: PIXEL_MINT, engine: 'easy' },
  { id: 'easy', title: 'EASY', tag: 'RETRO', subtitle: 'Friendly matches for quick sessions.', palette: PIXEL_CYAN, engine: 'easy' },
  { id: 'medium', title: 'MEDIUM', tag: 'ARCADE', subtitle: 'Balanced ladder for normal play.', palette: PIXEL_GOLD, engine: 'medium' },
  { id: 'hard', title: 'HARD', tag: 'TACTIC', subtitle: 'Sharper forcing lines and less mercy.', palette: PIXEL_PINK, engine: 'hard' },
  { id: 'master', title: 'MASTER', tag: 'BOSS', subtitle: 'Strongest mobile tier before premium AI.', palette: PIXEL_RED, engine: 'hard' },
];

function PixelBadge({ label, tint }: { label: string; tint: string }) {
  return (
    <View style={[styles.pixelBadge, { borderColor: tint, backgroundColor: PANEL_DARK }]}>
      <View style={[styles.pixelBadgeDot, { backgroundColor: tint }]} />
      <Text style={styles.pixelBadgeText}>{label}</Text>
    </View>
  );
}

function PixelPreview({ tint, label }: { tint: string; label: string }) {
  return (
    <View style={styles.previewShell}>
      <View style={[styles.previewFrame, { borderColor: tint }]}>
        <View style={[styles.previewBlockLg, { backgroundColor: tint }]} />
        <View style={[styles.previewBlockMd, { backgroundColor: PIXEL_WHITE }]} />
        <View style={[styles.previewBlockSm, { backgroundColor: tint }]} />
      </View>
      <Text style={styles.previewLabel}>{label}</Text>
    </View>
  );
}

function ModeButton({
  title,
  caption,
  active,
  tint,
  onPress,
}: {
  title: string;
  caption: string;
  active: boolean;
  tint: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.modeButton, active && { borderColor: tint, backgroundColor: PANEL_ALT }]}>
      <Text style={[styles.modeButtonTitle, active && { color: tint }]}>{title}</Text>
      <Text style={styles.modeButtonCaption}>{caption}</Text>
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
    <Pressable onPress={onPress} style={[styles.diffCard, active && { borderColor: item.palette, backgroundColor: PANEL_ALT }]}>
      <PixelPreview tint={item.palette} label={item.tag} />

      <View style={styles.diffTextBlock}>
        <View style={styles.diffTopRow}>
          <Text style={[styles.diffTitle, active && { color: item.palette }]}>{item.title}</Text>
          <PixelBadge label={item.tag} tint={item.palette} />
        </View>
        <Text style={styles.diffSubtitle}>{item.subtitle}</Text>
      </View>

      <View style={[styles.diffCursor, active && { borderColor: item.palette, backgroundColor: item.palette }]}>
        <Text style={[styles.diffCursorText, active && styles.diffCursorTextActive]}>{active ? 'GO' : '...'}</Text>
      </View>
    </Pressable>
  );
}

function UtilityButton({ title, onPress }: { title: string; onPress?: () => void }) {
  return (
    <Pressable disabled={!onPress} onPress={onPress} style={styles.utilityButton}>
      <Text style={styles.utilityButtonText}>{title}</Text>
    </Pressable>
  );
}

export default function HomeScreen({ onStart, onArena, onConcept, onShop, onSettings, onResultPreview }: Props) {
  const [mode, setMode] = useState<GameMode>('vs-ai');
  const [uiDifficulty, setUiDifficulty] = useState<UiDifficulty>('medium');

  const selectedDifficulty = useMemo(
    () => DIFFICULTIES.find(item => item.id === uiDifficulty) ?? DIFFICULTIES[2],
    [uiDifficulty],
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} bounces={false}>
        <View style={styles.heroPanel}>
          <View style={styles.utilityRow}>
            <UtilityButton title="SHOP" onPress={onShop} />
            <UtilityButton title="SETTINGS" onPress={onSettings} />
          </View>

          <View style={styles.heroBody}>
            <View style={styles.heroMark}>
              <View style={[styles.heroPixel, styles.heroPixelA]} />
              <View style={[styles.heroPixel, styles.heroPixelB]} />
              <View style={[styles.heroPixel, styles.heroPixelC]} />
              <View style={[styles.heroPixel, styles.heroPixelD]} />
            </View>

            <View style={styles.heroTextBlock}>
              <Text style={styles.heroKicker}>THAI CHECKERS 198X EDITION</Text>
              <Text style={styles.heroTitle}>MAKHOS</Text>
              <Text style={styles.heroSubtitle}>
                Pixel-retro shell for the mobile build. Final art can replace every preview tile later.
              </Text>
            </View>
          </View>

          <View style={styles.heroMetaRow}>
            <PixelBadge label="ADS READY" tint={PIXEL_GOLD} />
            <PixelBadge label="PREMIUM AI" tint={PIXEL_PINK} />
            <PixelBadge label="LOCAL PLAY" tint={PIXEL_MINT} />
          </View>
        </View>

        <View style={styles.sectionPanel}>
          <Text style={styles.sectionTitle}>MODE SELECT</Text>
          <View style={styles.modeRow}>
            <ModeButton
              title="VS AI"
              caption="Solo ladder and premium battle modes."
              active={mode === 'vs-ai'}
              tint={PIXEL_GOLD}
              onPress={() => setMode('vs-ai')}
            />
            <ModeButton
              title="VS HUMAN"
              caption="Local pass-and-play on one device."
              active={mode === 'vs-human'}
              tint={PIXEL_MINT}
              onPress={() => setMode('vs-human')}
            />
          </View>
        </View>

        <View style={styles.sectionPanel}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>{mode === 'vs-ai' ? 'DIFFICULTY LADDER' : 'LOCAL MATCH'}</Text>
            <Text style={styles.sectionMeta}>{mode === 'vs-ai' ? '5 VISUAL LEVELS / 3 ENGINE TIERS' : 'NO ADS MID-GAME'}</Text>
          </View>

          {mode === 'vs-ai' ? (
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
          ) : (
            <View style={styles.localPanel}>
              <PixelPreview tint={PIXEL_MINT} label="P1 VS P2" />
              <View style={styles.localTextBlock}>
                <Text style={styles.localTitle}>LOCAL ARCADE BOARD</Text>
                <Text style={styles.localSubtitle}>
                  One phone, two players, same ruleset. Great for quick tabletop-style sessions.
                </Text>
              </View>
            </View>
          )}
        </View>

        <Pressable style={styles.conceptPanel} onPress={onConcept}>
          <View style={styles.conceptLeft}>
            <Text style={styles.conceptTitle}>STUDY THE CONCEPT</Text>
            <Text style={styles.conceptCopy}>
              Read how Makhos mixes classic board-play, AI tiers, ads, and premium unlocks in one product flow.
            </Text>
          </View>
          <View style={styles.conceptRight}>
            <Text style={styles.conceptRightText}>OPEN</Text>
          </View>
        </Pressable>

        <View style={styles.ctaPanel}>
          <Pressable
            style={[styles.primaryButton, { borderColor: selectedDifficulty.palette }]}
            onPress={() => onStart({ mode, difficulty: selectedDifficulty.engine, humanSide: 1 })}
          >
            <Text style={styles.primaryButtonText}>
              {mode === 'vs-ai' ? `OPEN ${selectedDifficulty.title} SETUP` : 'OPEN LOCAL SETUP'}
            </Text>
          </Pressable>

          <View style={styles.footerGrid}>
            <Pressable style={styles.secondaryButton} onPress={onArena}>
              <Text style={styles.secondaryButtonText}>OPEN ARENA / LABS</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={onResultPreview}>
              <Text style={styles.secondaryButtonText}>RESULT PREVIEW</Text>
            </Pressable>
          </View>
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
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 18,
    gap: 12,
  },
  heroPanel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 12,
    gap: 10,
    shadowColor: PIXEL_SHADOW,
    shadowOpacity: 0.45,
    shadowRadius: 0,
    shadowOffset: { width: 6, height: 6 },
    elevation: 6,
  },
  utilityRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  utilityButton: {
    minWidth: 92,
    paddingHorizontal: 10,
    paddingVertical: 8,
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    alignItems: 'center',
  },
  utilityButtonText: {
    color: PIXEL_WHITE,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  heroBody: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
  },
  heroMark: {
    width: 84,
    height: 84,
    borderWidth: 3,
    borderColor: PIXEL_GOLD,
    backgroundColor: PANEL_DARK,
    position: 'relative',
  },
  heroPixel: {
    position: 'absolute',
    width: 18,
    height: 18,
  },
  heroPixelA: {
    left: 8,
    top: 8,
    backgroundColor: PIXEL_GOLD,
  },
  heroPixelB: {
    right: 8,
    top: 18,
    backgroundColor: PIXEL_PINK,
  },
  heroPixelC: {
    left: 18,
    bottom: 8,
    backgroundColor: PIXEL_CYAN,
  },
  heroPixelD: {
    right: 14,
    bottom: 14,
    backgroundColor: PIXEL_MINT,
  },
  heroTextBlock: {
    flex: 1,
    gap: 4,
  },
  heroKicker: {
    color: PIXEL_GOLD,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.4,
  },
  heroTitle: {
    color: PIXEL_WHITE,
    fontSize: 28,
    fontWeight: '900',
    letterSpacing: 1.8,
  },
  heroSubtitle: {
    color: PIXEL_SOFT,
    fontSize: 12,
    lineHeight: 17,
  },
  heroMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  pixelBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    borderWidth: 2,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  pixelBadgeDot: {
    width: 10,
    height: 10,
  },
  pixelBadgeText: {
    color: PIXEL_WHITE,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  sectionPanel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 14,
    gap: 10,
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
    alignItems: 'center',
  },
  sectionTitle: {
    color: PIXEL_WHITE,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1.3,
  },
  sectionMeta: {
    flex: 1,
    textAlign: 'right',
    color: PIXEL_SOFT,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  modeRow: {
    flexDirection: 'row',
    gap: 8,
  },
  modeButton: {
    flex: 1,
    minHeight: 76,
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    padding: 10,
    gap: 6,
  },
  modeButtonTitle: {
    color: PIXEL_WHITE,
    fontSize: 15,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  modeButtonCaption: {
    color: PIXEL_SOFT,
    fontSize: 11,
    lineHeight: 16,
  },
  diffList: {
    gap: 8,
  },
  diffCard: {
    minHeight: 88,
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  previewShell: {
    width: 62,
    gap: 4,
    alignItems: 'center',
  },
  previewFrame: {
    width: 62,
    height: 62,
    borderWidth: 2,
    backgroundColor: '#170f26',
    position: 'relative',
  },
  previewBlockLg: {
    position: 'absolute',
    left: 7,
    top: 7,
    width: 22,
    height: 22,
  },
  previewBlockMd: {
    position: 'absolute',
    right: 8,
    top: 18,
    width: 14,
    height: 14,
  },
  previewBlockSm: {
    position: 'absolute',
    left: 20,
    bottom: 8,
    width: 18,
    height: 18,
  },
  previewLabel: {
    color: PIXEL_SOFT,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  diffTextBlock: {
    flex: 1,
    gap: 4,
  },
  diffTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 8,
  },
  diffTitle: {
    color: PIXEL_WHITE,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1,
  },
  diffSubtitle: {
    color: PIXEL_SOFT,
    fontSize: 11,
    lineHeight: 16,
  },
  diffCursor: {
    width: 36,
    height: 36,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL,
    alignItems: 'center',
    justifyContent: 'center',
  },
  diffCursorText: {
    color: PIXEL_SOFT,
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.7,
  },
  diffCursorTextActive: {
    color: PANEL_DARK,
  },
  localPanel: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    padding: 12,
  },
  localTextBlock: {
    flex: 1,
    gap: 6,
  },
  localTitle: {
    color: PIXEL_MINT,
    fontSize: 16,
    fontWeight: '900',
    letterSpacing: 0.9,
  },
  localSubtitle: {
    color: PIXEL_SOFT,
    fontSize: 12,
    lineHeight: 18,
  },
  conceptPanel: {
    backgroundColor: PANEL_ALT,
    borderWidth: 3,
    borderColor: PIXEL_CYAN,
    padding: 12,
    flexDirection: 'row',
    gap: 12,
    alignItems: 'center',
  },
  conceptLeft: {
    flex: 1,
    gap: 4,
  },
  conceptTitle: {
    color: PIXEL_CYAN,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  conceptCopy: {
    color: PIXEL_WHITE,
    fontSize: 11,
    lineHeight: 16,
  },
  conceptRight: {
    minWidth: 64,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderWidth: 2,
    borderColor: PIXEL_CYAN,
    backgroundColor: PANEL_DARK,
    alignItems: 'center',
  },
  conceptRightText: {
    color: PIXEL_CYAN,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1,
  },
  ctaPanel: {
    gap: 8,
  },
  primaryButton: {
    minHeight: 52,
    backgroundColor: PANEL_ALT,
    borderWidth: 3,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  primaryButtonText: {
    color: PIXEL_WHITE,
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 1.1,
    textAlign: 'center',
  },
  secondaryButton: {
    flex: 1,
    minHeight: 46,
    backgroundColor: PANEL,
    borderWidth: 2,
    borderColor: LINE,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  secondaryButtonText: {
    color: PIXEL_SOFT,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.9,
  },
  footerGrid: {
    flexDirection: 'row',
    gap: 8,
  },
});
