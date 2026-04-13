import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { GameConfig } from './types';

interface Props {
  language: AppLanguage;
  onStart: (config: GameConfig) => void;
  onArena: () => void;
  onConcept: () => void;
  onShop: () => void;
  onSettings: () => void;
  onResultPreview: () => void;
}

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const CYAN = '#5ec5ff';
const MINT = '#77f7cf';
const PINK = '#ff7dc4';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

const COPY = {
  th: {
    kicker: 'MAIN MENU',
    title: 'MAKHOS',
    subtitle: 'choose mode and enter each screen from one clean menu.',
    quickLabel: 'PLAY',
    quickHint: 'open setup with medium preset',
    arenaHint: 'engine lab and sandbox battle screen',
    conceptHint: 'study game concept and learning flow',
    shopHint: 'premium unlock and ad-remove path',
    settingsHint: 'language, sound, and controls',
    resultHint: 'post-match result screen preview',
    arena: 'ARENA / LABS',
    concept: 'CONCEPT',
    shop: 'SHOP',
    settings: 'SETTINGS',
    result: 'RESULT PREVIEW',
    footer: 'pixel-retro code-only shell',
  },
  en: {
    kicker: 'MAIN MENU',
    title: 'MAKHOS',
    subtitle: 'choose mode and enter each screen from one clean menu.',
    quickLabel: 'PLAY',
    quickHint: 'open setup with medium preset',
    arenaHint: 'engine lab and sandbox battle screen',
    conceptHint: 'study game concept and learning flow',
    shopHint: 'premium unlock and ad-remove path',
    settingsHint: 'language, sound, and controls',
    resultHint: 'post-match result screen preview',
    arena: 'ARENA / LABS',
    concept: 'CONCEPT',
    shop: 'SHOP',
    settings: 'SETTINGS',
    result: 'RESULT PREVIEW',
    footer: 'pixel-retro code-only shell',
  },
} as const;

function PixelScene() {
  return (
    <View style={styles.sceneWrap}>
      <View style={styles.sceneGround} />

      <View style={styles.sceneMachineFrameA} />
      <View style={styles.sceneMachineFrameB} />
      <View style={styles.sceneMachineFrameC} />

      <View style={styles.sceneStackWrap}>
        <View style={styles.sceneStackLogA} />
        <View style={styles.sceneStackLogB} />
        <View style={styles.sceneStackLogC} />
      </View>

      <View style={styles.sceneTorchBase} />
      <View style={styles.sceneTorchGlow} />
    </View>
  );
}

function MenuButton({
  title,
  subtitle,
  tint,
  onPress,
}: {
  title: string;
  subtitle: string;
  tint: string;
  onPress: () => void;
}) {
  return (
    <Pressable style={[styles.menuButton, { borderColor: tint }]} onPress={onPress}>
      <View style={styles.menuTopRow}>
        <View style={[styles.menuIcon, { borderColor: tint }]}>
          <View style={[styles.menuIconPixel, { backgroundColor: tint }]} />
          <View style={[styles.menuIconPixel, { backgroundColor: tint }]} />
          <View style={[styles.menuIconPixel, { backgroundColor: tint }]} />
          <View style={[styles.menuIconPixel, { backgroundColor: tint }]} />
        </View>
        <Text style={[styles.menuTitle, { color: tint }]}>{title}</Text>
        <View style={[styles.menuEnterBadge, { borderColor: tint }]}>
          <Text style={[styles.menuEnterText, { color: tint }]}>ENTER</Text>
        </View>
      </View>
      <Text style={styles.menuSubtitle}>{subtitle}</Text>
    </Pressable>
  );
}

function InfoChip({ label, tint }: { label: string; tint: string }) {
  return (
    <View style={[styles.infoChip, { borderColor: tint }]}>
      <Text style={[styles.infoChipText, { color: tint }]}>{label}</Text>
    </View>
  );
}

export default function HomeScreen({ language, onStart, onArena, onConcept, onShop, onSettings, onResultPreview }: Props) {
  const t = COPY[language];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.heroPanel}>
          <View style={styles.heroTopRow}>
            <View style={styles.heroCopy}>
              <Text style={styles.kicker}>{t.kicker}</Text>
              <Text style={styles.title}>{t.title}</Text>
              <Text style={styles.subtitle}>{t.subtitle}</Text>
            </View>
            <PixelScene />
          </View>
          <View style={styles.heroChipRow}>
            <InfoChip label="LOCAL BOARD" tint={MINT} />
            <InfoChip label="AI LADDER" tint={GOLD} />
            <InfoChip label="PREMIUM" tint={PINK} />
          </View>
        </View>

        <MenuButton
          title={t.quickLabel}
          subtitle={t.quickHint}
          tint={GOLD}
          onPress={() => onStart({ mode: 'vs-ai', difficulty: 'medium', humanSide: 1 })}
        />

        <View style={styles.dualRow}>
          <View style={styles.dualCol}>
            <MenuButton title={t.concept} subtitle={t.conceptHint} tint={PINK} onPress={onConcept} />
          </View>
          <View style={styles.dualCol}>
            <MenuButton title={t.shop} subtitle={t.shopHint} tint={GOLD} onPress={onShop} />
          </View>
        </View>

        <View style={styles.dualRow}>
          <View style={styles.dualCol}>
            <MenuButton title={t.settings} subtitle={t.settingsHint} tint={CYAN} onPress={onSettings} />
          </View>
          <View style={styles.dualCol}>
            <MenuButton title={t.result} subtitle={t.resultHint} tint={MINT} onPress={onResultPreview} />
          </View>
        </View>

        <MenuButton title={t.arena} subtitle={t.arenaHint} tint={CYAN} onPress={onArena} />

        <View style={styles.footerPanel}>
          <Text style={styles.footerText}>{t.footer}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: BG },
  scrollContent: { paddingHorizontal: 14, paddingTop: 10, paddingBottom: 20, gap: 10 },
  heroPanel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 10,
    gap: 8,
    shadowColor: '#000000',
    shadowOpacity: 0.36,
    shadowOffset: { width: 4, height: 4 },
    shadowRadius: 0,
    elevation: 4,
  },
  heroTopRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
    gap: 8,
  },
  heroCopy: {
    flex: 1,
    gap: 6,
  },
  kicker: {
    color: GOLD,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  title: {
    color: WHITE,
    fontSize: 28,
    fontWeight: '900',
    letterSpacing: 1.6,
    lineHeight: 30,
  },
  subtitle: {
    color: SOFT,
    fontSize: 12,
    lineHeight: 16,
  },
  heroChipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  infoChip: {
    borderWidth: 2,
    backgroundColor: PANEL_DARK,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  infoChipText: {
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  sceneWrap: {
    width: 118,
    borderWidth: 2,
    borderColor: '#7a6a51',
    backgroundColor: '#4b5941',
    position: 'relative',
    overflow: 'hidden',
  },
  sceneGround: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 34,
    backgroundColor: '#7a694e',
  },
  sceneMachineFrameA: {
    position: 'absolute',
    left: 46,
    bottom: 26,
    width: 30,
    height: 48,
    backgroundColor: '#7d5432',
    borderWidth: 2,
    borderColor: '#533722',
  },
  sceneMachineFrameB: {
    position: 'absolute',
    left: 38,
    bottom: 24,
    width: 12,
    height: 44,
    backgroundColor: '#a9784d',
    borderWidth: 2,
    borderColor: '#533722',
  },
  sceneMachineFrameC: {
    position: 'absolute',
    left: 74,
    bottom: 24,
    width: 12,
    height: 44,
    backgroundColor: '#a9784d',
    borderWidth: 2,
    borderColor: '#533722',
  },
  sceneStackWrap: {
    position: 'absolute',
    left: 10,
    bottom: 18,
    width: 32,
    gap: 2,
  },
  sceneStackLogA: {
    height: 8,
    backgroundColor: '#926740',
    borderWidth: 1,
    borderColor: '#533722',
  },
  sceneStackLogB: {
    height: 8,
    backgroundColor: '#a8794f',
    borderWidth: 1,
    borderColor: '#533722',
  },
  sceneStackLogC: {
    height: 8,
    backgroundColor: '#845839',
    borderWidth: 1,
    borderColor: '#533722',
  },
  sceneTorchBase: {
    position: 'absolute',
    right: 8,
    bottom: 8,
    width: 20,
    height: 14,
    backgroundColor: '#34302b',
    borderWidth: 2,
    borderColor: '#171513',
  },
  sceneTorchGlow: {
    position: 'absolute',
    right: 11,
    bottom: 11,
    width: 12,
    height: 8,
    backgroundColor: '#ff8641',
  },
  menuButton: {
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    minHeight: 68,
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 5,
  },
  menuTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
  },
  menuIcon: {
    width: 18,
    height: 18,
    borderWidth: 2,
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 2,
    gap: 1,
  },
  menuIconPixel: {
    width: 4,
    height: 4,
  },
  menuTitle: {
    flex: 1,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1,
  },
  menuEnterBadge: {
    minWidth: 46,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
    paddingVertical: 3,
  },
  menuEnterText: {
    fontSize: 8,
    fontWeight: '900',
    letterSpacing: 0.6,
  },
  menuSubtitle: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 15,
  },
  dualRow: {
    flexDirection: 'row',
    gap: 8,
  },
  dualCol: {
    flex: 1,
  },
  footerPanel: {
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  footerText: {
    color: SOFT,
    fontSize: 10,
    letterSpacing: 0.7,
    textAlign: 'center',
  },
});
