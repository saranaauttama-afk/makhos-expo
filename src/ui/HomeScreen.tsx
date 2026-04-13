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
    subtitle: 'เลือกเมนูเพื่อเข้าไปทีละส่วนของเกม',
    quickLabel: 'PLAY',
    quickHint: 'เข้า Setup เพื่อเลือกโหมดและความยาก',
    arenaHint: 'โหมดทดลองเอนจินและบอร์ดทดสอบ',
    conceptHint: 'แนวคิดเกม + AI + โมเดลรายได้',
    shopHint: 'ปลดล็อกพรีเมียมและปิดโฆษณา',
    settingsHint: 'ภาษา เสียง และค่าระบบ',
    resultHint: 'พรีวิวหน้าผลหลังจบเกม',
    arena: 'ARENA / LABS',
    concept: 'CONCEPT',
    shop: 'SHOP',
    settings: 'SETTINGS',
    result: 'RESULT PREVIEW',
    footer: 'Pixel-retro skeleton UI',
  },
  en: {
    kicker: 'MAIN MENU',
    title: 'MAKHOS',
    subtitle: 'Choose a menu button to open each section.',
    quickLabel: 'PLAY',
    quickHint: 'Open Setup and choose mode + difficulty',
    arenaHint: 'Engine experiments and board test tools.',
    conceptHint: 'Game + monetization + AI concept notes.',
    shopHint: 'Premium unlock and ad removal flow.',
    settingsHint: 'Language, sound, and preference controls.',
    resultHint: 'UI preview for post-match result.',
    arena: 'ARENA / LABS',
    concept: 'CONCEPT',
    shop: 'SHOP',
    settings: 'SETTINGS',
    result: 'RESULT PREVIEW',
    footer: 'Pixel-retro skeleton UI',
  },
} as const;

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
      <Text style={[styles.menuTitle, { color: tint }]}>{title}</Text>
      <Text style={styles.menuSubtitle}>{subtitle}</Text>
    </Pressable>
  );
}

export default function HomeScreen({ language, onStart, onArena, onConcept, onShop, onSettings, onResultPreview }: Props) {
  const t = COPY[language];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.heroPanel}>
          <Text style={styles.kicker}>{t.kicker}</Text>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <MenuButton
          title={t.quickLabel}
          subtitle={t.quickHint}
          tint={GOLD}
          onPress={() => onStart({ mode: 'vs-ai', difficulty: 'medium', humanSide: 1 })}
        />
        <MenuButton title={t.arena} subtitle={t.arenaHint} tint={MINT} onPress={onArena} />
        <MenuButton title={t.concept} subtitle={t.conceptHint} tint={PINK} onPress={onConcept} />
        <MenuButton title={t.shop} subtitle={t.shopHint} tint={GOLD} onPress={onShop} />
        <MenuButton title={t.settings} subtitle={t.settingsHint} tint={CYAN} onPress={onSettings} />
        <MenuButton title={t.result} subtitle={t.resultHint} tint={MINT} onPress={onResultPreview} />

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
    padding: 12,
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
    letterSpacing: 1.5,
  },
  subtitle: {
    color: SOFT,
    fontSize: 12,
    lineHeight: 17,
  },
  menuButton: {
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    minHeight: 62,
    paddingHorizontal: 12,
    paddingVertical: 10,
    justifyContent: 'center',
    gap: 4,
  },
  menuTitle: {
    fontSize: 15,
    fontWeight: '900',
    letterSpacing: 1.1,
  },
  menuSubtitle: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 15,
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
