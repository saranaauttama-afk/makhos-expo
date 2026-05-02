import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { GameConfig, MonetizationState } from './types';

interface Props {
  language: AppLanguage;
  monetization: MonetizationState;
  companyName: string;
  appVersion: string;
  defaultSetupConfig: GameConfig;
  onQuickPlay: (config: GameConfig) => void;
  onStart: (config: GameConfig) => void;
  onAccount: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

const COPY = {
  th: {
    title: 'หมากฮอสไทย',
    play: 'เล่นทันที',
    setup: 'ตั้งค่าการเล่น',
    account: 'บัญชี',
  },
  en: {
    title: 'MAKHOS (Thai Checker)',
    play: 'PLAY NOW',
    setup: 'CUSTOM MATCH',
    account: 'ACCOUNT',
  },
} as const;

function MenuButton({
  title,
  onPress,
  tint = SOFT,
  primary = false,
}: {
  title: string;
  onPress: () => void;
  tint?: string;
  primary?: boolean;
}) {
  return (
    <Pressable
      style={[
        styles.menuButton,
        { borderColor: tint },
        primary && styles.menuButtonPrimary,
      ]}
      onPress={onPress}
    >
      <Text style={[styles.menuButtonText, { color: tint }, primary && styles.menuButtonTextPrimary]}>{title}</Text>
    </Pressable>
  );
}

export default function HomeScreen({
  language,
  monetization,
  companyName,
  appVersion,
  defaultSetupConfig,
  onQuickPlay,
  onStart,
  onAccount,
}: Props) {
  const t = COPY[language];
  const openSetup = () => onStart(defaultSetupConfig);
  const quickPlay = () => onQuickPlay({ mode: 'vs-ai', difficulty: 'easy', humanSide: 1, unlimitedThink: false });

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.edgeSpacer} />

        <View style={styles.headerPanel}>
          <Text style={styles.title}>{t.title}</Text>
        </View>

        <View style={styles.menuGroup}>
          <MenuButton title={t.play} onPress={quickPlay} tint={BG} primary />
          <MenuButton title={t.setup} onPress={openSetup} tint={CYAN} />
          <MenuButton title={t.account} onPress={onAccount} />
        </View>

        <View style={styles.edgeSpacer} />

        <View style={styles.metaBlock}>
          <Text style={styles.metaText}>{companyName}</Text>
          <Text style={styles.metaText}>Version {appVersion}</Text>
        </View>
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
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 18,
    paddingTop: 12,
    paddingBottom: 24,
    gap: 10,
  },
  edgeSpacer: {
    flexGrow: 1,
    minHeight: 0,
  },
  headerPanel: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 2,
  },
  title: {
    color: WHITE,
    fontSize: 26,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1.2,
  },
  menuGroup: {
    gap: 10,
  },
  menuButton: {
    minHeight: 46,
    backgroundColor: PANEL_DARK,
    borderWidth: 1,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 12,
    shadowColor: '#000',
    shadowOpacity: 0.16,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  menuButtonPrimary: {
    minHeight: 56,
    backgroundColor: GOLD,
    borderColor: GOLD,
    shadowOpacity: 0.24,
    shadowRadius: 10,
    elevation: 6,
  },
  menuButtonText: {
    fontSize: 13,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.8,
  },
  menuButtonTextPrimary: {
    color: '#264742',
    fontSize: 15,
    letterSpacing: 1,
  },
  metaBlock: {
    width: '100%',
    alignItems: 'center',
    gap: 2,
    paddingVertical: 4,
  },
  metaText: {
    color: SOFT,
    fontSize: 10,
    opacity: 0.86,
    fontFamily: 'Kanit_500Medium',
  },
});

