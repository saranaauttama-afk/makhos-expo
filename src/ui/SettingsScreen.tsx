import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';

interface Props {
  language: AppLanguage;
  onLanguageChange: (language: AppLanguage) => void;
  onBack: () => void;
}

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const CYAN = '#5ec5ff';
const MINT = '#77f7cf';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

const COPY = {
  th: {
    system: 'SYSTEM MENU',
    title: 'SETTINGS',
    subtitle: 'จัดการภาษา เสียง และตัวเลือกระบบ',
    preferences: 'PREFERENCES',
    language: 'Language',
    languageSub: 'เปลี่ยนภาษาแอปแบบทันที',
    sound: 'Sound',
    soundSub: 'เสียงเอฟเฟกต์ตอนเดินและผลเกม',
    vibration: 'Vibration',
    vibrationSub: 'สั่นเบาๆ ตอนยืนยันการเดิน',
    retroFx: 'Retro FX',
    retroFxSub: 'เอฟเฟกต์พิกเซลและสไตล์อาร์เคด',
    account: 'ACCOUNT / PURCHASE',
    restore: 'Restore Purchase',
    premium: 'Manage Premium Access',
    info: 'INFO',
    privacy: 'Privacy Policy',
    terms: 'Terms of Service',
    about: 'About Makhos',
    back: 'BACK',
    open: 'OPEN',
  },
  en: {
    system: 'SYSTEM MENU',
    title: 'SETTINGS',
    subtitle: 'Manage language, sound, and system options.',
    preferences: 'PREFERENCES',
    language: 'Language',
    languageSub: 'Switch app language instantly.',
    sound: 'Sound',
    soundSub: 'Move sounds and match result cues.',
    vibration: 'Vibration',
    vibrationSub: 'Light feedback on move confirm.',
    retroFx: 'Retro FX',
    retroFxSub: 'Pixel overlays and arcade effects.',
    account: 'ACCOUNT / PURCHASE',
    restore: 'Restore Purchase',
    premium: 'Manage Premium Access',
    info: 'INFO',
    privacy: 'Privacy Policy',
    terms: 'Terms of Service',
    about: 'About Makhos',
    back: 'BACK',
    open: 'OPEN',
  },
} as const;

function ToggleRow({
  title,
  subtitle,
  active,
  onToggle,
}: {
  title: string;
  subtitle: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <Pressable onPress={onToggle} style={styles.toggleRow}>
      <View style={styles.toggleTextBlock}>
        <Text style={styles.toggleTitle}>{title}</Text>
        <Text style={styles.toggleSubtitle}>{subtitle}</Text>
      </View>
      <View style={[styles.togglePill, active && styles.togglePillActive]}>
        <View style={[styles.toggleKnob, active && styles.toggleKnobActive]} />
      </View>
    </Pressable>
  );
}

function ActionRow({ title, tint, openText }: { title: string; tint: string; openText: string }) {
  return (
    <Pressable style={styles.actionRow}>
      <Text style={[styles.actionTitle, { color: tint }]}>{title}</Text>
      <Text style={styles.actionArrow}>{openText}</Text>
    </Pressable>
  );
}

function LanguageRow({
  language,
  onChange,
  title,
  subtitle,
}: {
  language: AppLanguage;
  onChange: (language: AppLanguage) => void;
  title: string;
  subtitle: string;
}) {
  return (
    <View style={styles.languageRow}>
      <View style={styles.toggleTextBlock}>
        <Text style={styles.toggleTitle}>{title}</Text>
        <Text style={styles.toggleSubtitle}>{subtitle}</Text>
      </View>
      <View style={styles.langButtons}>
        <Pressable style={[styles.langButton, language === 'th' && styles.langButtonActive]} onPress={() => onChange('th')}>
          <Text style={[styles.langButtonText, language === 'th' && styles.langButtonTextActive]}>TH</Text>
        </Pressable>
        <Pressable style={[styles.langButton, language === 'en' && styles.langButtonActive]} onPress={() => onChange('en')}>
          <Text style={[styles.langButtonText, language === 'en' && styles.langButtonTextActive]}>EN</Text>
        </Pressable>
      </View>
    </View>
  );
}

export default function SettingsScreen({ language, onLanguageChange, onBack }: Props) {
  const [soundOn, setSoundOn] = useState(true);
  const [vibrationOn, setVibrationOn] = useState(true);
  const [retroFx, setRetroFx] = useState(true);
  const t = COPY[language];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>{t.back}</Text>
          </Pressable>
          <Text style={styles.kicker}>{t.system}</Text>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.preferences}</Text>
          <LanguageRow language={language} onChange={onLanguageChange} title={t.language} subtitle={t.languageSub} />
          <ToggleRow title={t.sound} subtitle={t.soundSub} active={soundOn} onToggle={() => setSoundOn(v => !v)} />
          <ToggleRow title={t.vibration} subtitle={t.vibrationSub} active={vibrationOn} onToggle={() => setVibrationOn(v => !v)} />
          <ToggleRow title={t.retroFx} subtitle={t.retroFxSub} active={retroFx} onToggle={() => setRetroFx(v => !v)} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.account}</Text>
          <ActionRow title={t.restore} tint={CYAN} openText={t.open} />
          <ActionRow title={t.premium} tint={GOLD} openText={t.open} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.info}</Text>
          <ActionRow title={t.privacy} tint={MINT} openText={t.open} />
          <ActionRow title={t.terms} tint={CYAN} openText={t.open} />
          <ActionRow title={t.about} tint={GOLD} openText={t.open} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: BG },
  scrollContent: { paddingHorizontal: 14, paddingTop: 10, paddingBottom: 18, gap: 12 },
  headerPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: LINE, padding: 12, gap: 8 },
  backButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 8, borderWidth: 2, borderColor: LINE, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  title: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  panel: { backgroundColor: PANEL, borderWidth: 3, borderColor: LINE, padding: 12, gap: 8 },
  sectionTitle: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  languageRow: {
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    minHeight: 60,
    paddingHorizontal: 10,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  langButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  langButton: {
    minWidth: 44,
    height: 30,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL,
    alignItems: 'center',
    justifyContent: 'center',
  },
  langButtonActive: {
    borderColor: MINT,
    backgroundColor: '#18302d',
  },
  langButtonText: {
    color: SOFT,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.9,
  },
  langButtonTextActive: {
    color: MINT,
  },
  toggleRow: {
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    minHeight: 60,
    paddingHorizontal: 10,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  toggleTextBlock: { flex: 1, gap: 4 },
  toggleTitle: { color: WHITE, fontSize: 12, fontWeight: '800' },
  toggleSubtitle: { color: SOFT, fontSize: 11, lineHeight: 15 },
  togglePill: { width: 52, height: 28, borderWidth: 2, borderColor: LINE, backgroundColor: PANEL, padding: 3, justifyContent: 'center' },
  togglePillActive: { borderColor: MINT, backgroundColor: '#18302d' },
  toggleKnob: { width: 16, height: 16, backgroundColor: SOFT },
  toggleKnobActive: { backgroundColor: MINT, alignSelf: 'flex-end' },
  actionRow: { backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: LINE, minHeight: 46, paddingHorizontal: 10, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  actionTitle: { fontSize: 12, fontWeight: '800', letterSpacing: 0.7 },
  actionArrow: { color: SOFT, fontSize: 10, fontWeight: '900', letterSpacing: 0.7 },
});
