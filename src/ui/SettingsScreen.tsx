import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { AdConsentStatus, MonetizationState } from './types';

interface Props {
  language: AppLanguage;
  onLanguageChange: (language: AppLanguage) => void;
  monetization: MonetizationState;
  onAdConsentChange: (consent: AdConsentStatus) => void;
  onRestorePurchase: () => void;
  onManagePremium: () => void;
  aiModels: Array<{ id: string; label: string }>;
  aiModelId: string;
  onAiModelChange: (modelId: string) => void;
  onBack: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const MINT = '#b8f3df';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

const COPY = {
  th: {
    system: 'SYSTEM MENU',
    title: 'SETTINGS',
    subtitle: 'จัดการภาษา เสียง และตัวเลือกระบบ',
    preferences: 'PREFERENCES',
    language: 'Language',
    languageSub: 'สลับภาษาแอปได้ทันที',
    model: 'AI Model',
    modelSub: 'เลือก ONNX model ที่ใช้ตอนเล่นจริง',
    sound: 'Sound',
    soundSub: 'เสียงเอฟเฟกต์ตอนเดินและผลเกม',
    vibration: 'Vibration',
    vibrationSub: 'สั่นเบาๆ ตอนยืนยันการเดิน',
    retroFx: 'Retro FX',
    retroFxSub: 'เอฟเฟกต์พิกเซลและสไตล์อาร์เคด',
    account: 'ACCOUNT / PURCHASE',
    adConsent: 'Ad Consent',
    adConsentSub: 'อนุญาตโฆษณาและ rewarded video สำหรับรับรางวัล',
    allow: 'ALLOW',
    deny: 'DENY',
    noAdsState: 'No Ads Status',
    noAdsActive: 'ACTIVE',
    noAdsFree: 'FREE MODE',
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
    model: 'AI Model',
    modelSub: 'Select the ONNX model used in live gameplay.',
    sound: 'Sound',
    soundSub: 'Move sounds and match result cues.',
    vibration: 'Vibration',
    vibrationSub: 'Light feedback on move confirm.',
    retroFx: 'Retro FX',
    retroFxSub: 'Pixel overlays and arcade effects.',
    account: 'ACCOUNT / PURCHASE',
    adConsent: 'Ad Consent',
    adConsentSub: 'Allow ads and rewarded videos for ad-based rewards.',
    allow: 'ALLOW',
    deny: 'DENY',
    noAdsState: 'No Ads Status',
    noAdsActive: 'ACTIVE',
    noAdsFree: 'FREE MODE',
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

function ActionRow({ title, tint, openText, onPress }: { title: string; tint: string; openText: string; onPress?: () => void }) {
  return (
    <Pressable style={styles.actionRow} onPress={onPress}>
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

function ConsentRow({
  title,
  subtitle,
  allowText,
  denyText,
  value,
  onChange,
}: {
  title: string;
  subtitle: string;
  allowText: string;
  denyText: string;
  value: AdConsentStatus;
  onChange: (v: AdConsentStatus) => void;
}) {
  return (
    <View style={styles.modelRow}>
      <View style={styles.toggleTextBlock}>
        <Text style={styles.toggleTitle}>{title}</Text>
        <Text style={styles.toggleSubtitle}>{subtitle}</Text>
      </View>
      <View style={styles.modelButtons}>
        <Pressable style={[styles.modelButton, value === 'granted' && styles.modelButtonActive]} onPress={() => onChange('granted')}>
          <Text style={[styles.modelButtonText, value === 'granted' && styles.modelButtonTextActive]}>{allowText}</Text>
        </Pressable>
        <Pressable style={[styles.modelButton, value === 'denied' && styles.modelButtonActive]} onPress={() => onChange('denied')}>
          <Text style={[styles.modelButtonText, value === 'denied' && styles.modelButtonTextActive]}>{denyText}</Text>
        </Pressable>
      </View>
    </View>
  );
}

function ModelRow({
  title,
  subtitle,
  models,
  activeModelId,
  onChange,
}: {
  title: string;
  subtitle: string;
  models: Array<{ id: string; label: string }>;
  activeModelId: string;
  onChange: (modelId: string) => void;
}) {
  return (
    <View style={styles.modelRow}>
      <View style={styles.toggleTextBlock}>
        <Text style={styles.toggleTitle}>{title}</Text>
        <Text style={styles.toggleSubtitle}>{subtitle}</Text>
      </View>
      <View style={styles.modelButtons}>
        {models.map(model => {
          const active = model.id === activeModelId;
          return (
            <Pressable
              key={model.id}
              style={[styles.modelButton, active && styles.modelButtonActive]}
              onPress={() => onChange(model.id)}
            >
              <Text style={[styles.modelButtonText, active && styles.modelButtonTextActive]}>{model.label}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

export default function SettingsScreen({
  language,
  onLanguageChange,
  monetization,
  onAdConsentChange,
  onRestorePurchase,
  onManagePremium,
  aiModels,
  aiModelId,
  onAiModelChange,
  onBack,
}: Props) {
  const [soundOn, setSoundOn] = useState(true);
  const [vibrationOn, setVibrationOn] = useState(true);
  const [retroFx, setRetroFx] = useState(true);
  const t = COPY[language];

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
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
          <ModelRow
            title={t.model}
            subtitle={t.modelSub}
            models={aiModels}
            activeModelId={aiModelId}
            onChange={onAiModelChange}
          />
          <ToggleRow title={t.sound} subtitle={t.soundSub} active={soundOn} onToggle={() => setSoundOn(v => !v)} />
          <ToggleRow title={t.vibration} subtitle={t.vibrationSub} active={vibrationOn} onToggle={() => setVibrationOn(v => !v)} />
          <ToggleRow title={t.retroFx} subtitle={t.retroFxSub} active={retroFx} onToggle={() => setRetroFx(v => !v)} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.account}</Text>
          <ConsentRow
            title={t.adConsent}
            subtitle={t.adConsentSub}
            allowText={t.allow}
            denyText={t.deny}
            value={monetization.adConsent}
            onChange={onAdConsentChange}
          />
          <ActionRow
            title={`${t.noAdsState}: ${monetization.noAdsUnlocked ? t.noAdsActive : t.noAdsFree}`}
            tint={MINT}
            openText={t.open}
          />
          <ActionRow title={t.restore} tint={CYAN} openText={t.open} onPress={onRestorePurchase} />
          <ActionRow title={t.premium} tint={GOLD} openText={t.open} onPress={onManagePremium} />
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
  scrollContent: { paddingHorizontal: 18, paddingTop: 12, paddingBottom: 22, gap: 12 },
  headerPanel: { backgroundColor: PANEL, borderWidth: 1, borderRadius: 14, borderColor: LINE, padding: 12, gap: 8 },
  backButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 8, borderWidth: 1, borderRadius: 999, borderColor: LINE, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  title: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  panel: { backgroundColor: PANEL, borderWidth: 1, borderRadius: 14, borderColor: LINE, padding: 12, gap: 8 },
  sectionTitle: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  languageRow: {
    backgroundColor: PANEL_DARK,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    minHeight: 60,
    paddingHorizontal: 10,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  modelRow: {
    backgroundColor: PANEL_DARK,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    minHeight: 60,
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 8,
  },
  modelButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  modelButton: {
    minHeight: 28,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL,
    borderRadius: 999,
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  modelButtonActive: {
    borderColor: GOLD,
    backgroundColor: '#302718',
  },
  modelButtonText: {
    color: SOFT,
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  modelButtonTextActive: {
    color: GOLD,
  },
  langButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  langButton: {
    minWidth: 44,
    height: 30,
    borderWidth: 1,
    borderColor: LINE,
    backgroundColor: PANEL,
    borderRadius: 999,
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
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
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
  togglePill: { width: 52, height: 28, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL, padding: 3, justifyContent: 'center' },
  togglePillActive: { borderColor: MINT, backgroundColor: '#18302d' },
  toggleKnob: { width: 16, height: 16, backgroundColor: SOFT },
  toggleKnobActive: { backgroundColor: MINT, alignSelf: 'flex-end' },
  actionRow: { backgroundColor: PANEL_DARK, borderWidth: 1, borderRadius: 12, borderColor: LINE, minHeight: 46, paddingHorizontal: 10, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  actionTitle: { fontSize: 12, fontWeight: '800', letterSpacing: 0.7 },
  actionArrow: { color: SOFT, fontSize: 10, fontWeight: '900', letterSpacing: 0.7 },
});
