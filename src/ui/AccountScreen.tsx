import React, { useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import PurchaseCard from './components/PurchaseCard';
import RewardActionRow from './components/RewardActionRow';
import WalletStatCard from './components/WalletStatCard';
import { ACCOUNT_TEXT } from './i18n/accountText';
import { AdConsentStatus, MonetizationState } from './types';
import { RewardKind } from './walletStore';

interface Props {
  language: AppLanguage;
  onLanguageChange: (language: AppLanguage) => void;
  monetization: MonetizationState;
  onAdConsentChange: (consent: AdConsentStatus) => void;
  aiModels: Array<{ id: string; label: string }>;
  aiModelId: string;
  onAiModelChange: (modelId: string) => void;
  onClaimFreeReward: (kind: RewardKind) => Promise<boolean>;
  onBuyNoAds: () => void;
  onBuyStarterPack: () => void;
  onRestorePurchase: () => void;
  onBack: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const MINT = '#b8f3df';
const PINK = '#f2c5c5';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

function ToggleRow({
  title,
  active,
  onToggle,
}: {
  title: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <Pressable onPress={onToggle} style={styles.rowCard}>
      <Text style={styles.rowTitle}>{title}</Text>
      <View style={[styles.togglePill, active && styles.togglePillActive]}>
        <View style={[styles.toggleKnob, active && styles.toggleKnobActive]} />
      </View>
    </Pressable>
  );
}

export default function AccountScreen({
  language,
  onLanguageChange,
  monetization,
  onAdConsentChange,
  aiModels,
  aiModelId,
  onAiModelChange,
  onClaimFreeReward,
  onBuyNoAds,
  onBuyStarterPack,
  onRestorePurchase,
  onBack,
}: Props) {
  const [soundOn, setSoundOn] = useState(true);
  const [vibrationOn, setVibrationOn] = useState(true);
  const [retroFx, setRetroFx] = useState(true);
  const [busyReward, setBusyReward] = useState<RewardKind | null>(null);
  const t = ACCOUNT_TEXT[language];

  async function claim(kind: RewardKind) {
    if (busyReward) return;
    setBusyReward(kind);
    const ok = await onClaimFreeReward(kind);
    setBusyReward(null);
    if (!ok) Alert.alert(t.adUnavailableTitle, t.adUnavailableBody);
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>{t.back}</Text>
          </Pressable>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <View style={styles.walletRow}>
          <WalletStatCard value={monetization.coins} label={t.coins} accent={GOLD} />
          <WalletStatCard value={monetization.hintCredits} label={t.hintCredits} accent={CYAN} />
          <WalletStatCard value={monetization.undoCredits} label={t.undoCredits} accent={MINT} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.app}</Text>

          <View style={styles.rowCard}>
            <View style={styles.rowTextBlock}>
              <Text style={styles.rowTitle}>{t.language}</Text>
              <Text style={styles.rowSub}>{t.languageSub}</Text>
            </View>
            <View style={styles.langButtons}>
              <Pressable style={[styles.langButton, language === 'th' && styles.langButtonActive]} onPress={() => onLanguageChange('th')}>
                <Text style={[styles.langButtonText, language === 'th' && styles.langButtonTextActive]}>TH</Text>
              </Pressable>
              <Pressable style={[styles.langButton, language === 'en' && styles.langButtonActive]} onPress={() => onLanguageChange('en')}>
                <Text style={[styles.langButtonText, language === 'en' && styles.langButtonTextActive]}>EN</Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.rowCardColumn}>
            <Text style={styles.rowTitle}>{t.model}</Text>
            <Text style={styles.rowSub}>{t.modelSub}</Text>
            <View style={styles.modelButtons}>
              {aiModels.map(model => {
                const active = model.id === aiModelId;
                return (
                  <Pressable
                    key={model.id}
                    style={[styles.modelButton, active && styles.modelButtonActive]}
                    onPress={() => onAiModelChange(model.id)}
                  >
                    <Text style={[styles.modelButtonText, active && styles.modelButtonTextActive]}>{model.label}</Text>
                  </Pressable>
                );
              })}
            </View>
          </View>

          <ToggleRow title={t.sound} active={soundOn} onToggle={() => setSoundOn(v => !v)} />
          <ToggleRow title={t.vibration} active={vibrationOn} onToggle={() => setVibrationOn(v => !v)} />
          <ToggleRow title={t.retroFx} active={retroFx} onToggle={() => setRetroFx(v => !v)} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.rewards}</Text>
          <RewardActionRow
            title={t.adCoins}
            subtitle={t.optionalText}
            cta={busyReward === 'coins' ? t.loading : t.watchAd}
            tint={GOLD}
            disabled={!!busyReward}
            onPress={() => {
              void claim('coins');
            }}
          />
          <RewardActionRow
            title={t.adHint}
            subtitle={t.optionalText}
            cta={busyReward === 'hint' ? t.loading : t.watchAd}
            tint={CYAN}
            disabled={!!busyReward}
            onPress={() => {
              void claim('hint');
            }}
          />
          <RewardActionRow
            title={t.adUndo}
            subtitle={t.optionalText}
            cta={busyReward === 'undo' ? t.loading : t.watchAd}
            tint={MINT}
            disabled={!!busyReward}
            onPress={() => {
              void claim('undo');
            }}
          />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>{t.purchases}</Text>

          <View style={styles.rowCardColumn}>
            <Text style={styles.rowTitle}>{t.consent}</Text>
            <Text style={styles.rowSub}>{t.consentSub}</Text>
            <View style={styles.modelButtons}>
              <Pressable
                style={[styles.modelButton, monetization.adConsent === 'granted' && styles.modelButtonActive]}
                onPress={() => onAdConsentChange('granted')}
              >
                <Text style={[styles.modelButtonText, monetization.adConsent === 'granted' && styles.modelButtonTextActive]}>{t.allow}</Text>
              </Pressable>
              <Pressable
                style={[styles.modelButton, monetization.adConsent === 'denied' && styles.modelButtonActive]}
                onPress={() => onAdConsentChange('denied')}
              >
                <Text style={[styles.modelButtonText, monetization.adConsent === 'denied' && styles.modelButtonTextActive]}>{t.deny}</Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.rowCard}>
            <Text style={styles.rowTitle}>{t.noAds}</Text>
            <Text style={[styles.statusPill, monetization.noAdsUnlocked ? styles.statusActive : styles.statusFree]}>
              {monetization.noAdsUnlocked ? t.active : t.free}
            </Text>
          </View>

          <PurchaseCard
            title={t.noAdsTitle}
            subtitle={t.noAdsSub}
            price={t.noAdsPrice}
            copy={t.noAdsCopy}
            cta={t.noAdsCta}
            tint={PINK}
            onPress={onBuyNoAds}
            owned={monetization.noAdsUnlocked}
            ownedLabel={t.owned}
          />

          <PurchaseCard
            title={t.starterTitle}
            subtitle={t.starterSub}
            price={t.starterPrice}
            copy={t.starterCopy}
            cta={t.starterCta}
            tint={CYAN}
            onPress={onBuyStarterPack}
          />

          <PurchaseCard
            title={t.restoreTitle}
            price={t.restorePrice}
            copy={t.restoreCopy}
            cta={t.restoreCta}
            tint={GOLD}
            onPress={onRestorePurchase}
          />
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
  scrollContent: { paddingHorizontal: 18, paddingTop: 12, paddingBottom: 24, gap: 12 },
  headerPanel: { backgroundColor: PANEL, borderWidth: 1, borderColor: LINE, borderRadius: 14, padding: 12, gap: 6 },
  backButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 8, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  title: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },

  panel: { backgroundColor: PANEL, borderWidth: 1, borderColor: LINE, borderRadius: 14, padding: 12, gap: 8 },
  sectionTitle: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.8 },
  walletRow: { flexDirection: 'row', gap: 8 },
  rowCard: {
    minHeight: 52,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    backgroundColor: PANEL_DARK,
    paddingHorizontal: 10,
    paddingVertical: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 8,
  },
  rowCardColumn: {
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    backgroundColor: PANEL_DARK,
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 7,
  },
  rowTextBlock: { flex: 1, gap: 3 },
  rowTitle: { color: WHITE, fontSize: 12, fontWeight: '800' },
  rowSub: { color: SOFT, fontSize: 10, lineHeight: 14 },

  langButtons: { flexDirection: 'row', gap: 6 },
  langButton: {
    minWidth: 44,
    height: 30,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 999,
    backgroundColor: PANEL,
    alignItems: 'center',
    justifyContent: 'center',
  },
  langButtonActive: { borderColor: MINT, backgroundColor: '#2f6b62' },
  langButtonText: { color: SOFT, fontSize: 11, fontWeight: '900', letterSpacing: 0.8 },
  langButtonTextActive: { color: WHITE },

  modelButtons: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  modelButton: {
    minHeight: 30,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 999,
    backgroundColor: PANEL,
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  modelButtonActive: { borderColor: GOLD, backgroundColor: '#376d64' },
  modelButtonText: { color: SOFT, fontSize: 10, fontWeight: '900', letterSpacing: 0.7 },
  modelButtonTextActive: { color: WHITE },

  togglePill: { width: 52, height: 28, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL, padding: 3, justifyContent: 'center' },
  togglePillActive: { borderColor: MINT, backgroundColor: '#2f6b62' },
  toggleKnob: { width: 16, height: 16, backgroundColor: SOFT, borderRadius: 8 },
  toggleKnobActive: { backgroundColor: MINT, alignSelf: 'flex-end' },

  statusPill: {
    minHeight: 28,
    borderRadius: 999,
    paddingHorizontal: 10,
    textAlignVertical: 'center',
    includeFontPadding: false,
    lineHeight: 28,
    fontSize: 10,
    fontWeight: '900',
    overflow: 'hidden',
  },
  statusActive: { color: MINT, backgroundColor: '#2f6b62' },
  statusFree: { color: SOFT, backgroundColor: PANEL },
});
