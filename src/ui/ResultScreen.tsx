import React, { useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { MonetizationState } from './types';

type RewardKind = 'hint' | 'undo';

interface Props {
  language: AppLanguage;
  monetization: MonetizationState;
  onBack: () => void;
  onOpenShop: () => void;
  onWatchReward: (reward: RewardKind) => Promise<boolean>;
  onReplay: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_ALT = '#2b5f59';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const MINT = '#b8f3df';
const PINK = '#f2c5c5';
const ORANGE = '#f4c88e';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

const COPY = {
  th: {
    marquee: ['VICTORY', 'CLEAR', 'LEVEL UP', 'PIXEL WIN'],
    back: 'BACK',
    kicker: 'POST MATCH ARCADE',
    title: 'RESULT CHAMBER',
    subtitle: 'หน้าสรุปผลหลังเกม รวมรางวัลและทางเลือกซื้อไว้ที่เดียว โดยไม่รบกวนหน้ากระดาน',
    heroWord: 'VICTORY',
    heroCaption: 'คุณผ่านคู่แข่งรอบนี้แล้ว เตรียมท้าชั้นถัดไปได้เลย',
    rank: 'RANK',
    time: 'TIME',
    caps: 'CAPS',
    ladderTitle: 'AI LADDER',
    ladderMm5: 'CLEARED',
    ladderMm7: 'DOMINATED',
    ladderMm9: 'NOW OPEN',
    ladderMm11: 'LOCKED BOSS',
    payoutTitle: 'MATCH PAYOUT',
    noAdsActive: 'เปิด No Ads แล้ว',
    freeMode: 'โหมดฟรี (มี interstitial หลังจบบางแมตช์)',
    freeFlowTitle: 'FREE FLOW',
    freeFlowCopy: 'เล่นต่อแมตช์ปกติได้ทันทีตาม flow ของโหมดฟรี',
    premiumFlowTitle: 'PREMIUM FLOW',
    premiumFlowCopy: 'ปลด No Ads เพื่อเล่นต่อแบบลื่นและเข้าถึงคู่แข่งโหดขึ้นได้สะดวก',
    offerTitle: 'AD SLOT / PREMIUM GATE',
    offerCopy: 'โซนนี้ใช้วาง interstitial, rewarded, หรือข้อเสนอ No Ads หลังจบเกมเท่านั้น',
    removeAds: 'REMOVE ADS + STRONG AI',
    loadingHint: 'LOADING HINT...',
    loadingUndo: 'LOADING UNDO...',
    watchHint: 'WATCH AD: +1 HINT',
    watchUndo: 'WATCH AD: +1 UNDO',
    rewardAdded: (kind: RewardKind) => `รับรางวัลแล้ว: +1 ${kind.toUpperCase()}`,
    rewardFailed: 'โฆษณาไม่พร้อม ตรวจ Ad Consent ใน Settings',
    playAgain: 'PLAY AGAIN',
    returnHome: 'RETURN HOME',
  },
  en: {
    marquee: ['VICTORY', 'CLEAR', 'LEVEL UP', 'PIXEL WIN'],
    back: 'BACK',
    kicker: 'POST MATCH ARCADE',
    title: 'RESULT CHAMBER',
    subtitle: 'High-energy post-match screen with rewards and premium options, without interrupting board play.',
    heroWord: 'VICTORY',
    heroCaption: 'You pushed past this rival. The next ladder rung is open.',
    rank: 'RANK',
    time: 'TIME',
    caps: 'CAPS',
    ladderTitle: 'AI LADDER',
    ladderMm5: 'CLEARED',
    ladderMm7: 'DOMINATED',
    ladderMm9: 'NOW OPEN',
    ladderMm11: 'LOCKED BOSS',
    payoutTitle: 'MATCH PAYOUT',
    noAdsActive: 'No Ads active',
    freeMode: 'Free mode with interstitial ads',
    freeFlowTitle: 'FREE FLOW',
    freeFlowCopy: 'Play the next regular match with standard post-result ad slots.',
    premiumFlowTitle: 'PREMIUM FLOW',
    premiumFlowCopy: 'Remove ads permanently and unlock stronger rematches with less friction.',
    offerTitle: 'AD SLOT / PREMIUM GATE',
    offerCopy: 'Use this area for interstitials, rewarded boosts, or one-time No Ads offers after emotional payoff.',
    removeAds: 'REMOVE ADS + STRONG AI',
    loadingHint: 'LOADING HINT...',
    loadingUndo: 'LOADING UNDO...',
    watchHint: 'WATCH AD: +1 HINT',
    watchUndo: 'WATCH AD: +1 UNDO',
    rewardAdded: (kind: RewardKind) => `Reward added: +1 ${kind.toUpperCase()}`,
    rewardFailed: 'Reward ad unavailable. Check consent in Settings.',
    playAgain: 'PLAY AGAIN',
    returnHome: 'RETURN HOME',
  },
} as const;

function RewardChip({ value, label, tint }: { value: string; label: string; tint: string }) {
  return (
    <View style={[styles.rewardChip, { borderColor: tint }]}>
      <Text style={[styles.rewardValue, { color: tint }]}>{value}</Text>
      <Text style={styles.rewardLabel}>{label}</Text>
    </View>
  );
}

function LadderRow({
  label,
  status,
  active,
}: {
  label: string;
  status: string;
  active?: boolean;
}) {
  return (
    <View style={[styles.ladderRow, active && styles.ladderRowActive]}>
      <Text style={[styles.ladderLabel, active && styles.ladderLabelActive]}>{label}</Text>
      <Text style={[styles.ladderStatus, active && styles.ladderStatusActive]}>{status}</Text>
    </View>
  );
}

export default function ResultScreen({ language, monetization, onBack, onOpenShop, onWatchReward, onReplay }: Props) {
  const t = COPY[language];
  const [marqueeFrame, setMarqueeFrame] = useState(0);
  const [badgeFrame, setBadgeFrame] = useState(0);
  const [busyReward, setBusyReward] = useState<RewardKind | null>(null);
  const [rewardNote, setRewardNote] = useState('');

  useEffect(() => {
    const marqueeId = setInterval(() => {
      setMarqueeFrame(frame => (frame + 1) % t.marquee.length);
    }, 900);
    const badgeId = setInterval(() => {
      setBadgeFrame(frame => (frame + 1) % 6);
    }, 120);
    return () => {
      clearInterval(marqueeId);
      clearInterval(badgeId);
    };
  }, [t.marquee.length]);

  const badgeScale = [1, 1.04, 1.08, 1.04, 1, 0.98][badgeFrame];
  const badgeGlow = [0.35, 0.48, 0.62, 0.48, 0.35, 0.28][badgeFrame];
  const noAdsLine = monetization.noAdsUnlocked ? t.noAdsActive : t.freeMode;

  async function runReward(kind: RewardKind) {
    if (busyReward) return;
    setBusyReward(kind);
    const ok = await onWatchReward(kind);
    setBusyReward(null);
    setRewardNote(ok ? t.rewardAdded(kind) : t.rewardFailed);
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <View style={styles.headerTop}>
            <Pressable style={styles.backButton} onPress={onBack}>
              <Text style={styles.backButtonText}>{t.back}</Text>
            </Pressable>
            <View style={styles.marquee}>
              <Text style={styles.marqueeText}>{t.marquee[marqueeFrame]}</Text>
            </View>
          </View>

          <Text style={styles.kicker}>{t.kicker}</Text>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <View style={styles.heroPanel}>
          <View
            style={[
              styles.heroBadge,
              {
                transform: [{ scale: badgeScale }],
                shadowOpacity: badgeGlow,
              },
            ]}
          >
            <Text style={styles.heroBadgeText}>V</Text>
          </View>

          <Text style={styles.heroWord}>{t.heroWord}</Text>
          <Text style={styles.heroCaption}>{t.heroCaption}</Text>

          <View style={styles.rewardRow}>
            <RewardChip value="+18" label={t.rank} tint={CYAN} />
            <RewardChip value="12:34" label={t.time} tint={MINT} />
            <RewardChip value="07" label={t.caps} tint={ORANGE} />
          </View>
        </View>

        <View style={styles.ladderPanel}>
          <Text style={styles.sectionTitle}>{t.ladderTitle}</Text>
          <LadderRow label="MM5" status={t.ladderMm5} />
          <LadderRow label="MM7" status={t.ladderMm7} />
          <LadderRow label="MM9" status={t.ladderMm9} active />
          <LadderRow label="MM11" status={t.ladderMm11} />
        </View>

        <View style={styles.rewardPanel}>
          <Text style={styles.sectionTitle}>{t.payoutTitle}</Text>
          <Text style={styles.payoutBanner}>{noAdsLine}</Text>
          <View style={styles.payoutGrid}>
            <View style={styles.payoutCard}>
              <Text style={styles.payoutTitle}>{t.freeFlowTitle}</Text>
              <Text style={styles.payoutCopy}>{t.freeFlowCopy}</Text>
            </View>
            <View style={styles.payoutCard}>
              <Text style={styles.payoutTitle}>{t.premiumFlowTitle}</Text>
              <Text style={styles.payoutCopy}>{t.premiumFlowCopy}</Text>
            </View>
          </View>
        </View>

        <View style={styles.offerPanel}>
          <Text style={styles.offerTitle}>{t.offerTitle}</Text>
          <Text style={styles.offerCopy}>{t.offerCopy}</Text>
          <View style={styles.offerButtons}>
            <Pressable style={styles.primaryOfferButton} onPress={onOpenShop}>
              <Text style={styles.primaryOfferText}>{t.removeAds}</Text>
            </Pressable>
            <Pressable
              style={[styles.secondaryOfferButton, busyReward === 'hint' && styles.offerButtonDisabled]}
              onPress={() => { void runReward('hint'); }}
            >
              <Text style={styles.secondaryOfferText}>
                {busyReward === 'hint' ? t.loadingHint : t.watchHint}
              </Text>
            </Pressable>
            <Pressable
              style={[styles.secondaryOfferButton, busyReward === 'undo' && styles.offerButtonDisabled]}
              onPress={() => { void runReward('undo'); }}
            >
              <Text style={styles.secondaryOfferText}>
                {busyReward === 'undo' ? t.loadingUndo : t.watchUndo}
              </Text>
            </Pressable>
            {!!rewardNote && <Text style={styles.rewardNote}>{rewardNote}</Text>}
          </View>
        </View>

        <View style={styles.ctaPanel}>
          <Pressable style={styles.primaryButton} onPress={onReplay}>
            <Text style={styles.primaryButtonText}>{t.playAgain}</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={onBack}>
            <Text style={styles.secondaryButtonText}>{t.returnHome}</Text>
          </Pressable>
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
  headerPanel: { backgroundColor: PANEL, borderWidth: 1, borderColor: LINE, borderRadius: 14, padding: 12, gap: 8 },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 10 },
  backButton: { paddingHorizontal: 10, paddingVertical: 8, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  marquee: { flex: 1, minHeight: 34, backgroundColor: PANEL_DARK, borderWidth: 1, borderColor: CYAN, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  marqueeText: { color: CYAN, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  title: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  heroPanel: { backgroundColor: PANEL_ALT, borderWidth: 1, borderColor: GOLD, borderRadius: 14, padding: 12, gap: 8, alignItems: 'center' },
  heroBadge: {
    width: 58,
    height: 58,
    backgroundColor: '#2f6b62',
    borderWidth: 1,
    borderRadius: 29,
    borderColor: GOLD,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: PINK,
    shadowRadius: 18,
    elevation: 8,
  },
  heroBadgeText: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  heroWord: { color: GOLD, fontSize: 28, fontWeight: '900', letterSpacing: 1 },
  heroCaption: { color: WHITE, fontSize: 12, lineHeight: 17, textAlign: 'center' },
  rewardRow: { flexDirection: 'row', gap: 8, width: '100%' },
  rewardChip: { flex: 1, minHeight: 70, backgroundColor: PANEL_DARK, borderWidth: 1, borderRadius: 12, alignItems: 'center', justifyContent: 'center', gap: 4 },
  rewardValue: { fontSize: 18, fontWeight: '900', letterSpacing: 0.9 },
  rewardLabel: { color: SOFT, fontSize: 10, fontWeight: '800', letterSpacing: 0.7 },
  ladderPanel: { backgroundColor: PANEL, borderWidth: 1, borderColor: MINT, borderRadius: 14, padding: 12, gap: 8 },
  sectionTitle: { color: MINT, fontSize: 12, fontWeight: '900', letterSpacing: 0.9 },
  ladderRow: {
    minHeight: 40,
    backgroundColor: PANEL_DARK,
    borderWidth: 1,
    borderRadius: 10,
    borderColor: LINE,
    paddingHorizontal: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  ladderRowActive: { borderColor: GOLD, backgroundColor: '#33203f' },
  ladderLabel: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 0.7 },
  ladderLabelActive: { color: GOLD },
  ladderStatus: { color: SOFT, fontSize: 10, fontWeight: '800', letterSpacing: 0.7 },
  ladderStatusActive: { color: CYAN },
  rewardPanel: { backgroundColor: PANEL, borderWidth: 1, borderColor: CYAN, borderRadius: 14, padding: 12, gap: 8 },
  payoutBanner: { color: GOLD, fontSize: 11, fontWeight: '800' },
  payoutGrid: { gap: 8 },
  payoutCard: { minHeight: 68, backgroundColor: PANEL_DARK, borderWidth: 1, borderColor: LINE, borderRadius: 10, padding: 10, gap: 4 },
  payoutTitle: { color: CYAN, fontSize: 11, fontWeight: '900', letterSpacing: 0.8 },
  payoutCopy: { color: WHITE, fontSize: 11, lineHeight: 16 },
  offerPanel: { backgroundColor: PANEL, borderWidth: 1, borderColor: PINK, borderRadius: 14, padding: 12, gap: 8 },
  offerTitle: { color: PINK, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  offerCopy: { color: WHITE, fontSize: 11, lineHeight: 16 },
  offerButtons: { gap: 8 },
  primaryOfferButton: { minHeight: 44, backgroundColor: PANEL_DARK, borderWidth: 1, borderColor: PINK, borderRadius: 999, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 10 },
  primaryOfferText: { color: PINK, fontSize: 11, fontWeight: '900', letterSpacing: 0.7, textAlign: 'center' },
  secondaryOfferButton: { minHeight: 42, backgroundColor: '#2f6b62', borderWidth: 1, borderColor: GOLD, borderRadius: 999, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 10 },
  secondaryOfferText: { color: GOLD, fontSize: 11, fontWeight: '900', letterSpacing: 0.7, textAlign: 'center' },
  offerButtonDisabled: { opacity: 0.6 },
  rewardNote: { color: MINT, fontSize: 11, lineHeight: 16 },
  ctaPanel: { gap: 8 },
  primaryButton: { minHeight: 50, backgroundColor: PANEL_ALT, borderWidth: 1, borderColor: CYAN, borderRadius: 999, alignItems: 'center', justifyContent: 'center' },
  primaryButtonText: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  secondaryButton: { minHeight: 44, backgroundColor: PANEL, borderWidth: 1, borderColor: LINE, borderRadius: 999, alignItems: 'center', justifyContent: 'center' },
  secondaryButtonText: { color: SOFT, fontSize: 11, fontWeight: '800', letterSpacing: 0.7 },
});
