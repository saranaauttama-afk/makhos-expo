import React, { useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface Props {
  onBack: () => void;
  onOpenShop: () => void;
  onReplay: () => void;
}

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_ALT = '#2c1f49';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const CYAN = '#5ec5ff';
const MINT = '#77f7cf';
const PINK = '#ff7dc4';
const ORANGE = '#ff9a3c';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

const MARQUEE = ['VICTORY', 'CLEAR', 'LEVEL UP', 'PIXEL WIN'];

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

export default function ResultScreen({ onBack, onOpenShop, onReplay }: Props) {
  const [marqueeFrame, setMarqueeFrame] = useState(0);
  const [badgeFrame, setBadgeFrame] = useState(0);

  useEffect(() => {
    const marqueeId = setInterval(() => {
      setMarqueeFrame(frame => (frame + 1) % MARQUEE.length);
    }, 900);
    const badgeId = setInterval(() => {
      setBadgeFrame(frame => (frame + 1) % 6);
    }, 120);
    return () => {
      clearInterval(marqueeId);
      clearInterval(badgeId);
    };
  }, []);

  const badgeScale = [1, 1.04, 1.08, 1.04, 1, 0.98][badgeFrame];
  const badgeGlow = [0.35, 0.48, 0.62, 0.48, 0.35, 0.28][badgeFrame];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <View style={styles.headerTop}>
            <Pressable style={styles.backButton} onPress={onBack}>
              <Text style={styles.backButtonText}>BACK</Text>
            </Pressable>
            <View style={styles.marquee}>
              <Text style={styles.marqueeText}>{MARQUEE[marqueeFrame]}</Text>
            </View>
          </View>

          <Text style={styles.kicker}>POST MATCH ARCADE</Text>
          <Text style={styles.title}>RESULT CHAMBER</Text>
          <Text style={styles.subtitle}>
            This is the high-energy finish screen. Rewards, upsell, and ad moments live here instead of interrupting the board.
          </Text>
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

          <Text style={styles.heroWord}>VICTORY</Text>
          <Text style={styles.heroCaption}>
            You pushed past the current rival and unlocked the next danger rung in the ladder.
          </Text>

          <View style={styles.rewardRow}>
            <RewardChip value="+18" label="RANK" tint={CYAN} />
            <RewardChip value="12:34" label="TIME" tint={MINT} />
            <RewardChip value="07" label="CAPS" tint={ORANGE} />
          </View>
        </View>

        <View style={styles.ladderPanel}>
          <Text style={styles.sectionTitle}>AI LADDER</Text>
          <LadderRow label="MM5" status="CLEARED" />
          <LadderRow label="MM7" status="DOMINATED" />
          <LadderRow label="MM9" status="NOW OPEN" active />
          <LadderRow label="MM11" status="LOCKED BOSS" />
        </View>

        <View style={styles.rewardPanel}>
          <Text style={styles.sectionTitle}>MATCH PAYOUT</Text>
          <View style={styles.payoutGrid}>
            <View style={styles.payoutCard}>
              <Text style={styles.payoutTitle}>FREE FLOW</Text>
              <Text style={styles.payoutCopy}>Play the next regular match with the standard post-result ad slot.</Text>
            </View>
            <View style={styles.payoutCard}>
              <Text style={styles.payoutTitle}>PREMIUM FLOW</Text>
              <Text style={styles.payoutCopy}>Remove ads permanently and unlock stronger AI rematches without friction.</Text>
            </View>
          </View>
        </View>

        <View style={styles.offerPanel}>
          <Text style={styles.offerTitle}>AD SLOT / PREMIUM GATE</Text>
          <Text style={styles.offerCopy}>
            Use this zone for interstitial ads, rewarded rematches, or a one-time remove-ads offer. Keep the board clean and place monetization only after the emotional payoff.
          </Text>
          <View style={styles.offerButtons}>
            <Pressable style={styles.primaryOfferButton} onPress={onOpenShop}>
              <Text style={styles.primaryOfferText}>REMOVE ADS + STRONG AI</Text>
            </Pressable>
            <Pressable style={styles.secondaryOfferButton} onPress={onReplay}>
              <Text style={styles.secondaryOfferText}>WATCH AD FOR BOSS REMATCH</Text>
            </Pressable>
          </View>
        </View>

        <View style={styles.ctaPanel}>
          <Pressable style={styles.primaryButton} onPress={onReplay}>
            <Text style={styles.primaryButtonText}>PLAY AGAIN</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={onBack}>
            <Text style={styles.secondaryButtonText}>RETURN HOME</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: BG },
  scrollContent: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 24, gap: 16 },
  headerPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: LINE, padding: 14, gap: 10 },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 10 },
  backButton: { paddingHorizontal: 12, paddingVertical: 10, borderWidth: 2, borderColor: LINE, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  marquee: { flex: 1, minHeight: 40, backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: CYAN, alignItems: 'center', justifyContent: 'center' },
  marqueeText: { color: CYAN, fontSize: 12, fontWeight: '900', letterSpacing: 1.2 },
  kicker: { color: GOLD, fontSize: 11, fontWeight: '800', letterSpacing: 1.3 },
  title: { color: WHITE, fontSize: 28, fontWeight: '900', letterSpacing: 1.1 },
  subtitle: { color: SOFT, fontSize: 13, lineHeight: 19 },
  heroPanel: { backgroundColor: PANEL_ALT, borderWidth: 3, borderColor: GOLD, padding: 16, gap: 12, alignItems: 'center' },
  heroBadge: {
    width: 74,
    height: 74,
    backgroundColor: PINK,
    borderWidth: 3,
    borderColor: GOLD,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: PINK,
    shadowRadius: 18,
    elevation: 8,
  },
  heroBadgeText: { color: WHITE, fontSize: 32, fontWeight: '900', letterSpacing: 1.2 },
  heroWord: { color: GOLD, fontSize: 34, fontWeight: '900', letterSpacing: 1.2 },
  heroCaption: { color: WHITE, fontSize: 13, lineHeight: 19, textAlign: 'center' },
  rewardRow: { flexDirection: 'row', gap: 10, width: '100%' },
  rewardChip: { flex: 1, minHeight: 88, backgroundColor: PANEL_DARK, borderWidth: 2, alignItems: 'center', justifyContent: 'center', gap: 6 },
  rewardValue: { fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  rewardLabel: { color: SOFT, fontSize: 11, fontWeight: '800', letterSpacing: 0.8 },
  ladderPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: MINT, padding: 14, gap: 10 },
  sectionTitle: { color: MINT, fontSize: 14, fontWeight: '900', letterSpacing: 1 },
  ladderRow: {
    minHeight: 48,
    backgroundColor: PANEL_DARK,
    borderWidth: 2,
    borderColor: LINE,
    paddingHorizontal: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  ladderRowActive: { borderColor: GOLD, backgroundColor: '#33203f' },
  ladderLabel: { color: WHITE, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  ladderLabelActive: { color: GOLD },
  ladderStatus: { color: SOFT, fontSize: 12, fontWeight: '800', letterSpacing: 0.8 },
  ladderStatusActive: { color: CYAN },
  rewardPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: CYAN, padding: 14, gap: 10 },
  payoutGrid: { gap: 10 },
  payoutCard: { minHeight: 82, backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: LINE, padding: 12, gap: 6 },
  payoutTitle: { color: CYAN, fontSize: 13, fontWeight: '900', letterSpacing: 0.9 },
  payoutCopy: { color: WHITE, fontSize: 12, lineHeight: 18 },
  offerPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: PINK, padding: 14, gap: 10 },
  offerTitle: { color: PINK, fontSize: 15, fontWeight: '900', letterSpacing: 1 },
  offerCopy: { color: WHITE, fontSize: 12, lineHeight: 18 },
  offerButtons: { gap: 10 },
  primaryOfferButton: { minHeight: 52, backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: PINK, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 12 },
  primaryOfferText: { color: PINK, fontSize: 12, fontWeight: '900', letterSpacing: 0.8, textAlign: 'center' },
  secondaryOfferButton: { minHeight: 50, backgroundColor: '#33203f', borderWidth: 2, borderColor: GOLD, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 12 },
  secondaryOfferText: { color: GOLD, fontSize: 12, fontWeight: '900', letterSpacing: 0.8, textAlign: 'center' },
  ctaPanel: { gap: 10 },
  primaryButton: { minHeight: 58, backgroundColor: PANEL_ALT, borderWidth: 3, borderColor: CYAN, alignItems: 'center', justifyContent: 'center' },
  primaryButtonText: { color: WHITE, fontSize: 15, fontWeight: '900', letterSpacing: 1 },
  secondaryButton: { minHeight: 52, backgroundColor: PANEL, borderWidth: 2, borderColor: LINE, alignItems: 'center', justifyContent: 'center' },
  secondaryButtonText: { color: SOFT, fontSize: 12, fontWeight: '800', letterSpacing: 0.8 },
});
