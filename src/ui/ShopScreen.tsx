import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MonetizationState } from './types';

interface Props {
  monetization: MonetizationState;
  onBuyNoAds: () => void;
  onBuyStarterPack: () => void;
  onRestorePurchase: () => void;
  onBack: () => void;
}

const BG = '#3f837b';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const PANEL_ALT = '#2b5f59';
const PANEL_DARK = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const MINT = '#b8f3df';
const CYAN = '#9be7da';
const PINK = '#f2c5c5';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

function OfferCard({
  title,
  price,
  tint,
  copy,
  cta,
  onPress,
  owned = false,
}: {
  title: string;
  price: string;
  tint: string;
  copy: string;
  cta: string;
  onPress: () => void;
  owned?: boolean;
}) {
  return (
    <View style={[styles.offerCard, { borderColor: tint }]}>
      <View style={styles.offerTop}>
        <Text style={[styles.offerTitle, { color: tint }]}>{title}</Text>
        <Text style={styles.offerPrice}>{price}</Text>
      </View>
      <Text style={styles.offerCopy}>{copy}</Text>
      <Pressable style={[styles.offerButton, { borderColor: tint }, owned && styles.offerButtonOwned]} onPress={onPress}>
        <Text style={[styles.offerButtonText, { color: tint }]}>{owned ? 'OWNED' : cta}</Text>
      </Pressable>
    </View>
  );
}

export default function ShopScreen({ monetization, onBuyNoAds, onBuyStarterPack, onRestorePurchase, onBack }: Props) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>BACK</Text>
          </Pressable>
          <Text style={styles.kicker}>PIXEL STORE</Text>
          <Text style={styles.title}>SHOP / NO ADS</Text>
          <Text style={styles.subtitle}>Skeleton commerce page for rewarded access, premium AI, and ad removal.</Text>
        </View>

        <OfferCard
          title="REMOVE ADS"
          price="$2.99"
          tint={PINK}
          copy="Turns off interstitial ads and keeps the board flow clean after every match."
          cta="BUY NO ADS"
          onPress={onBuyNoAds}
          owned={monetization.noAds}
        />
        <OfferCard
          title="STARTER PACK"
          price="$4.99"
          tint={CYAN}
          copy="+500 coins, +2 hint credits, +2 undo credits, and No Ads unlock."
          cta="GET STARTER PACK"
          onPress={onBuyStarterPack}
        />
        <OfferCard
          title="RESTORE PURCHASE"
          price="FREE"
          tint={GOLD}
          copy="Restore previous purchases for returning players."
          cta="RESTORE NOW"
          onPress={onRestorePurchase}
        />

        <View style={styles.infoPanel}>
          <Text style={styles.infoTitle}>AD STRATEGY</Text>
          <Text style={styles.infoCopy}>Ads should appear after results or before special premium matches. Never interrupt the board in the middle of a turn.</Text>
        </View>

        <View style={styles.infoPanel}>
          <Text style={styles.infoTitle}>RESTORE PURCHASE</Text>
          <Text style={styles.infoCopy}>Keep a visible restore button in both Shop and Settings so returning players can recover premium access easily.</Text>
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
  backButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 8, borderWidth: 1, borderColor: LINE, borderRadius: 999, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  title: { color: WHITE, fontSize: 24, fontWeight: '900', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  offerCard: { backgroundColor: PANEL, borderWidth: 1, borderRadius: 14, padding: 12, gap: 8 },
  offerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  offerTitle: { fontSize: 14, fontWeight: '900', letterSpacing: 0.9 },
  offerPrice: { color: WHITE, fontSize: 16, fontWeight: '900' },
  offerCopy: { color: SOFT, fontSize: 11, lineHeight: 16 },
  offerButton: { minHeight: 42, borderWidth: 1, borderRadius: 999, backgroundColor: PANEL_DARK, alignItems: 'center', justifyContent: 'center' },
  offerButtonOwned: { opacity: 0.65 },
  offerButtonText: { fontSize: 12, fontWeight: '900', letterSpacing: 0.9 },
  infoPanel: { backgroundColor: PANEL_ALT, borderWidth: 1, borderRadius: 12, borderColor: LINE, padding: 12, gap: 4 },
  infoTitle: { color: MINT, fontSize: 12, fontWeight: '900', letterSpacing: 0.9 },
  infoCopy: { color: WHITE, fontSize: 11, lineHeight: 16 },
});
