import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface Props {
  onBack: () => void;
}

const BG = '#120c1c';
const PANEL = '#211638';
const PANEL_ALT = '#2c1f49';
const PANEL_DARK = '#0f0918';
const LINE = '#5d4d8a';
const GOLD = '#f3c969';
const MINT = '#77f7cf';
const CYAN = '#5ec5ff';
const PINK = '#ff7dc4';
const WHITE = '#f7f2ff';
const SOFT = '#b9abd8';

function OfferCard({
  title,
  price,
  tint,
  copy,
  cta,
}: {
  title: string;
  price: string;
  tint: string;
  copy: string;
  cta: string;
}) {
  return (
    <View style={[styles.offerCard, { borderColor: tint }]}>
      <View style={styles.offerTop}>
        <Text style={[styles.offerTitle, { color: tint }]}>{title}</Text>
        <Text style={styles.offerPrice}>{price}</Text>
      </View>
      <Text style={styles.offerCopy}>{copy}</Text>
      <Pressable style={[styles.offerButton, { borderColor: tint }]}>
        <Text style={[styles.offerButtonText, { color: tint }]}>{cta}</Text>
      </Pressable>
    </View>
  );
}

export default function ShopScreen({ onBack }: Props) {
  return (
    <SafeAreaView style={styles.safeArea}>
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
        />
        <OfferCard
          title="PREMIUM AI PASS"
          price="$4.99"
          tint={CYAN}
          copy="Unlock stronger AI tiers and future premium challenge ladders."
          cta="UNLOCK AI"
        />
        <OfferCard
          title="RETRO SUPPORTER"
          price="$6.99"
          tint={GOLD}
          copy="Bundle placeholder themes, no ads, and premium AI in one nostalgic supporter pack."
          cta="GET BUNDLE"
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
  scrollContent: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 24, gap: 16 },
  headerPanel: { backgroundColor: PANEL, borderWidth: 3, borderColor: LINE, padding: 14, gap: 10 },
  backButton: { alignSelf: 'flex-start', paddingHorizontal: 12, paddingVertical: 10, borderWidth: 2, borderColor: LINE, backgroundColor: PANEL_DARK },
  backButtonText: { color: WHITE, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 11, fontWeight: '800', letterSpacing: 1.3 },
  title: { color: WHITE, fontSize: 28, fontWeight: '900', letterSpacing: 1.1 },
  subtitle: { color: SOFT, fontSize: 13, lineHeight: 19 },
  offerCard: { backgroundColor: PANEL, borderWidth: 3, padding: 14, gap: 10 },
  offerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  offerTitle: { fontSize: 16, fontWeight: '900', letterSpacing: 1 },
  offerPrice: { color: WHITE, fontSize: 18, fontWeight: '900' },
  offerCopy: { color: SOFT, fontSize: 12, lineHeight: 18 },
  offerButton: { minHeight: 48, borderWidth: 2, backgroundColor: PANEL_DARK, alignItems: 'center', justifyContent: 'center' },
  offerButtonText: { fontSize: 13, fontWeight: '900', letterSpacing: 1 },
  infoPanel: { backgroundColor: PANEL_ALT, borderWidth: 2, borderColor: LINE, padding: 14, gap: 6 },
  infoTitle: { color: MINT, fontSize: 14, fontWeight: '900', letterSpacing: 1 },
  infoCopy: { color: WHITE, fontSize: 12, lineHeight: 18 },
});
