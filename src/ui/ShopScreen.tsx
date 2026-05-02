import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { AppLanguage } from '../../App';
import { MonetizationState } from './types';

interface Props {
  language: AppLanguage;
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

const COPY = {
  th: {
    back: 'BACK',
    kicker: 'PIXEL STORE',
    title: 'SHOP / NO ADS',
    subtitle: 'หน้าซื้อสินค้า: ของรางวัล, premium, และปลดโฆษณา',
    removeAdsTitle: 'REMOVE ADS',
    removeAdsPrice: '29 THB / $0.99',
    removeAdsCopy: 'ปิด interstitial ads และให้ flow การเล่นลื่นขึ้น',
    removeAdsCta: 'BUY NO ADS',
    starterTitle: 'STARTER PACK',
    starterPrice: '59 THB / $1.99',
    starterCopy: '+500 coins, +2 hint credits, +2 undo credits, และปลด No Ads',
    starterCta: 'GET STARTER PACK',
    restoreTitle: 'RESTORE PURCHASE',
    restorePrice: 'FREE',
    restoreCopy: 'กู้คืนสิทธิ์ซื้อเดิมสำหรับผู้เล่นที่กลับมา',
    restoreCta: 'RESTORE NOW',
    owned: 'OWNED',
    strategyTitle: 'AD STRATEGY',
    strategyCopy: 'ควรแสดงโฆษณาหลังจบแมตช์หรือก่อนเข้าโหมดพิเศษเท่านั้น ไม่ขัดจังหวะระหว่างเล่น',
    restoreGuideTitle: 'RESTORE PURCHASE',
    restoreGuideCopy: 'ควรมีปุ่มกู้คืนทั้งใน Shop และ Settings เพื่อให้ผู้เล่นกู้สิทธิ์ได้ง่าย',
  },
  en: {
    back: 'BACK',
    kicker: 'PIXEL STORE',
    title: 'SHOP / NO ADS',
    subtitle: 'Commerce screen for rewards, premium access, and ad removal.',
    removeAdsTitle: 'REMOVE ADS',
    removeAdsPrice: '29 THB / $0.99',
    removeAdsCopy: 'Turn off interstitial ads and keep board flow clean after each match.',
    removeAdsCta: 'BUY NO ADS',
    starterTitle: 'STARTER PACK',
    starterPrice: '59 THB / $1.99',
    starterCopy: '+500 coins, +2 hint credits, +2 undo credits, and No Ads unlock.',
    starterCta: 'GET STARTER PACK',
    restoreTitle: 'RESTORE PURCHASE',
    restorePrice: 'FREE',
    restoreCopy: 'Restore previous purchases for returning players.',
    restoreCta: 'RESTORE NOW',
    owned: 'OWNED',
    strategyTitle: 'AD STRATEGY',
    strategyCopy: 'Show ads after results or before premium modes. Never interrupt active board turns.',
    restoreGuideTitle: 'RESTORE PURCHASE',
    restoreGuideCopy: 'Keep a restore button in both Shop and Settings so returning players can recover premium access.',
  },
} as const;

function OfferCard({
  title,
  price,
  tint,
  copy,
  cta,
  onPress,
  owned = false,
  ownedLabel = 'OWNED',
}: {
  title: string;
  price: string;
  tint: string;
  copy: string;
  cta: string;
  onPress: () => void;
  owned?: boolean;
  ownedLabel?: string;
}) {
  return (
    <View style={[styles.offerCard, { borderColor: tint }]}>
      <View style={styles.offerTop}>
        <Text style={[styles.offerTitle, { color: tint }]}>{title}</Text>
        <Text style={styles.offerPrice}>{price}</Text>
      </View>
      <Text style={styles.offerCopy}>{copy}</Text>
      <Pressable style={[styles.offerButton, { borderColor: tint }, owned && styles.offerButtonOwned]} onPress={onPress}>
        <Text style={[styles.offerButtonText, { color: tint }]}>{owned ? ownedLabel : cta}</Text>
      </Pressable>
    </View>
  );
}

export default function ShopScreen({ language, monetization, onBuyNoAds, onBuyStarterPack, onRestorePurchase, onBack }: Props) {
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
          <Text style={styles.kicker}>{t.kicker}</Text>
          <Text style={styles.title}>{t.title}</Text>
          <Text style={styles.subtitle}>{t.subtitle}</Text>
        </View>

        <OfferCard
          title={t.removeAdsTitle}
          price={t.removeAdsPrice}
          tint={PINK}
          copy={t.removeAdsCopy}
          cta={t.removeAdsCta}
          onPress={onBuyNoAds}
          owned={monetization.noAdsUnlocked}
          ownedLabel={t.owned}
        />
        <OfferCard
          title={t.starterTitle}
          price={t.starterPrice}
          tint={CYAN}
          copy={t.starterCopy}
          cta={t.starterCta}
          onPress={onBuyStarterPack}
        />
        <OfferCard
          title={t.restoreTitle}
          price={t.restorePrice}
          tint={GOLD}
          copy={t.restoreCopy}
          cta={t.restoreCta}
          onPress={onRestorePurchase}
        />

        <View style={styles.infoPanel}>
          <Text style={styles.infoTitle}>{t.strategyTitle}</Text>
          <Text style={styles.infoCopy}>{t.strategyCopy}</Text>
        </View>

        <View style={styles.infoPanel}>
          <Text style={styles.infoTitle}>{t.restoreGuideTitle}</Text>
          <Text style={styles.infoCopy}>{t.restoreGuideCopy}</Text>
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
  backButtonText: { color: WHITE, fontSize: 11, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 1 },
  kicker: { color: GOLD, fontSize: 10, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 1.1 },
  title: { color: WHITE, fontSize: 24, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 1 },
  subtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  offerCard: { backgroundColor: PANEL, borderWidth: 1, borderRadius: 14, padding: 12, gap: 8 },
  offerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  offerTitle: { fontSize: 14, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.9 },
  offerPrice: { color: WHITE, fontSize: 16, fontFamily: 'Kanit_800ExtraBold' },
  offerCopy: { color: SOFT, fontSize: 11, lineHeight: 16 },
  offerButton: { minHeight: 42, borderWidth: 1, borderRadius: 999, backgroundColor: PANEL_DARK, alignItems: 'center', justifyContent: 'center' },
  offerButtonOwned: { opacity: 0.65 },
  offerButtonText: { fontSize: 12, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.9 },
  infoPanel: { backgroundColor: PANEL_ALT, borderWidth: 1, borderRadius: 12, borderColor: LINE, padding: 12, gap: 4 },
  infoTitle: { color: MINT, fontSize: 12, fontFamily: 'Kanit_800ExtraBold', letterSpacing: 0.9 },
  infoCopy: { color: WHITE, fontSize: 11, lineHeight: 16 },
});

