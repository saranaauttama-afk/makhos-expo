import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface Props {
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

function StatCard({ title, value, tint }: { title: string; value: string; tint: string }) {
  return (
    <View style={[styles.statCard, { borderColor: tint }]}>
      <Text style={[styles.statValue, { color: tint }]}>{value}</Text>
      <Text style={styles.statTitle}>{title}</Text>
    </View>
  );
}

function ConceptBlock({
  title,
  tint,
  lines,
}: {
  title: string;
  tint: string;
  lines: string[];
}) {
  return (
    <View style={styles.block}>
      <View style={styles.blockHead}>
        <View style={[styles.blockDot, { backgroundColor: tint }]} />
        <Text style={[styles.blockTitle, { color: tint }]}>{title}</Text>
      </View>
      {lines.map(line => (
        <Text key={line} style={styles.blockCopy}>
          {line}
        </Text>
      ))}
    </View>
  );
}

export default function ConceptScreen({ onBack }: Props) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View pointerEvents="none" style={styles.bgAuraLarge} />
      <View pointerEvents="none" style={styles.bgAuraSmall} />
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>BACK</Text>
          </Pressable>

          <View style={styles.headerTextBlock}>
            <Text style={styles.kicker}>PROJECT CONCEPT</Text>
            <Text style={styles.title}>MAKHOS STUDY PAGE</Text>
            <Text style={styles.subtitle}>
              A compact explanation page for the game concept, AI ladder, and ad model. Built as a learning screen inside the app.
            </Text>
          </View>
        </View>

        <View style={styles.statRow}>
          <StatCard title="CORE MODES" value="02" tint={GOLD} />
          <StatCard title="VISUAL LEVELS" value="05" tint={CYAN} />
          <StatCard title="PREMIUM GOAL" value="NO ADS" tint={PINK} />
        </View>

        <ConceptBlock
          title="WHAT THIS GAME IS"
          tint={GOLD}
          lines={[
            'Makhos is a Thai checkers game with a retro shell on top of a modern AI stack.',
            'The visual design can feel arcade and nostalgic, while the rules and move quality stay serious.',
          ]}
        />

        <ConceptBlock
          title="HOW THE AI LADDER WORKS"
          tint={MINT}
          lines={[
            'Free players can start with light AI tiers that respond quickly and feel fair on mobile.',
            'Higher tiers can unlock stronger hybrid and AZ-based play, either through rewarded ads or premium purchase.',
          ]}
        />

        <ConceptBlock
          title="WHY ADS FIT THIS PRODUCT"
          tint={CYAN}
          lines={[
            'Ads should live outside the board flow, mainly after a result screen or before a premium challenge.',
            'The board itself should stay clean, so players never feel interrupted in the middle of a tactical sequence.',
          ]}
        />

        <ConceptBlock
          title="PREMIUM VALUE"
          tint={PINK}
          lines={[
            'Premium should remove ads and unlock stronger AI modes permanently.',
            'That makes the advanced engine feel like a real feature upgrade, not just a hidden settings toggle.',
          ]}
        />

        <View style={styles.roadmapPanel}>
          <Text style={styles.roadmapTitle}>STUDY NOTES</Text>
          <View style={styles.roadmapList}>
            <Text style={styles.roadmapItem}>1. Build clean pixel-retro UI with placeholder art first.</Text>
            <Text style={styles.roadmapItem}>2. Ship fast AI tiers before exposing premium AI.</Text>
            <Text style={styles.roadmapItem}>3. Swap placeholder tiles with final sprite packs later.</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: BG,
  },
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
    paddingHorizontal: 18,
    paddingTop: 12,
    paddingBottom: 22,
    gap: 12,
  },
  headerPanel: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderRadius: 14,
    borderColor: LINE,
    padding: 12,
    gap: 10,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderWidth: 1,
    borderRadius: 999,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
  },
  backButtonText: {
    color: WHITE,
    fontSize: 11,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1,
  },
  headerTextBlock: {
    gap: 6,
  },
  kicker: {
    color: GOLD,
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1.3,
  },
  title: {
    color: WHITE,
    fontSize: 24,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1.2,
  },
  subtitle: {
    color: SOFT,
    fontSize: 12,
    lineHeight: 17,
  },
  statRow: {
    flexDirection: 'row',
    gap: 8,
  },
  statCard: {
    flex: 1,
    minHeight: 76,
    backgroundColor: PANEL_ALT,
    borderWidth: 1,
    borderRadius: 12,
    padding: 10,
    justifyContent: 'space-between',
  },
  statValue: {
    fontSize: 20,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1.2,
  },
  statTitle: {
    color: SOFT,
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.8,
  },
  block: {
    backgroundColor: PANEL,
    borderWidth: 1,
    borderRadius: 14,
    borderColor: LINE,
    padding: 12,
    gap: 8,
  },
  blockHead: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  blockDot: {
    width: 10,
    height: 10,
  },
  blockTitle: {
    fontSize: 13,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1,
  },
  blockCopy: {
    color: WHITE,
    fontSize: 11,
    lineHeight: 16,
  },
  roadmapPanel: {
    backgroundColor: PANEL_DARK,
    borderWidth: 1,
    borderRadius: 14,
    borderColor: CYAN,
    padding: 12,
    gap: 8,
  },
  roadmapTitle: {
    color: CYAN,
    fontSize: 13,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 1,
  },
  roadmapList: {
    gap: 6,
  },
  roadmapItem: {
    color: WHITE,
    fontSize: 11,
    lineHeight: 16,
  },
});

