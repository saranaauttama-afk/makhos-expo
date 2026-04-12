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
  scrollContent: {
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 18,
    gap: 12,
  },
  headerPanel: {
    backgroundColor: PANEL,
    borderWidth: 3,
    borderColor: LINE,
    padding: 12,
    gap: 10,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderWidth: 2,
    borderColor: LINE,
    backgroundColor: PANEL_DARK,
  },
  backButtonText: {
    color: WHITE,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1,
  },
  headerTextBlock: {
    gap: 6,
  },
  kicker: {
    color: GOLD,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.3,
  },
  title: {
    color: WHITE,
    fontSize: 24,
    fontWeight: '900',
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
    borderWidth: 2,
    padding: 10,
    justifyContent: 'space-between',
  },
  statValue: {
    fontSize: 20,
    fontWeight: '900',
    letterSpacing: 1.2,
  },
  statTitle: {
    color: SOFT,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  block: {
    backgroundColor: PANEL,
    borderWidth: 3,
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
    fontWeight: '900',
    letterSpacing: 1,
  },
  blockCopy: {
    color: WHITE,
    fontSize: 11,
    lineHeight: 16,
  },
  roadmapPanel: {
    backgroundColor: PANEL_DARK,
    borderWidth: 3,
    borderColor: CYAN,
    padding: 12,
    gap: 8,
  },
  roadmapTitle: {
    color: CYAN,
    fontSize: 13,
    fontWeight: '900',
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
