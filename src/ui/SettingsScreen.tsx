import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface Props {
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

function ActionRow({ title, tint }: { title: string; tint: string }) {
  return (
    <Pressable style={styles.actionRow}>
      <Text style={[styles.actionTitle, { color: tint }]}>{title}</Text>
      <Text style={styles.actionArrow}>OPEN</Text>
    </Pressable>
  );
}

export default function SettingsScreen({ onBack }: Props) {
  const [soundOn, setSoundOn] = useState(true);
  const [vibrationOn, setVibrationOn] = useState(true);
  const [retroFx, setRetroFx] = useState(true);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.headerPanel}>
          <Pressable style={styles.backButton} onPress={onBack}>
            <Text style={styles.backButtonText}>BACK</Text>
          </Pressable>
          <Text style={styles.kicker}>SYSTEM MENU</Text>
          <Text style={styles.title}>SETTINGS</Text>
          <Text style={styles.subtitle}>Skeleton control page for audio, accessibility, restore purchase, and legal links.</Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>PREFERENCES</Text>
          <ToggleRow title="Sound" subtitle="Arcade taps, move cues, and result stingers." active={soundOn} onToggle={() => setSoundOn(v => !v)} />
          <ToggleRow title="Vibration" subtitle="Light feedback on move confirm and result moments." active={vibrationOn} onToggle={() => setVibrationOn(v => !v)} />
          <ToggleRow title="Retro FX" subtitle="Decorative scanlines, bloom, and pixel overlays." active={retroFx} onToggle={() => setRetroFx(v => !v)} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>ACCOUNT / PURCHASE</Text>
          <ActionRow title="Restore Purchase" tint={CYAN} />
          <ActionRow title="Manage Premium Access" tint={GOLD} />
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>INFO</Text>
          <ActionRow title="Privacy Policy" tint={MINT} />
          <ActionRow title="Terms of Service" tint={CYAN} />
          <ActionRow title="About Makhos" tint={GOLD} />
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
  panel: { backgroundColor: PANEL, borderWidth: 3, borderColor: LINE, padding: 14, gap: 10 },
  sectionTitle: { color: WHITE, fontSize: 15, fontWeight: '900', letterSpacing: 1 },
  toggleRow: { backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: LINE, minHeight: 72, paddingHorizontal: 12, paddingVertical: 10, flexDirection: 'row', alignItems: 'center', gap: 12 },
  toggleTextBlock: { flex: 1, gap: 4 },
  toggleTitle: { color: WHITE, fontSize: 14, fontWeight: '800' },
  toggleSubtitle: { color: SOFT, fontSize: 12, lineHeight: 17 },
  togglePill: { width: 60, height: 30, borderWidth: 2, borderColor: LINE, backgroundColor: PANEL, padding: 3, justifyContent: 'center' },
  togglePillActive: { borderColor: MINT, backgroundColor: '#18302d' },
  toggleKnob: { width: 18, height: 18, backgroundColor: SOFT },
  toggleKnobActive: { backgroundColor: MINT, alignSelf: 'flex-end' },
  actionRow: { backgroundColor: PANEL_DARK, borderWidth: 2, borderColor: LINE, minHeight: 56, paddingHorizontal: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  actionTitle: { fontSize: 13, fontWeight: '800', letterSpacing: 0.8 },
  actionArrow: { color: SOFT, fontSize: 11, fontWeight: '900', letterSpacing: 0.8 },
});
