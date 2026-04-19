import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const CARD_BG = '#214b46';
const LINE = 'rgba(223, 247, 240, 0.34)';
const GOLD = '#f6e2aa';
const SOFT = '#d7efe8';

interface Props {
  value: number;
  label: string;
  accent?: string;
}

export default function WalletStatCard({ value, label, accent = GOLD }: Props) {
  return (
    <View style={styles.card}>
      <Text style={[styles.value, { color: accent }]}>{value}</Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    minHeight: 64,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    backgroundColor: CARD_BG,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 2,
  },
  value: {
    fontSize: 17,
    fontWeight: '900',
    letterSpacing: 0.6,
  },
  label: {
    color: SOFT,
    fontSize: 10,
  },
});
