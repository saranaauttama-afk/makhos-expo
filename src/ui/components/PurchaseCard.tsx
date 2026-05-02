import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

const PANEL_DARK = '#214b46';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const LINE = 'rgba(223, 247, 240, 0.34)';
const WHITE = '#f5f2e8';
const SOFT = '#d7efe8';

interface Props {
  title: string;
  subtitle?: string;
  price: string;
  copy: string;
  cta: string;
  tint: string;
  onPress: () => void;
  owned?: boolean;
  ownedLabel?: string;
}

export default function PurchaseCard({
  title,
  subtitle,
  price,
  copy,
  cta,
  tint,
  onPress,
  owned = false,
  ownedLabel = 'OWNED',
}: Props) {
  return (
    <View style={[styles.card, { borderColor: tint }]}>
      <View style={styles.top}>
        <View style={styles.topTitle}>
          <Text style={[styles.title, { color: tint }]}>{title}</Text>
          {!!subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
        </View>
        <Text style={styles.price}>{price}</Text>
      </View>
      <Text style={styles.copy}>{copy}</Text>
      <Pressable
        onPress={onPress}
        disabled={owned}
        style={[styles.button, { borderColor: tint }, owned && styles.buttonOwned]}
      >
        <Text style={[styles.buttonText, { color: tint }]}>{owned ? ownedLabel : cta}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderRadius: 12,
    backgroundColor: PANEL_DARK,
    padding: 10,
    gap: 6,
  },
  top: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
  },
  topTitle: {
    flex: 1,
    gap: 2,
  },
  title: {
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.7,
  },
  subtitle: {
    color: SOFT,
    fontSize: 10,
    fontFamily: 'Kanit_500Medium',
  },
  price: {
    color: WHITE,
    fontSize: 13,
    fontFamily: 'Kanit_800ExtraBold',
  },
  copy: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 16,
    fontFamily: 'Kanit_500Medium',
  },
  button: {
    minHeight: 36,
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: PANEL,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonOwned: {
    opacity: 0.6,
  },
  buttonText: {
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.7,
  },
});

