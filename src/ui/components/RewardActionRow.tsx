import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

const PANEL_DARK = '#214b46';
const PANEL = 'rgba(27, 69, 64, 0.74)';
const LINE = 'rgba(223, 247, 240, 0.34)';
const SOFT = '#d7efe8';
const WHITE = '#f5f2e8';

interface Props {
  title: string;
  subtitle: string;
  cta: string;
  tint: string;
  onPress: () => void;
  disabled?: boolean;
}

export default function RewardActionRow({
  title,
  subtitle,
  cta,
  tint,
  onPress,
  disabled = false,
}: Props) {
  return (
    <View style={styles.row}>
      <View style={styles.copy}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.subtitle}>{subtitle}</Text>
      </View>
      <Pressable
        onPress={onPress}
        disabled={disabled}
        style={[styles.button, { borderColor: tint }, disabled && styles.buttonDisabled]}
      >
        <Text style={[styles.buttonText, { color: tint }]}>{cta}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    minHeight: 56,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 12,
    backgroundColor: PANEL_DARK,
    paddingHorizontal: 10,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  copy: {
    flex: 1,
    gap: 2,
  },
  title: {
    color: WHITE,
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
  },
  subtitle: {
    color: SOFT,
    fontSize: 10,
    lineHeight: 14,
    fontFamily: 'Kanit_500Medium',
  },
  button: {
    minHeight: 30,
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: PANEL,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: 10,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.6,
  },
});

