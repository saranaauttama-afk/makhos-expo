import React from 'react';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';

const SHEET = 'rgba(27, 69, 64, 0.96)';
const BG = 'rgba(12, 33, 31, 0.45)';
const LINE = 'rgba(223, 247, 240, 0.34)';
const PANEL_DARK = '#214b46';
const GOLD = '#f6e2aa';
const CYAN = '#9be7da';
const SOFT = '#d7efe8';
const WHITE = '#f5f2e8';

interface Props {
  visible: boolean;
  title: string;
  helperText?: string;
  cost: number;
  canSpendCoins: boolean;
  onSpendCoins: () => void;
  onWatchAd: () => void;
  onCancel: () => void;
  spendLabel?: string;
  watchAdLabel?: string;
  cancelLabel?: string;
}

export default function SpendOrWatchAdModal({
  visible,
  title,
  helperText,
  cost,
  canSpendCoins,
  onSpendCoins,
  onWatchAd,
  onCancel,
  spendLabel,
  watchAdLabel,
  cancelLabel,
}: Props) {
  const spendText = spendLabel ?? `Spend ${cost} coins`;
  const watchText = watchAdLabel ?? 'Watch Ad for free';
  const cancelText = cancelLabel ?? 'Cancel';
  return (
    <Modal visible={visible} animationType="fade" transparent onRequestClose={onCancel}>
      <View style={styles.backdrop}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onCancel} />
        <View style={styles.sheet}>
          <Text style={styles.title}>{title}</Text>
          {!!helperText && <Text style={styles.helper}>{helperText}</Text>}
          {canSpendCoins ? (
            <Pressable style={[styles.actionBtn, styles.coinBtn]} onPress={onSpendCoins}>
              <Text style={[styles.actionText, { color: GOLD }]}>{spendText}</Text>
            </Pressable>
          ) : null}
          <Pressable style={[styles.actionBtn, styles.adBtn]} onPress={onWatchAd}>
            <Text style={[styles.actionText, { color: CYAN }]}>{watchText}</Text>
          </Pressable>
          <Pressable style={[styles.actionBtn, styles.cancelBtn]} onPress={onCancel}>
            <Text style={[styles.actionText, { color: SOFT }]}>{cancelText}</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: BG,
    justifyContent: 'flex-end',
    padding: 14,
  },
  sheet: {
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 14,
    backgroundColor: SHEET,
    padding: 12,
    gap: 8,
  },
  title: {
    color: WHITE,
    fontSize: 15,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.5,
  },
  helper: {
    color: SOFT,
    fontSize: 11,
    lineHeight: 16,
    marginBottom: 2,
  },
  actionBtn: {
    minHeight: 40,
    borderWidth: 1,
    borderColor: LINE,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: PANEL_DARK,
  },
  coinBtn: {
    borderColor: 'rgba(246, 226, 170, 0.6)',
  },
  adBtn: {
    borderColor: 'rgba(155, 231, 218, 0.6)',
  },
  cancelBtn: {
    opacity: 0.88,
  },
  actionText: {
    fontSize: 12,
    fontFamily: 'Kanit_800ExtraBold',
    letterSpacing: 0.5,
  },
});

