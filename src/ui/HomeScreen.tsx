import React, { useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Difficulty, GameConfig, GameMode } from './types';

interface Props {
  onStart: (config: GameConfig) => void;
}

function ChoiceBtn({ title, active, onPress }: { title: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.choiceBtn, active && styles.choiceBtnActive]}>
      <Text style={[styles.choiceText, active && styles.choiceTextActive]}>{title}</Text>
    </Pressable>
  );
}

export default function HomeScreen({ onStart }: Props) {
  const [mode, setMode] = useState<GameMode>('vs-ai');
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>หมากหัว</Text>
        <Text style={styles.subtitle}>Makhos — Thai Checkers</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.label}>โหมดเกม</Text>
        <View style={styles.row}>
          <ChoiceBtn title="🤖  เล่นกับ AI" active={mode === 'vs-ai'} onPress={() => setMode('vs-ai')} />
          <ChoiceBtn title="👥  เล่นกับเพื่อน" active={mode === 'vs-human'} onPress={() => setMode('vs-human')} />
        </View>
      </View>

      {mode === 'vs-ai' && (
        <View style={styles.section}>
          <Text style={styles.label}>ระดับความยาก</Text>
          <View style={styles.row}>
            <ChoiceBtn title="ง่าย" active={difficulty === 'easy'} onPress={() => setDifficulty('easy')} />
            <ChoiceBtn title="กลาง" active={difficulty === 'medium'} onPress={() => setDifficulty('medium')} />
            <ChoiceBtn title="ยาก" active={difficulty === 'hard'} onPress={() => setDifficulty('hard')} />
          </View>
        </View>
      )}

      <Pressable style={styles.startBtn} onPress={() => onStart({ mode, difficulty, humanSide: 1 })}>
        <Text style={styles.startText}>▶  เริ่มเกม</Text>
      </Pressable>
    </SafeAreaView>
  );
}

const ACCENT = '#55aa33';

const styles = StyleSheet.create({
  container:        { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 28, paddingHorizontal: 24 },
  header:           { alignItems: 'center', gap: 4 },
  title:            { fontSize: 42, fontWeight: '800', letterSpacing: 1 },
  subtitle:         { fontSize: 15, opacity: 0.5 },
  section:          { width: '100%', gap: 10 },
  label:            { fontSize: 12, fontWeight: '700', opacity: 0.45, textTransform: 'uppercase', letterSpacing: 1.2 },
  row:              { flexDirection: 'row', gap: 8 },
  choiceBtn:        { flex: 1, paddingVertical: 13, borderRadius: 12, borderWidth: 2, borderColor: '#ddd', alignItems: 'center' },
  choiceBtnActive:  { borderColor: ACCENT, backgroundColor: 'rgba(85,170,51,0.1)' },
  choiceText:       { fontSize: 14, fontWeight: '600', color: '#888' },
  choiceTextActive: { color: ACCENT },
  startBtn:         { marginTop: 4, backgroundColor: ACCENT, borderRadius: 14, paddingVertical: 16, paddingHorizontal: 52 },
  startText:        { fontSize: 18, fontWeight: '700', color: '#fff', letterSpacing: 0.5 },
});
