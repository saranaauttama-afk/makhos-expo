// App.tsx — AI vs AI (กด AI Move เพื่อเดินทีละตา)
import React, { useMemo, useState } from 'react';
import { View, Text, Button, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Board } from './src/ui/Board';
import { initialPosition, Position } from './src/core/position';
import { generateMoves, applyMove } from './src/core/movegen';
import { useEngine } from './src/ui/useEngine';

export default function App() {
  const [pos, setPos] = useState<Position>(() => initialPosition());
  const { think, thinking, lastInfo } = useEngine();

  // คำนวณ moves ปัจจุบัน (ไว้เช็คแพ้/จบเกม และโชว์สถานะ)
  const myMoves = useMemo(() => generateMoves(pos), [pos]);

  function onNewGame() {
    setPos(initialPosition());
  }

  function onAIMove() {
    // จบเกมแล้ว
    if (myMoves.length === 0) {
      Alert.alert('Game over', pos.side === 1 ? 'P1 ไม่มีทางเดิน' : 'P2 ไม่มีทางเดิน');
      return;
    }
    const best = think(pos, 900); // ปรับเวลาได้ตามต้องการ
    if (!best) return;
    const next = applyMove(pos, best);
    setPos(next);
  }

  return (
    <SafeAreaView style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 }}>
      <Text style={{ fontSize: 20, fontWeight: '700' }}>Makhos (Thai Checkers) – AI vs AI</Text>
      <Text style={{ opacity: 0.85 }}>
        Turn: {pos.side === 1 ? 'P1' : 'P2'} {thinking ? '• thinking…' : ''}
        {myMoves.length === 0 ? ' • Game Over' : ''}
      </Text>
      {lastInfo && (
        <Text style={{ fontSize: 12, opacity: 0.7 }}>
          depth {lastInfo.depth} • score {lastInfo.score} • nodes {lastInfo.nodes}
        </Text>
      )}

      <Board
        pos={pos}
        // ปิดการแตะบนกระดาน/ตัวหมาก (AI เท่านั้น)
        onTapSquare={() => {}}
        // ส่ง props ให้ครบตาม Board ปัจจุบัน (แต่ไม่ใช้จริง)
        fromSquares={[]}
        selectedFrom={null}
        destSquares={[]}
      />

      <View style={{ flexDirection: 'row', gap: 12, marginTop: 10 }}>
        <Button title="New Game" onPress={onNewGame} />
        <Button title={thinking ? 'Thinking…' : 'AI Move'} onPress={onAIMove} />
      </View>
    </SafeAreaView>
  );
}
