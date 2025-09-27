// app/index.tsx
import React, { useMemo, useState } from 'react';
import { SafeAreaView, View, Text, Button, Alert } from 'react-native';
import { Board } from '../src/ui/Board';
import { initialPosition, Position } from '../src/core/position';
import { generateMoves, applyMove, Move } from '../src/core/movegen';
import { useEngine } from '../src/ui/useEngine';

export default function App() {
  const [pos, setPos] = useState<Position>(() => initialPosition());
  const [sel, setSel] = useState<number | null>(null);
  const [legal, setLegal] = useState<Move[]>([]);
  const { think, thinking } = useEngine();

  const myMoves = useMemo(() => generateMoves(pos), [pos]);

  function onTapSquare(i: number) {
    const candidates = myMoves.filter(m => m.from === (sel ?? i));
    const to = candidates.find(m => m.to === i);
    if (sel === null) {
      if (candidates.length) setSel(i);
    } else if (to) {
      const next = applyMove(pos, to);
      setPos(next); setSel(null);
      // If vs AI and side switched to AI, think
      if (next.side === -1) {
        setTimeout(() => {
          const best = think(next, 600);
          if (best) setPos(applyMove(next, best));
        }, 0);
      }
    } else {
      setSel(null);
    }
  }

  const highlights = sel !== null ? myMoves.filter(m => m.from === sel).map(m => m.to) : [];

  return (
    <SafeAreaView style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 }}>
      <Text style={{ fontSize: 20, fontWeight: '600' }}>Makhos (Thai Checkers)</Text>
      <Board pos={pos} onTapSquare={onTapSquare} highlights={highlights} />
      <View style={{ flexDirection: 'row', gap: 12, marginTop: 8 }}>
        <Button title="New Game" onPress={() => { setPos(initialPosition()); setSel(null); }} />
        <Button title={thinking ? 'Thinking…' : 'AI Move'} onPress={() => {
          const best = think(pos, 600);
          if (best) setPos(applyMove(pos, best));
        }} />
      </View>
    </SafeAreaView>
  );
}