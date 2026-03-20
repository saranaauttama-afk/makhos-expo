// src/ui/Board.tsx
// Touch handling: Pressable grid overlay on top of SVG (no SVG onPress).
// SVG onPress with fill="rgba(0,0,0,0.001)" is unreliable on Android —
// a native Pressable grid is rock-solid on all platforms.
import React from 'react';
import { Pressable, View } from 'react-native';
import Svg, { Circle, G, Rect, Text as SvgText } from 'react-native-svg';
import { bits, toIndex, toRC } from '../coreClaude/bitboards';
import { Position } from '../coreClaude/position';
import { COLORS, SIZE } from './theme';

interface Dest { to: number; caps: number }
interface Props {
  pos: Position;
  onTapSquare: (sq: number) => void;
  fromSquares: number[];
  selectedFrom: number | null;
  destSquares: Dest[];
}

export const Board: React.FC<Props> = ({ pos, onTapSquare, fromSquares, selectedFrom, destSquares }) => {
  const N = 8;
  const S = SIZE.board / N;

  const fromSet  = new Set(fromSquares);
  const destSet  = new Set(destSquares.map(d => d.to));
  const capsMap  = new Map(destSquares.map(d => [d.to, d.caps]));

  return (
    <View style={{ width: SIZE.board, height: SIZE.board }}>

      {/* ── Visuals (SVG, no touch handlers) ─────────────────────────────── */}
      <Svg width={SIZE.board} height={SIZE.board} style={{ position: 'absolute' }}>

        {/* Board squares + highlights */}
        {Array.from({ length: 64 }, (_, k) => {
          const r = Math.floor(k / 8), c = k % 8;
          const dark = ((r + c) & 1) === 1;
          const idx  = dark ? toIndex(r, c) : -1;

          const isFrom     = idx >= 0 && fromSet.has(idx);
          const isSelected = idx >= 0 && selectedFrom === idx;
          const isDest     = idx >= 0 && destSet.has(idx);
          const x = c * S, y = r * S;

          return (
            <G key={k}>
              <Rect x={x} y={y} width={S} height={S} fill={dark ? COLORS.dark : COLORS.light} />

              {isFrom && (
                <Rect
                  x={x + 3} y={y + 3} width={S - 6} height={S - 6}
                  fill="none" stroke="#3b82f6" strokeWidth={3}
                  opacity={isSelected ? 1 : 0.85}
                />
              )}
              {isSelected && (
                <Rect
                  x={x + 7} y={y + 7} width={S - 14} height={S - 14}
                  fill="none" stroke="#1d4ed8" strokeWidth={3}
                />
              )}
              {isDest && (
                <>
                  <Circle
                    cx={x + S / 2} cy={y + S / 2} r={S * 0.18}
                    fill="none" stroke="#16a34a" strokeWidth={4}
                  />
                  <Circle cx={x + S * 0.78} cy={y + S * 0.22} r={S * 0.14} fill="#16a34a" />
                  <SvgText
                    x={x + S * 0.78} y={y + S * 0.22 + 4}
                    fontSize={S * 0.22} fill="#fff" fontWeight="bold" textAnchor="middle"
                  >
                    {capsMap.get(idx) ?? 0}
                  </SvgText>
                </>
              )}
            </G>
          );
        })}

        {/* Pieces */}
        {renderPieces(pos).map((p, i) => {
          const { r, c } = toRC(p.i);
          const cx = c * S + S / 2, cy = r * S + S / 2, rad = S * 0.38;
          return (
            <G key={`p-${i}`}>
              <Circle cx={cx} cy={cy} r={rad} fill={p.side === 1 ? COLORS.pieceP1 : COLORS.pieceP2} />
              {p.king && (
                <Circle cx={cx} cy={cy} r={rad * 0.65} fill="none" stroke={COLORS.kingRing} strokeWidth={3} />
              )}
            </G>
          );
        })}
      </Svg>

      {/* ── Touch overlay: 8×8 Pressable grid (absolute, covers entire board) ── */}
      <View
        style={{
          position: 'absolute', top: 0, left: 0,
          width: SIZE.board, height: SIZE.board,
          flexDirection: 'row', flexWrap: 'wrap',
        }}
      >
        {Array.from({ length: 64 }, (_, k) => {
          const r = Math.floor(k / 8), c = k % 8;
          const dark = ((r + c) & 1) === 1;
          const idx  = dark ? toIndex(r, c) : -1;
          return (
            <Pressable
              key={k}
              style={{ width: S, height: S }}
              onPress={() => idx >= 0 && onTapSquare(idx)}
            />
          );
        })}
      </View>

    </View>
  );
};

function renderPieces(pos: Position) {
  const out: { i: number; side: 1 | -1; king: boolean }[] = [];
  for (const i of bits(pos.p1Men))   out.push({ i, side: 1,  king: false });
  for (const i of bits(pos.p1Kings)) out.push({ i, side: 1,  king: true  });
  for (const i of bits(pos.p2Men))   out.push({ i, side: -1, king: false });
  for (const i of bits(pos.p2Kings)) out.push({ i, side: -1, king: true  });
  return out;
}
