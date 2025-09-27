// src/ui/Board.tsx
import React from 'react';
import { View } from 'react-native';
import Svg, { Rect, Circle, G, Text as SvgText } from 'react-native-svg';
import { COLORS, SIZE } from './theme';
import { Position } from '../core/position';
import { bits, toRC, toIndex } from '../core/bitboards';

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

  const fromSet = new Set(fromSquares);
  const destSet = new Set(destSquares.map(d => d.to));
  const capsMap = new Map(destSquares.map(d => [d.to, d.caps]));

  return (
    <View style={{ width: SIZE.board, height: SIZE.board }}>
      <Svg width={SIZE.board} height={SIZE.board}>
        {/* กระดาน (ช่อง) */}
        {Array.from({ length: 8 * 8 }, (_, k) => {
          const r = Math.floor(k / 8), c = k % 8;
          const dark = ((r + c) & 1) === 1;
          const idx = dark ? toIndex(r, c) : -1;

          const isFrom = idx >= 0 && fromSet.has(idx);
          const isSelected = idx >= 0 && selectedFrom === idx;
          const isDest = idx >= 0 && destSet.has(idx);

          const x = c * S, y = r * S;

          return (
            <G key={k}>
              {/* ช่องกระดาน + hit area โปร่งใส (ทำให้แตะช่องได้แม่น) */}
              <Rect x={x} y={y} width={S} height={S} fill={dark ? COLORS.dark : COLORS.light} />
              {idx >= 0 && (
                <Rect
                  x={x} y={y} width={S} height={S}
                  fill="rgba(0,0,0,0.001)" // โปร่งใส แต่มี fill เพื่อจับ onPress
                  onPress={() => onTapSquare(idx)}
                />
              )}

              {/* ไฮไลต์ */}
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
                    cx={x + S / 2}
                    cy={y + S / 2}
                    r={S * 0.18}
                    fill="none"
                    stroke="#16a34a"
                    strokeWidth={4}
                  />
                  {/* badge จำนวนตัวที่กิน */}
                  <Circle cx={x + S * 0.78} cy={y + S * 0.22} r={S * 0.14} fill="#16a34a" />
                  <SvgText
                    x={x + S * 0.78}
                    y={y + S * 0.22 + 4}
                    fontSize={S * 0.22}
                    fill="#fff"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {capsMap.get(idx) ?? 0}
                  </SvgText>
                </>
              )}
            </G>
          );
        })}

        {/* ชิ้นหมาก */}
        {renderPieces(pos).map((p, idx) => {
          const { r, c } = toRC(p.i);
          const cx = c * S + S / 2, cy = r * S + S / 2, rad = S * 0.38;
          return (
            <G key={`p-${idx}`}>
              {/* ตัวหมาก */}
              <Circle cx={cx} cy={cy} r={rad} fill={p.side === 1 ? COLORS.pieceP1 : COLORS.pieceP2} />
              {/* วง king */}
              {p.king && (
                <Circle cx={cx} cy={cy} r={rad * 0.65} fill="none" stroke={COLORS.kingRing} strokeWidth={3} />
              )}
              {/* hit area โปร่งใส (ใหญ่ขึ้นนิด) เพื่อให้แตะโดนง่าย + onPress ที่ชิ้นด้วย */}
              <Circle
                cx={cx} cy={cy} r={rad * 1.1}
                fill="rgba(0,0,0,0.001)"
                onPress={() => onTapSquare(p.i)}
              />
            </G>
          );
        })}
      </Svg>
    </View>
  );
};

function renderPieces(pos: Position) {
  const out: { i: number; side: 1|-1; king: boolean }[] = [];
  for (const i of bits(pos.p1Men)) out.push({ i, side: 1, king: false });
  for (const i of bits(pos.p1Kings)) out.push({ i, side: 1, king: true });
  for (const i of bits(pos.p2Men)) out.push({ i, side: -1, king: false });
  for (const i of bits(pos.p2Kings)) out.push({ i, side: -1, king: true });
  return out;
}
