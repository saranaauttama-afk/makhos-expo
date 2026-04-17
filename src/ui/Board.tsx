// src/ui/Board.tsx
// Touch handling: Pressable grid overlay on top of SVG (no SVG onPress).
// SVG onPress with transparent fills is unreliable on Android, so the
// native Pressable grid remains the source of truth for input.
import React, { useEffect, useRef, useState } from 'react';
import { Pressable, View } from 'react-native';
import Svg, { Circle, G, Rect, Text as SvgText } from 'react-native-svg';
import { bits, toIndex, toRC } from '../coreClaude/bitboards';
import { Position } from '../coreClaude/position';
import { COLORS, SIZE } from './theme';

interface Dest { to: number; caps: number }
interface LastMoveHint {
  from: number;
  to: number;
  captured?: number;
  promote?: boolean;
}
interface Props {
  pos: Position;
  onTapSquare: (sq: number) => void;
  fromSquares: number[];
  selectedFrom: number | null;
  destSquares: Dest[];
  lastMove?: LastMoveHint | null;
}

const HILITE_FROM = '#8ecbc2';
const HILITE_SELECTED = '#d8f2eb';
const HILITE_DEST = '#b8a777';
const HILITE_LAST = '#84c8bd';
const PIECE_P1_SHADOW = 'rgba(24,28,38,0.45)';
const PIECE_P2_SHADOW = 'rgba(78,83,90,0.35)';
const PIECE_P1_BASE = '#4c515e';
const PIECE_P2_BASE = '#d9dde4';
const KING_FILL = '#f6e5b8';

export const Board: React.FC<Props> = ({ pos, onTapSquare, fromSquares, selectedFrom, destSquares, lastMove = null }) => {
  const N = 8;
  const S = SIZE.board / N;
  const [animFrame, setAnimFrame] = useState(0);
  const [moveAnimFrame, setMoveAnimFrame] = useState(0);
  const [moveAnim, setMoveAnim] = useState<{
    key: string;
    from: number;
    to: number;
    side: 1 | -1;
    king: boolean;
    progress: number;
  } | null>(null);
  const lastMoveKeyRef = useRef<string | null>(null);
  const moveStepRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const moveClearRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fromSet = new Set(fromSquares);
  const destSet = new Set(destSquares.map(d => d.to));
  const capsMap = new Map(destSquares.map(d => [d.to, d.caps]));
  const hasAnimatedHints = selectedFrom !== null || fromSquares.length > 0 || destSquares.length > 0;
  const moveKey = lastMove ? `${lastMove.from}-${lastMove.to}-${pos.side}` : null;

  const stopMoveAnimation = () => {
    if (moveStepRef.current !== null) {
      clearInterval(moveStepRef.current);
      moveStepRef.current = null;
    }
    if (moveClearRef.current !== null) {
      clearTimeout(moveClearRef.current);
      moveClearRef.current = null;
    }
  };

  useEffect(() => {
    if (!hasAnimatedHints) {
      setAnimFrame(0);
      return;
    }
    const id = setInterval(() => {
      setAnimFrame(frame => (frame + 1) % 6);
    }, 120);
    return () => clearInterval(id);
  }, [hasAnimatedHints]);

  useEffect(() => {
    const lm = lastMove;
    if (!moveKey || !lm) {
      setMoveAnimFrame(0);
      setMoveAnim(null);
      lastMoveKeyRef.current = null;
      stopMoveAnimation();
      return;
    }
    if (moveKey === lastMoveKeyRef.current) return;

    lastMoveKeyRef.current = moveKey;
    setMoveAnimFrame(0);
    let frame = 0;
    const id = setInterval(() => {
      frame += 1;
      setMoveAnimFrame(frame);
      if (frame >= 5) clearInterval(id);
    }, 85);

    stopMoveAnimation();
    const stepMs = 48;
    const totalSteps = 5;
    const toMask = (1 << lm.to) >>> 0;
    const p1HasTo = ((pos.p1Men | pos.p1Kings) & toMask) !== 0;
    const side: 1 | -1 = p1HasTo ? 1 : -1;
    const king = (((pos.p1Kings | pos.p2Kings) & toMask) !== 0) || Boolean(lm.promote);

    setMoveAnim({
      key: moveKey,
      from: lm.from,
      to: lm.to,
      side,
      king,
      progress: 0,
    });

    let step = 0;
    moveStepRef.current = setInterval(() => {
      step += 1;
      const raw = Math.min(1, step / totalSteps);
      const eased = 1 - (1 - raw) * (1 - raw) * (1 - raw);
      setMoveAnim(prev => (prev && prev.key === moveKey ? { ...prev, progress: eased } : prev));
      if (step >= totalSteps && moveStepRef.current) {
        clearInterval(moveStepRef.current);
        moveStepRef.current = null;
      }
    }, stepMs);
    moveClearRef.current = setTimeout(() => {
      setMoveAnim(prev => (prev && prev.key === moveKey ? null : prev));
      stopMoveAnimation();
    }, stepMs * totalSteps + 90);

    return () => clearInterval(id);
  }, [moveKey, lastMove, pos.p1Men, pos.p1Kings, pos.p2Men, pos.p2Kings]);

  useEffect(() => () => stopMoveAnimation(), []);

  const fromAlpha = [0.5, 0.65, 0.82, 1, 0.82, 0.65][animFrame];
  const selectedThickness = [3, 3, 4, 4, 3, 3][animFrame];
  const destSize = [0.28, 0.31, 0.34, 0.36, 0.34, 0.31][animFrame];
  const destAlpha = [0.55, 0.68, 0.82, 1, 0.82, 0.68][animFrame];
  const moveFlashAlpha = [0.95, 0.8, 0.62, 0.42, 0.24, 0.12][Math.min(moveAnimFrame, 5)];
  const burstOuter = [0.24, 0.36, 0.5, 0.62, 0.72, 0.82][Math.min(moveAnimFrame, 5)];
  const burstInner = [0.16, 0.24, 0.34, 0.42, 0.5, 0.56][Math.min(moveAnimFrame, 5)];
  const crownLift = [18, 12, 8, 5, 2, 0][Math.min(moveAnimFrame, 5)];
  const pieces = renderPieces(pos);
  const isCaptureMove = (lastMove?.captured ?? 0) > 0;
  const isPromoteMove = Boolean(lastMove?.promote);

  const renderPieceSprite = (
    x: number,
    y: number,
    side: 1 | -1,
    king: boolean,
    key: string
  ) => {
    const cx = x + S * 0.5;
    const cy = y + S * 0.52;
    const outerR = S * 0.31;
    const innerR = S * 0.24;
    const shineR = S * 0.12;
    const shadowFill = side === 1 ? PIECE_P1_SHADOW : PIECE_P2_SHADOW;
    const baseFill = side === 1 ? PIECE_P1_BASE : PIECE_P2_BASE;
    const innerFill = side === 1 ? '#646a78' : '#eceff5';
    const kingStroke = side === 1 ? '#2b2f38' : '#8d939d';
    return (
      <G key={key}>
        <Circle cx={cx + 1.6} cy={cy + 3.8} r={outerR} fill={shadowFill} />
        <Circle cx={cx} cy={cy} r={outerR} fill={baseFill} />
        <Circle cx={cx} cy={cy - S * 0.02} r={innerR} fill={innerFill} />
        <Circle cx={cx - S * 0.11} cy={cy - S * 0.13} r={shineR} fill="rgba(255,255,255,0.22)" />

        {king && (
          <>
            <Circle cx={cx} cy={cy} r={S * 0.11} fill={KING_FILL} />
            <Rect x={cx - S * 0.02} y={cy - S * 0.14} width={S * 0.04} height={S * 0.2} fill={kingStroke} />
            <Rect x={cx - S * 0.1} y={cy - S * 0.04} width={S * 0.2} height={S * 0.04} fill={kingStroke} />
          </>
        )}
      </G>
    );
  };

  return (
    <View style={{ width: SIZE.board, height: SIZE.board }}>
      <Svg width={SIZE.board} height={SIZE.board} style={{ position: 'absolute' }}>
        {Array.from({ length: 64 }, (_, k) => {
          const r = Math.floor(k / 8);
          const c = k % 8;
          const dark = ((r + c) & 1) === 1;
          const idx = dark ? toIndex(r, c) : -1;
          const isFrom = idx >= 0 && fromSet.has(idx);
          const isSelected = idx >= 0 && selectedFrom === idx;
          const isDest = idx >= 0 && destSet.has(idx);
          const isLastFrom = idx >= 0 && lastMove?.from === idx;
              const isLastTo = idx >= 0 && lastMove?.to === idx;
              const x = c * S;
              const y = r * S;
              const capCount = idx >= 0 ? (capsMap.get(idx) ?? 0) : 0;

              return (
                <G key={k}>
                  <Rect x={x} y={y} width={S} height={S} fill={dark ? COLORS.dark : COLORS.light} />

                  {isLastFrom && (
                    <Circle
                      cx={x + S * 0.5}
                      cy={y + S * 0.5}
                      r={S * 0.33}
                      fill={HILITE_LAST}
                      opacity={moveFlashAlpha * 0.38}
                    />
                  )}

                  {isLastTo && (
                    <Circle
                      cx={x + S * 0.5}
                      cy={y + S * 0.5}
                      r={S * 0.4}
                      fill={HILITE_SELECTED}
                      opacity={moveFlashAlpha * 0.45}
                    />
                  )}

                  {isLastTo && isCaptureMove && moveAnimFrame < 5 && (
                    <>
                      <Circle
                        cx={x + S * 0.5}
                        cy={y + S * 0.5}
                        r={S * burstOuter * 0.5}
                        fill="#f3d995"
                        opacity={moveFlashAlpha * 0.7}
                      />
                      <Circle
                        cx={x + S * 0.5}
                        cy={y + S * 0.5}
                        r={S * burstInner * 0.5}
                        fill="#fff2ca"
                        opacity={moveFlashAlpha}
                      />
                    </>
                  )}

                  {isFrom && (
                    <Circle
                      cx={x + S * 0.5}
                      cy={y + S * 0.5}
                      r={S * 0.31}
                      fill="none"
                      stroke={HILITE_FROM}
                      strokeWidth={3}
                      opacity={fromAlpha}
                    />
                  )}

                  {isSelected && (
                    <Circle
                      cx={x + S * 0.5}
                      cy={y + S * 0.5}
                      r={S * 0.34}
                      fill="none"
                      stroke={HILITE_SELECTED}
                      strokeWidth={selectedThickness + 1}
                      opacity={destAlpha}
                    />
                  )}

                  {isDest && (
                    <>
                      <Circle
                        cx={x + S * 0.5}
                        cy={y + S * 0.5}
                        r={S * destSize * 0.46}
                        fill={HILITE_DEST}
                        opacity={destAlpha}
                      />
                      {capCount > 1 && (
                        <>
                          <Circle
                            cx={x + S * 0.78}
                            cy={y + S * 0.22}
                            r={S * 0.12}
                            fill="#f4ead1"
                            opacity={0.95}
                          />
                          <SvgText
                            x={x + S * 0.78}
                            y={y + S * 0.26}
                            fontSize={S * 0.18}
                            fill="#5a4a29"
                            fontWeight="bold"
                            textAnchor="middle"
                          >
                            {capCount}
                          </SvgText>
                        </>
                      )}
                    </>
                  )}
                </G>
              );
        })}

        {pieces.map((p, i) => {
          const shouldHideAtDestination =
            !!moveAnim &&
            moveAnim.progress < 1 &&
            p.i === moveAnim.to &&
            p.side === moveAnim.side;
          if (shouldHideAtDestination) return null;

          const { r, c } = toRC(p.i);
          const x = c * S;
          const y = r * S;
          return renderPieceSprite(x, y, p.side, p.king, `p-${i}`);
        })}
        {moveAnim && moveAnim.progress < 1 && (() => {
          const from = toRC(moveAnim.from);
          const to = toRC(moveAnim.to);
          const lerpR = from.r + (to.r - from.r) * moveAnim.progress;
          const lerpC = from.c + (to.c - from.c) * moveAnim.progress;
          const hop = Math.sin(moveAnim.progress * Math.PI) * (S * 0.16);
          const x = lerpC * S;
          const y = lerpR * S - hop;
          return renderPieceSprite(x, y, moveAnim.side, moveAnim.king, `moving-${moveAnim.key}`);
        })()}
        {lastMove && isPromoteMove && moveAnimFrame < 5 && (() => {
          const to = toRC(lastMove.to);
          const x = to.c * S;
          const y = to.r * S;
          const crownSize = S * 0.3;
          const crownX = x + (S - crownSize) / 2;
          const crownY = y - crownLift;

          return (
            <G key={`crown-${moveKey}`}>
              <Rect x={crownX} y={crownY} width={crownSize} height={crownSize} fill="#7f5f1a" opacity={moveFlashAlpha * 0.7} />
              <Rect x={crownX + 3} y={crownY + 3} width={crownSize - 6} height={crownSize - 6} fill={KING_FILL} opacity={moveFlashAlpha} />
              <Rect x={crownX - 5} y={crownY + crownSize * 0.7} width={8} height={4} fill="#ffe17d" opacity={moveFlashAlpha} />
              <Rect x={crownX + crownSize - 3} y={crownY + crownSize * 0.7} width={8} height={4} fill="#ffe17d" opacity={moveFlashAlpha} />
            </G>
          );
        })()}
      </Svg>

      <View
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: SIZE.board,
          height: SIZE.board,
          flexDirection: 'row',
          flexWrap: 'wrap',
        }}
      >
        {Array.from({ length: 64 }, (_, k) => {
          const r = Math.floor(k / 8);
          const c = k % 8;
          const dark = ((r + c) & 1) === 1;
          const idx = dark ? toIndex(r, c) : -1;
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
  for (const i of bits(pos.p1Men)) out.push({ i, side: 1, king: false });
  for (const i of bits(pos.p1Kings)) out.push({ i, side: 1, king: true });
  for (const i of bits(pos.p2Men)) out.push({ i, side: -1, king: false });
  for (const i of bits(pos.p2Kings)) out.push({ i, side: -1, king: true });
  return out;
}
