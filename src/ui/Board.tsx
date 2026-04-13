// src/ui/Board.tsx
// Touch handling: Pressable grid overlay on top of SVG (no SVG onPress).
// SVG onPress with transparent fills is unreliable on Android, so the
// native Pressable grid remains the source of truth for input.
import React, { useEffect, useRef, useState } from 'react';
import { Pressable, View } from 'react-native';
import Svg, { G, Rect, Text as SvgText } from 'react-native-svg';
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

const HILITE_FROM = '#5ec5ff';
const HILITE_SELECTED = '#f3c969';
const HILITE_DEST = '#77f7cf';
const HILITE_DEST_DARK = '#1a7f62';
const PIECE_P1_SHADOW = '#171126';
const PIECE_P2_SHADOW = '#7a2430';
const KING_FILL = '#fff3ba';

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
  const selectedInset = [8, 7, 6, 7, 8, 9][animFrame];
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
    const pad = S * 0.16;
    const body = S - pad * 2;
    const inset = S * 0.08;
    const shadowFill = side === 1 ? PIECE_P1_SHADOW : PIECE_P2_SHADOW;
    const baseFill = side === 1 ? COLORS.pieceP1 : COLORS.pieceP2;
    return (
      <G key={key}>
        <Rect x={x + pad + 2} y={y + pad + 4} width={body} height={body} fill={shadowFill} />
        <Rect x={x + pad} y={y + pad} width={body} height={body} fill={baseFill} />
        <Rect x={x + pad + 4} y={y + pad + 4} width={body - 8} height={body - 8} fill="rgba(255,255,255,0.12)" />

        {king && (
          <>
            <Rect x={x + pad + inset + 2} y={y + pad + inset + 2} width={body - inset * 2} height={body - inset * 2} fill="#7f5f1a" />
            <Rect x={x + pad + inset} y={y + pad + inset} width={body - inset * 2} height={body - inset * 2} fill={KING_FILL} />
            <Rect x={x + pad + inset + 6} y={y + pad + inset + 6} width={body - inset * 2 - 12} height={body - inset * 2 - 12} fill="rgba(0,0,0,0.10)" />
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

          return (
            <G key={k}>
              <Rect x={x} y={y} width={S} height={S} fill={dark ? COLORS.dark : COLORS.light} />
              <Rect x={x} y={y} width={S} height={2} fill="rgba(255,255,255,0.12)" />
              <Rect x={x} y={y + S - 2} width={S} height={2} fill="rgba(0,0,0,0.18)" />

              {isLastFrom && (
                <Rect
                  x={x + 5}
                  y={y + 5}
                  width={S - 10}
                  height={S - 10}
                  fill="#23143d"
                  opacity={moveFlashAlpha}
                />
              )}

              {isLastTo && (
                <>
                  <Rect
                    x={x + 3}
                    y={y + 3}
                    width={S - 6}
                    height={S - 6}
                    fill={HILITE_SELECTED}
                    opacity={moveFlashAlpha}
                  />
                  <Rect
                    x={x + 8}
                    y={y + 8}
                    width={S - 16}
                    height={S - 16}
                    fill="#fff2b1"
                    opacity={moveFlashAlpha * 0.8}
                  />
                </>
              )}

              {isLastTo && isCaptureMove && moveAnimFrame < 5 && (
                <>
                  <Rect
                    x={x + (S - S * burstOuter) / 2}
                    y={y + (S - S * burstOuter) / 2}
                    width={S * burstOuter}
                    height={S * burstOuter}
                    fill="#ff7dc4"
                    opacity={moveFlashAlpha * 0.7}
                  />
                  <Rect
                    x={x + (S - S * burstInner) / 2}
                    y={y + (S - S * burstInner) / 2}
                    width={S * burstInner}
                    height={S * burstInner}
                    fill="#ffe17d"
                    opacity={moveFlashAlpha}
                  />
                  <Rect x={x + S * 0.08} y={y + S * 0.46} width={S * 0.18} height={4} fill="#ffe17d" opacity={moveFlashAlpha} />
                  <Rect x={x + S * 0.74} y={y + S * 0.46} width={S * 0.18} height={4} fill="#ffe17d" opacity={moveFlashAlpha} />
                  <Rect x={x + S * 0.46} y={y + S * 0.08} width={4} height={S * 0.18} fill="#ffe17d" opacity={moveFlashAlpha} />
                  <Rect x={x + S * 0.46} y={y + S * 0.74} width={4} height={S * 0.18} fill="#ffe17d" opacity={moveFlashAlpha} />
                </>
              )}

              {isFrom && (
                <>
                  <Rect x={x + 2} y={y + 2} width={S - 4} height={4} fill={HILITE_FROM} opacity={fromAlpha} />
                  <Rect x={x + 2} y={y + S - 6} width={S - 4} height={4} fill={HILITE_FROM} opacity={fromAlpha} />
                  <Rect x={x + 2} y={y + 2} width={4} height={S - 4} fill={HILITE_FROM} opacity={fromAlpha} />
                  <Rect x={x + S - 6} y={y + 2} width={4} height={S - 4} fill={HILITE_FROM} opacity={fromAlpha} />
                </>
              )}

              {isSelected && (
                <>
                  <Rect
                    x={x + selectedInset}
                    y={y + selectedInset}
                    width={S - selectedInset * 2}
                    height={selectedThickness}
                    fill={HILITE_SELECTED}
                  />
                  <Rect
                    x={x + selectedInset}
                    y={y + S - selectedInset - selectedThickness}
                    width={S - selectedInset * 2}
                    height={selectedThickness}
                    fill={HILITE_SELECTED}
                  />
                  <Rect
                    x={x + selectedInset}
                    y={y + selectedInset}
                    width={selectedThickness}
                    height={S - selectedInset * 2}
                    fill={HILITE_SELECTED}
                  />
                  <Rect
                    x={x + S - selectedInset - selectedThickness}
                    y={y + selectedInset}
                    width={selectedThickness}
                    height={S - selectedInset * 2}
                    fill={HILITE_SELECTED}
                  />
                  <Rect
                    x={x + selectedInset + 5}
                    y={y + selectedInset + 5}
                    width={S - (selectedInset + 5) * 2}
                    height={2}
                    fill="#fff2b1"
                    opacity={destAlpha}
                  />
                </>
              )}

              {isDest && (
                <>
                  <Rect
                    x={x + (S - S * destSize) / 2}
                    y={y + (S - S * destSize) / 2}
                    width={S * destSize}
                    height={S * destSize}
                    fill={HILITE_DEST}
                    opacity={destAlpha}
                  />
                  <Rect
                    x={x + (S - S * (destSize - 0.05)) / 2}
                    y={y + (S - S * (destSize - 0.05)) / 2}
                    width={S * (destSize - 0.05)}
                    height={S * (destSize - 0.05)}
                    fill={HILITE_DEST_DARK}
                    opacity={0.35}
                  />
                  <Rect x={x + S * 0.70} y={y + S * 0.10} width={S * 0.20} height={S * 0.20} fill={HILITE_DEST} />
                  <SvgText
                    x={x + S * 0.80}
                    y={y + S * 0.27}
                    fontSize={S * 0.20}
                    fill="#0f0918"
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
