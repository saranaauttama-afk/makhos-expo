import { Platform, Vibration } from 'react-native';
import { useCallback, useRef } from 'react';

export type PixelFxEvent = 'move' | 'capture' | 'combo' | 'promote' | 'victory';

const VIBRATION_PATTERNS: Record<PixelFxEvent, number | number[]> = {
  move: 12,
  capture: [0, 24, 18],
  combo: [0, 20, 18, 36],
  promote: [0, 28, 22, 48],
  victory: [0, 36, 24, 36, 24, 72],
};

export function usePixelGameFx(enabled = true) {
  const lastTriggerRef = useRef(0);

  const triggerFx = useCallback((event: PixelFxEvent) => {
    if (!enabled || Platform.OS === 'web') return;

    const now = Date.now();
    if (now - lastTriggerRef.current < 40) return;
    lastTriggerRef.current = now;

    Vibration.vibrate(VIBRATION_PATTERNS[event]);
  }, [enabled]);

  return { triggerFx };
}
