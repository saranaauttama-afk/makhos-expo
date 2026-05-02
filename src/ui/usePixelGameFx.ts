import { Platform, Vibration } from 'react-native';
import { useCallback, useRef } from 'react';
import { Audio } from 'expo-av';

export type PixelFxEvent = 'move' | 'capture' | 'combo' | 'promote' | 'victory';
type PixelFxOptions = {
  enabled?: boolean;
  soundEnabled?: boolean;
  vibrationEnabled?: boolean;
};

const VIBRATION_PATTERNS: Record<PixelFxEvent, number | number[]> = {
  move: 12,
  capture: [0, 24, 18],
  combo: [0, 20, 18, 36],
  promote: [0, 28, 22, 48],
  victory: [0, 36, 24, 36, 24, 72],
};

const SOUND_SOURCES: Record<PixelFxEvent, number> = {
  move: require('../../assets/sfx/move.wav'),
  capture: require('../../assets/sfx/capture.wav'),
  combo: require('../../assets/sfx/capture.wav'),
  promote: require('../../assets/sfx/capture.wav'),
  victory: require('../../assets/sfx/victory.wav'),
};

const SOUND_VOLUME: Partial<Record<PixelFxEvent, number>> = {
  move: 0.3,
  capture: 0.42,
  combo: 0.45,
  promote: 0.45,
  victory: 0.48,
};

let audioModeReady = false;
let loadPromise: Promise<void> | null = null;
const soundCache = new Map<PixelFxEvent, Audio.Sound>();

async function ensureAudioLoaded() {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    if (!audioModeReady) {
      await Audio.setAudioModeAsync({
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
      });
      audioModeReady = true;
    }
    for (const event of Object.keys(SOUND_SOURCES) as PixelFxEvent[]) {
      if (soundCache.has(event)) continue;
      const { sound } = await Audio.Sound.createAsync(SOUND_SOURCES[event], {
        shouldPlay: false,
        volume: SOUND_VOLUME[event] ?? 0.4,
      });
      soundCache.set(event, sound);
    }
  })().catch(() => {
    loadPromise = null;
  });
  return loadPromise;
}

async function playFxSound(event: PixelFxEvent) {
  try {
    await ensureAudioLoaded();
    const sound = soundCache.get(event);
    if (!sound) return;
    await sound.replayAsync();
  } catch {
    // no-op: keep gameplay smooth even if audio fails.
  }
}

export function usePixelGameFx(options?: PixelFxOptions) {
  const enabled = options?.enabled ?? true;
  const soundEnabled = options?.soundEnabled ?? true;
  const vibrationEnabled = options?.vibrationEnabled ?? true;
  const lastTriggerRef = useRef(0);

  const triggerFx = useCallback((event: PixelFxEvent) => {
    if (!enabled) return;

    const now = Date.now();
    if (now - lastTriggerRef.current < 40) return;
    lastTriggerRef.current = now;

    if (vibrationEnabled && Platform.OS !== 'web') {
      Vibration.vibrate(VIBRATION_PATTERNS[event]);
    }
    if (soundEnabled) void playFxSound(event);
  }, [enabled, soundEnabled, vibrationEnabled]);

  return { triggerFx };
}
