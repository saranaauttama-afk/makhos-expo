import { Platform } from 'react-native';
import type { AdConsentStatus, MonetizationState } from './types';

const STORAGE_KEY = 'makhos.wallet.v1';

type PersistedWallet = MonetizationState & {
  version: 1;
};

interface AsyncKeyValueStorage {
  getItem: (key: string) => Promise<string | null>;
  setItem: (key: string, value: string) => Promise<void>;
}

interface MaybeAsyncStorageModule {
  getItem?: unknown;
  setItem?: unknown;
}

let nativeStorageCache: AsyncKeyValueStorage | null | undefined;

function sanitizeNumber(value: unknown, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function sanitizeAdConsent(value: unknown, fallback: AdConsentStatus): AdConsentStatus {
  if (value === 'granted' || value === 'denied' || value === 'unknown') return value;
  return fallback;
}

function sanitizeBoolean(value: unknown, fallback: boolean) {
  return typeof value === 'boolean' ? value : fallback;
}

function coercePersistedWallet(input: unknown): PersistedWallet | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Partial<PersistedWallet>;
  return {
    version: 1,
    coins: sanitizeNumber(raw.coins, 100),
    hintCredits: sanitizeNumber(raw.hintCredits, 0),
    undoCredits: sanitizeNumber(raw.undoCredits, 0),
    noAdsUnlocked: Boolean(raw.noAdsUnlocked),
    adConsent: sanitizeAdConsent(raw.adConsent, 'unknown'),
    soundEnabled: sanitizeBoolean(raw.soundEnabled, true),
    vibrationEnabled: sanitizeBoolean(raw.vibrationEnabled, true),
    interstitialCounter: sanitizeNumber(raw.interstitialCounter, 0),
    interstitialSeen: sanitizeNumber(raw.interstitialSeen, 0),
    rewardedSeen: sanitizeNumber(raw.rewardedSeen, 0),
  };
}

function getWebStorage(): AsyncKeyValueStorage | null {
  try {
    const ls = globalThis.localStorage;
    if (!ls) return null;
    return {
      getItem: async (key: string) => ls.getItem(key),
      setItem: async (key: string, value: string) => {
        ls.setItem(key, value);
      },
    };
  } catch {
    return null;
  }
}

function getNativeStorage(): AsyncKeyValueStorage | null {
  if (nativeStorageCache !== undefined) return nativeStorageCache;
  try {
    // TODO(storage-sdk): install @react-native-async-storage/async-storage
    // for production persistence on iOS/Android if it is not already installed.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require('@react-native-async-storage/async-storage');
    const instance = (mod?.default ?? mod) as MaybeAsyncStorageModule | null | undefined;
    if (
      instance &&
      typeof instance.getItem === 'function' &&
      typeof instance.setItem === 'function'
    ) {
      nativeStorageCache = {
        getItem: instance.getItem as AsyncKeyValueStorage['getItem'],
        setItem: instance.setItem as AsyncKeyValueStorage['setItem'],
      };
    } else {
      nativeStorageCache = null;
    }
  } catch {
    nativeStorageCache = null;
  }
  return nativeStorageCache;
}

function getStorage(): AsyncKeyValueStorage | null {
  if (Platform.OS === 'web') return getWebStorage();
  return getNativeStorage();
}

export async function loadPersistedMonetization(): Promise<MonetizationState | null> {
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = await storage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    return coercePersistedWallet(parsed);
  } catch {
    return null;
  }
}

export async function savePersistedMonetization(state: MonetizationState): Promise<void> {
  const storage = getStorage();
  if (!storage) return;
  const payload: PersistedWallet = {
    version: 1,
    coins: state.coins,
    hintCredits: state.hintCredits,
    undoCredits: state.undoCredits,
    noAdsUnlocked: state.noAdsUnlocked,
    adConsent: state.adConsent,
    soundEnabled: state.soundEnabled,
    vibrationEnabled: state.vibrationEnabled,
    interstitialCounter: state.interstitialCounter,
    interstitialSeen: state.interstitialSeen,
    rewardedSeen: state.rewardedSeen,
  };
  await storage.setItem(STORAGE_KEY, JSON.stringify(payload));
}
