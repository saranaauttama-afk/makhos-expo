import { Platform } from 'react-native';
import { GameConfig } from './types';

export type PersistedLanguage = 'th' | 'en';

export interface AppPreferences {
  language: PersistedLanguage;
  lastConfig: GameConfig;
}

type PersistedPreferences = AppPreferences & { version: 1 };

interface AsyncKeyValueStorage {
  getItem: (key: string) => Promise<string | null>;
  setItem: (key: string, value: string) => Promise<void>;
}

interface MaybeAsyncStorageModule {
  getItem?: unknown;
  setItem?: unknown;
}

const STORAGE_KEY = 'makhos.preferences.v1';

const DEFAULT_CONFIG: GameConfig = {
  mode: 'vs-ai',
  difficulty: 'normal',
  humanSide: 1,
  unlimitedThink: false,
};

let nativeStorageCache: AsyncKeyValueStorage | null | undefined;

function sanitizeLanguage(value: unknown): PersistedLanguage {
  return value === 'en' ? 'en' : 'th';
}

function sanitizeConfig(value: unknown): GameConfig {
  if (!value || typeof value !== 'object') return DEFAULT_CONFIG;
  const raw = value as Partial<GameConfig>;
  const mode = raw.mode === 'vs-human' ? 'vs-human' : 'vs-ai';
  const difficulty =
    raw.difficulty === 'easy' ||
    raw.difficulty === 'normal' ||
    raw.difficulty === 'hard' ||
    raw.difficulty === 'expert'
      ? raw.difficulty
      : raw.difficulty === 'master'
        ? 'expert'
        : 'normal';
  const humanSide = raw.humanSide === -1 ? -1 : 1;
  return {
    mode,
    difficulty,
    humanSide,
    unlimitedThink: false,
  };
}

function coercePreferences(input: unknown): PersistedPreferences | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Partial<PersistedPreferences>;
  return {
    version: 1,
    language: sanitizeLanguage(raw.language),
    lastConfig: sanitizeConfig(raw.lastConfig),
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

export function getDefaultConfig(): GameConfig {
  return { ...DEFAULT_CONFIG };
}

export async function loadAppPreferences(): Promise<AppPreferences | null> {
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = await storage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    const coerced = coercePreferences(parsed);
    if (!coerced) return null;
    return { language: coerced.language, lastConfig: coerced.lastConfig };
  } catch {
    return null;
  }
}

export async function saveAppPreferences(preferences: AppPreferences): Promise<void> {
  const storage = getStorage();
  if (!storage) return;
  const payload: PersistedPreferences = {
    version: 1,
    language: sanitizeLanguage(preferences.language),
    lastConfig: sanitizeConfig(preferences.lastConfig),
  };
  await storage.setItem(STORAGE_KEY, JSON.stringify(payload));
}
