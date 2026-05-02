import type { AdConsentStatus } from '../ui/types';

export type RewardedPlacement = 'hint' | 'undo' | 'coins';
export type InterstitialPlacement = 'post_match';

export interface RewardedAdProviderResult {
  completed: boolean;
}

export interface InterstitialAdProviderResult {
  shown: boolean;
}

export interface AdServiceProvider {
  name: string;
  initialize?: () => Promise<void> | void;
  showRewardedAd: (placement: RewardedPlacement) => Promise<RewardedAdProviderResult>;
  showInterstitialAd: (placement: InterstitialPlacement) => Promise<InterstitialAdProviderResult>;
}

export interface RewardedAdInput {
  placement: RewardedPlacement;
  adConsent: AdConsentStatus;
}

export interface RewardedAdResult {
  granted: boolean;
  placement: RewardedPlacement;
  reason: 'completed' | 'consent_denied' | 'provider_error';
}

export interface MatchEndInterstitialInput {
  noAdsUnlocked: boolean;
  adConsent: AdConsentStatus;
  interstitialEveryMatches: number;
  completedMatches: number;
}

export interface MatchEndInterstitialResult {
  shown: boolean;
  reason: 'shown' | 'no_ads' | 'consent_denied' | 'throttled' | 'provider_error';
  nextCompletedMatches: number;
}

function sleep(ms: number) {
  return new Promise<void>(resolve => setTimeout(resolve, ms));
}

const mockAdProvider: AdServiceProvider = {
  name: 'mock',
  async initialize() {
    await sleep(30);
  },
  async showRewardedAd() {
    await sleep(1200);
    return { completed: true };
  },
  async showInterstitialAd() {
    await sleep(900);
    return { shown: true };
  },
};

let adProvider: AdServiceProvider = mockAdProvider;
let adProviderInitialized = false;
let adProviderInitPromise: Promise<void> | null = null;

export function setAdProvider(provider: AdServiceProvider) {
  adProvider = provider;
  adProviderInitialized = false;
  adProviderInitPromise = null;
}

export function getAdProviderName() {
  return adProvider.name;
}

export async function initializeAdService() {
  if (adProviderInitialized) return;
  if (adProviderInitPromise) return adProviderInitPromise;
  adProviderInitPromise = (async () => {
    await adProvider.initialize?.();
    adProviderInitialized = true;
  })().finally(() => {
    adProviderInitPromise = null;
  });
  return adProviderInitPromise;
}

export async function prepareInterstitialAfterMatch(input: MatchEndInterstitialInput): Promise<MatchEndInterstitialResult> {
  if (input.noAdsUnlocked) {
    return { shown: false, reason: 'no_ads', nextCompletedMatches: input.completedMatches };
  }

  const every = Math.max(2, input.interstitialEveryMatches);
  const nextCompletedMatches = input.completedMatches + 1;
  if (nextCompletedMatches % every !== 0) {
    return { shown: false, reason: 'throttled', nextCompletedMatches };
  }

  try {
    // TODO(ad-sdk): setAdProvider(...) from your real SDK bridge.
    // Interstitial is intentionally requested only after match end.
    await initializeAdService();
    const result = await adProvider.showInterstitialAd('post_match');
    return { shown: result.shown, reason: result.shown ? 'shown' : 'throttled', nextCompletedMatches };
  } catch {
    return { shown: false, reason: 'provider_error', nextCompletedMatches };
  }
}

export async function showRewardedAd(input: RewardedAdInput): Promise<RewardedAdResult> {
  try {
    // TODO(ad-sdk): map rewarded unit ids by placement:
    // hint | undo | coins. Must remain user-triggered only.
    await initializeAdService();
    const result = await adProvider.showRewardedAd(input.placement);
    return {
      granted: result.completed,
      placement: input.placement,
      reason: result.completed ? 'completed' : 'provider_error',
    };
  } catch {
    return { granted: false, placement: input.placement, reason: 'provider_error' };
  }
}
