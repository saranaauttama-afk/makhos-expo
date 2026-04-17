import type { AdConsentStatus } from '../ui/types';

export interface InterstitialGateInput {
  noAds: boolean;
  consent: AdConsentStatus;
  interstitialEvery: number;
  interstitialCounter: number;
}

export interface InterstitialGateResult {
  shown: boolean;
  reason: 'shown' | 'no_ads' | 'consent_denied' | 'throttled';
  nextCounter: number;
}

export interface RewardedGateInput {
  consent: AdConsentStatus;
}

export interface RewardedGateResult {
  granted: boolean;
  reason: 'completed' | 'consent_denied';
}

function sleep(ms: number) {
  return new Promise<void>(resolve => setTimeout(resolve, ms));
}

export async function maybeShowInterstitialAd(input: InterstitialGateInput): Promise<InterstitialGateResult> {
  if (input.noAds) {
    return { shown: false, reason: 'no_ads', nextCounter: input.interstitialCounter };
  }
  if (input.consent !== 'granted') {
    return { shown: false, reason: 'consent_denied', nextCounter: input.interstitialCounter };
  }

  const every = Math.max(1, input.interstitialEvery);
  const nextCounter = input.interstitialCounter + 1;
  if (nextCounter % every !== 0) {
    return { shown: false, reason: 'throttled', nextCounter };
  }

  // Mock ad latency. Replace with real SDK call later.
  await sleep(900);
  return { shown: true, reason: 'shown', nextCounter };
}

export async function showRewardedAd(input: RewardedGateInput): Promise<RewardedGateResult> {
  if (input.consent !== 'granted') {
    return { granted: false, reason: 'consent_denied' };
  }
  // Mock rewarded ad playback.
  await sleep(1200);
  return { granted: true, reason: 'completed' };
}

