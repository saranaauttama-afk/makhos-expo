import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AdConsentStatus, MonetizationState } from './types';
import { loadPersistedMonetization, savePersistedMonetization } from './walletPersistence';

export type SpendKind = 'hint' | 'undo';
export type RewardKind = 'coins' | 'hint' | 'undo';
export type MatchOutcome = 'win' | 'loss' | 'draw';
export type SpendSource = 'credit' | 'coins' | 'none';

export const HINT_COST = 10;
export const UNDO_COST = 10;
export const WIN_REWARD = 8;
export const LOSE_REWARD = 3;
export const REWARDED_COINS = 20;

export const DEFAULT_MONETIZATION_STATE: MonetizationState = {
  coins: 100,
  hintCredits: 0,
  undoCredits: 0,
  noAdsUnlocked: false,
  adConsent: 'unknown',
  interstitialCounter: 0,
  interstitialSeen: 0,
  rewardedSeen: 0,
};

export interface SpendPreview {
  kind: SpendKind;
  cost: number;
  coins: number;
  availableCredits: number;
  canSpendCoins: boolean;
}

interface SpendResolution {
  source: SpendSource;
  nextState: MonetizationState;
}

function getCost(kind: SpendKind) {
  return kind === 'hint' ? HINT_COST : UNDO_COST;
}

function getCredits(state: MonetizationState, kind: SpendKind) {
  return kind === 'hint' ? state.hintCredits : state.undoCredits;
}

function setCredits(state: MonetizationState, kind: SpendKind, value: number): MonetizationState {
  if (kind === 'hint') return { ...state, hintCredits: value };
  return { ...state, undoCredits: value };
}

export function makeSpendPreview(state: MonetizationState, kind: SpendKind): SpendPreview {
  const cost = getCost(kind);
  const availableCredits = getCredits(state, kind);
  return {
    kind,
    cost,
    coins: state.coins,
    availableCredits,
    canSpendCoins: state.coins >= cost,
  };
}

export function resolveSpend(state: MonetizationState, kind: SpendKind): SpendResolution {
  const availableCredits = getCredits(state, kind);
  if (availableCredits > 0) {
    return {
      source: 'credit',
      nextState: setCredits(state, kind, availableCredits - 1),
    };
  }
  const cost = getCost(kind);
  if (state.coins >= cost) {
    return {
      source: 'coins',
      nextState: { ...state, coins: state.coins - cost },
    };
  }
  return {
    source: 'none',
    nextState: state,
  };
}

export function grantRewarded(state: MonetizationState, kind: RewardKind): MonetizationState {
  const rewardedSeen = state.rewardedSeen + 1;
  if (kind === 'coins') return { ...state, coins: state.coins + REWARDED_COINS, rewardedSeen };
  if (kind === 'hint') return { ...state, hintCredits: state.hintCredits + 1, rewardedSeen };
  return { ...state, undoCredits: state.undoCredits + 1, rewardedSeen };
}

export function grantMatchReward(state: MonetizationState, outcome: MatchOutcome): MonetizationState {
  if (outcome === 'win') return { ...state, coins: state.coins + WIN_REWARD };
  if (outcome === 'loss') return { ...state, coins: state.coins + LOSE_REWARD };
  return state;
}

export function useWalletStore(initial?: Partial<MonetizationState>) {
  const [monetization, setMonetization] = useState<MonetizationState>({
    ...DEFAULT_MONETIZATION_STATE,
    ...initial,
  });
  const [walletHydrated, setWalletHydrated] = useState(false);
  const stateRef = useRef<MonetizationState>(monetization);
  const didHydrateRef = useRef(false);

  useEffect(() => {
    stateRef.current = monetization;
  }, [monetization]);

  useEffect(() => {
    if (didHydrateRef.current) return;
    didHydrateRef.current = true;
    let active = true;
    loadPersistedMonetization()
      .then(stored => {
        if (!active) return;
        if (!stored) {
          setWalletHydrated(true);
          return;
        }
        setMonetization(prev => {
          const next = { ...prev, ...stored };
          stateRef.current = next;
          return next;
        });
        setWalletHydrated(true);
      })
      .catch(() => {
        if (active) setWalletHydrated(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!walletHydrated) return;
    const timeoutId = setTimeout(() => {
      void savePersistedMonetization(monetization);
    }, 120);
    return () => clearTimeout(timeoutId);
  }, [walletHydrated, monetization]);

  const setAdConsent = useCallback((adConsent: AdConsentStatus) => {
    setMonetization(prev => ({ ...prev, adConsent }));
  }, []);

  const consumeSpend = useCallback((kind: SpendKind): SpendSource => {
    const result = resolveSpend(stateRef.current, kind);
    if (result.source !== 'none') {
      stateRef.current = result.nextState;
      setMonetization(result.nextState);
    }
    return result.source;
  }, []);

  const claimReward = useCallback((kind: RewardKind) => {
    setMonetization(prev => {
      const next = grantRewarded(prev, kind);
      stateRef.current = next;
      return next;
    });
  }, []);

  const applyMatchOutcome = useCallback((outcome: MatchOutcome) => {
    setMonetization(prev => {
      const next = grantMatchReward(prev, outcome);
      stateRef.current = next;
      return next;
    });
  }, []);

  const setInterstitialCounter = useCallback((nextCounter: number) => {
    setMonetization(prev => {
      const next = { ...prev, interstitialCounter: nextCounter };
      stateRef.current = next;
      return next;
    });
  }, []);

  const markInterstitialShown = useCallback(() => {
    setMonetization(prev => {
      const next = { ...prev, interstitialSeen: prev.interstitialSeen + 1 };
      stateRef.current = next;
      return next;
    });
  }, []);

  const buyNoAds = useCallback(() => {
    setMonetization(prev => {
      const next = { ...prev, noAdsUnlocked: true };
      stateRef.current = next;
      return next;
    });
  }, []);

  const buyStarterPack = useCallback(() => {
    setMonetization(prev => {
      const next = {
        ...prev,
        noAdsUnlocked: true,
        coins: prev.coins + 500,
        hintCredits: prev.hintCredits + 2,
        undoCredits: prev.undoCredits + 2,
      };
      stateRef.current = next;
      return next;
    });
  }, []);

  const restorePurchase = useCallback(() => {
    setMonetization(prev => {
      const next = { ...prev, noAdsUnlocked: true };
      stateRef.current = next;
      return next;
    });
  }, []);

  return useMemo(
    () => ({
      monetization,
      walletHydrated,
      setMonetization,
      setAdConsent,
      consumeSpend,
      claimReward,
      applyMatchOutcome,
      setInterstitialCounter,
      markInterstitialShown,
      buyNoAds,
      buyStarterPack,
      restorePurchase,
    }),
    [
      monetization,
      walletHydrated,
      setAdConsent,
      consumeSpend,
      claimReward,
      applyMatchOutcome,
      setInterstitialCounter,
      markInterstitialShown,
      buyNoAds,
      buyStarterPack,
      restorePurchase,
    ],
  );
}
