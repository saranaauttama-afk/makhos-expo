export type GameMode = 'vs-ai' | 'vs-human';
export type Difficulty = 'easy' | 'normal' | 'hard' | 'expert' | 'master' | 'alpha';
export type AdConsentStatus = 'unknown' | 'granted' | 'denied';

export interface GameConfig {
  mode: GameMode;
  difficulty: Difficulty;
  humanSide: 1 | -1;
  unlimitedThink: boolean;
}

export interface WalletModel {
  coins: number;
  hintCredits: number;
  undoCredits: number;
  noAdsUnlocked: boolean;
  adConsent: AdConsentStatus;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
}

export interface MonetizationState {
  coins: number;
  hintCredits: number;
  undoCredits: number;
  noAdsUnlocked: boolean;
  adConsent: AdConsentStatus;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
  interstitialCounter: number;
  interstitialSeen: number;
  rewardedSeen: number;
}
