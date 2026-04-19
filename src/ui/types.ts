export type GameMode = 'vs-ai' | 'vs-human';
export type Difficulty = 'easy' | 'normal' | 'hard' | 'expert' | 'master';
export type AdConsentStatus = 'unknown' | 'granted' | 'denied';

export interface GameConfig {
  mode: GameMode;
  difficulty: Difficulty;
  humanSide: 1 | -1;
}

export interface WalletModel {
  coins: number;
  hintCredits: number;
  undoCredits: number;
  noAdsUnlocked: boolean;
  adConsent: AdConsentStatus;
}

export interface MonetizationState {
  coins: number;
  hintCredits: number;
  undoCredits: number;
  noAdsUnlocked: boolean;
  adConsent: AdConsentStatus;
  interstitialCounter: number;
  interstitialSeen: number;
  rewardedSeen: number;
}
