export type GameMode = 'vs-ai' | 'vs-human';
export type Difficulty = 'easy' | 'normal' | 'hard' | 'expert' | 'master';
export type AdConsentStatus = 'unknown' | 'granted' | 'denied';

export interface GameConfig {
  mode: GameMode;
  difficulty: Difficulty;
  humanSide: 1 | -1;
}

export interface MonetizationState {
  noAds: boolean;
  consent: AdConsentStatus;
  coins: number;
  rewardedHints: number;
  rewardedUndos: number;
  interstitialCounter: number;
  interstitialSeen: number;
  rewardedSeen: number;
}
