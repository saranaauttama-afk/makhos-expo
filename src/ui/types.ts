export type GameMode = 'vs-ai' | 'vs-human';
export type Difficulty = 'easy' | 'medium' | 'hard';

export interface GameConfig {
  mode: GameMode;
  difficulty: Difficulty;
  humanSide: 1 | -1;
}
