/**
 * React Hook for ML Player
 *
 * Usage in Expo/React Native app
 */

import { useState, useEffect } from 'react';
import { Asset } from 'expo-asset';
import { MLPlayer } from './mlPlayer';

export function useMLPlayer() {
  const [mlPlayer, setMLPlayer] = useState<MLPlayer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModel();
  }, []);

  async function loadModel() {
    try {
      setLoading(true);

      // Load model from assets
      const asset = Asset.fromModule(require('../../../assets/best_model.onnx'));
      await asset.downloadAsync();

      // Create ML player
      const player = new MLPlayer({
        modelPath: asset.localUri!,
        temperature: 0.1,
        topK: 10
      });

      await player.load();

      setMLPlayer(player);
      setError(null);
    } catch (e) {
      console.error('Failed to load ML model:', e);
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return { mlPlayer, loading, error };
}