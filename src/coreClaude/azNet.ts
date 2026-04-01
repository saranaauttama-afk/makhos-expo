// azNet.ts — ONNX model loading and inference for AlphaZero
//
// Network: 128 → FC+BN+ReLU → 4×ResBlock(256) → policy(1024) + value(1)
// Model file: data/iter_0009.onnx (single-file ONNX)

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';

let session: ort.InferenceSession | null = null;
let sessionPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const [asset] = await Asset.loadAsync(require('../../data/iter_0009.onnx'));
      const uri = asset.localUri ?? asset.uri;
      const s = await ort.InferenceSession.create(uri);
      session = s;
      return s;
    })();
  }
  return sessionPromise;
}

/** Preload model in background — call on screen mount to avoid first-move latency. */
export function preloadAZModel(): void {
  getSession().catch(() => {}); // fire-and-forget
}

/** Run one forward pass. Returns raw policy logits (1024) and value ∈ [-1,1]. */
export async function azInfer(
  features: Float32Array,
): Promise<{ policyLogits: Float32Array; value: number }> {
  const s = await getSession();
  const tensor = new ort.Tensor('float32', features, [1, 128]);
  const results = await s.run({ features: tensor });
  return {
    policyLogits: results['policy_logits'].data as Float32Array,
    value: (results['value'].data as Float32Array)[0],
  };
}
