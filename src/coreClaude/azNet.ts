// azNet.ts - lazy ONNX runtime loading for mobile builds.
//
// Expo Go does not provide the native onnxruntime-react-native module.
// We therefore load it only at runtime and let callers fall back gracefully.

import { Asset } from 'expo-asset';
import { NativeModules } from 'react-native';

type OrtModule = typeof import('onnxruntime-react-native');
type InferenceSession = import('onnxruntime-react-native').InferenceSession;
type OrtTensor = import('onnxruntime-react-native').Tensor;

let session: InferenceSession | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;
let ortModulePromise: Promise<OrtModule> | null = null;
let azRuntimeAvailable = true;

async function getOrtModule(): Promise<OrtModule> {
  if (!azRuntimeAvailable) {
    throw new Error('AZ runtime unavailable');
  }
  if (!NativeModules.Onnxruntime) {
    azRuntimeAvailable = false;
    throw new Error('ONNX native module is not available in this runtime');
  }
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-react-native').catch(err => {
      azRuntimeAvailable = false;
      throw err;
    });
  }
  return ortModulePromise;
}

async function getSession(): Promise<InferenceSession> {
  if (session) return session;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      try {
        const ort = await getOrtModule();
        const [asset] = await Asset.loadAsync(require('../../assets/models/makhos_az.onnx'));
        const uri = asset.localUri ?? asset.uri;
        const created = await ort.InferenceSession.create(uri);
        session = created;
        return created;
      } catch (error) {
        azRuntimeAvailable = false;
        sessionPromise = null;
        throw error;
      }
    })();
  }
  return sessionPromise;
}

export function isAZRuntimeAvailable() {
  return azRuntimeAvailable;
}

export function preloadAZModel(): void {
  getSession().catch(() => {});
}

export async function azInfer(
  features: Float32Array,
): Promise<{ policyLogits: Float32Array; value: number }> {
  const loadedSession = await getSession();
  const ort = await getOrtModule();
  const tensor = new ort.Tensor('float32', features, [1, 128]) as OrtTensor;
  const results = await loadedSession.run({ features: tensor });
  return {
    policyLogits: results['policy_logits'].data as Float32Array,
    value: (results['value'].data as Float32Array)[0],
  };
}
