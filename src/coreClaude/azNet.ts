// azNet.ts - lazy ONNX runtime loading for mobile builds.
//
// Expo Go does not provide the native onnxruntime-react-native module.
// We therefore load it only at runtime and let callers fall back gracefully.

import { Asset } from 'expo-asset';
import { NativeModules } from 'react-native';
import { getBundledAZModels } from './azModelCatalog';

type OrtModule = typeof import('onnxruntime-react-native');
type InferenceSession = import('onnxruntime-react-native').InferenceSession;
type OrtTensor = import('onnxruntime-react-native').Tensor;

let session: InferenceSession | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;
let ortModulePromise: Promise<OrtModule> | null = null;
let azRuntimeAvailable = true;
let loadedModelId: string | null = null;

let activeModelId = getBundledAZModels()[0]?.id ?? 'makhos_az';

function resetSession() {
  session = null;
  sessionPromise = null;
  loadedModelId = null;
}

function findModelById(modelId: string) {
  return getBundledAZModels().find(m => m.id === modelId);
}

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
  if (session && loadedModelId === activeModelId) return session;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      try {
        const model = findModelById(activeModelId);
        if (!model) {
          throw new Error(`Unknown AZ model id: ${activeModelId}`);
        }
        const ort = await getOrtModule();
        const [asset] = await Asset.loadAsync(model.asset);
        const uri = asset.localUri ?? asset.uri;
        const created = await ort.InferenceSession.create(uri);
        session = created;
        loadedModelId = activeModelId;
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

export function getActiveAZModelId() {
  return activeModelId;
}

export function getAvailableAZModels() {
  return getBundledAZModels().map(m => ({ id: m.id, label: m.label }));
}

export function setActiveAZModel(modelId: string): boolean {
  const model = findModelById(modelId);
  if (!model) return false;
  if (activeModelId === modelId) return true;
  activeModelId = modelId;
  azRuntimeAvailable = true;
  resetSession();
  return true;
}

export function preloadAZModel(modelId?: string): void {
  if (modelId) setActiveAZModel(modelId);
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
