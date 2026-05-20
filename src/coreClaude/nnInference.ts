/**
 * nnInference.ts - ONNX Neural Network inference for React Native
 *
 * Uses onnxruntime-react-native for mobile inference
 * Safe version with proper error handling
 */

import { Position } from './position';
import { extractNNFeatures } from './nnFeatures';
import { Move } from './movegen';

let session: any = null;
let ort: any = null;
let isInitializing = false;
let initFailed = false;

/**
 * Initialize ONNX inference session for React Native
 */
export async function initNNInference(_modelPath?: string): Promise<void> {
  if (session) return; // Already initialized
  if (isInitializing) {
    // Wait for ongoing initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    return initNNInference(_modelPath);
  }
  if (initFailed) {
    throw new Error('NN initialization previously failed');
  }

  isInitializing = true;

  try {
    console.log('[NN] ========================================');
    console.log('[NN] Starting NN initialization...');
    console.log('[NN] ========================================');

    // Step 1: Load ONNX Runtime
    console.log('[NN] Step 1: Loading ONNX Runtime for React Native...');
    try {
      ort = await import('onnxruntime-react-native');
      console.log('[NN] ✅ ONNX Runtime loaded successfully');
    } catch (err) {
      console.error('[NN] ❌ Failed to load ONNX Runtime:', err);
      throw new Error(`ONNX Runtime load failed: ${err}`);
    }

    // Step 2: Load Expo modules
    console.log('[NN] Step 2: Loading Expo modules...');
    let Asset, FileSystem;
    try {
      const assetModule = await import('expo-asset');
      Asset = assetModule.Asset;
      FileSystem = await import('expo-file-system');
      console.log('[NN] ✅ Expo modules loaded');
    } catch (err) {
      console.error('[NN] ❌ Failed to load Expo modules:', err);
      throw new Error(`Expo modules load failed: ${err}`);
    }

    // Step 3: Load model asset
    console.log('[NN] Step 3: Loading model asset...');
    let modelAsset;
    try {
      modelAsset = Asset.fromModule(require('../../models/thai_checkers_v3_mobile.onnx'));
      console.log('[NN] Model asset object created');
      await modelAsset.downloadAsync();
      console.log('[NN] ✅ Model asset downloaded');
    } catch (err) {
      console.error('[NN] ❌ Failed to load model asset:', err);
      throw new Error(`Model asset load failed: ${err}`);
    }

    // Step 4: Get model URI
    const modelUri = modelAsset.localUri || modelAsset.uri;
    if (!modelUri) {
      throw new Error('Failed to get model URI from asset');
    }
    console.log('[NN] Model URI:', modelUri);

    // Step 5: Create ONNX session
    console.log('[NN] Step 5: Creating ONNX inference session...');
    console.log('[NN] Model URI:', modelUri);
    console.log('[NN] Attempting to create session with onnxruntime-react-native...');
    try {
      session = await ort.InferenceSession.create(modelUri);
      console.log('[NN] ✅✅✅ NN model loaded successfully on mobile!');
      console.log('[NN] Model inputs:', session.inputNames);
      console.log('[NN] Model outputs:', session.outputNames);
      console.log('[NN] ========================================');
    } catch (err) {
      console.error('[NN] ❌ Failed to create ONNX session:', err);
      console.error('[NN] Error type:', err instanceof Error ? err.constructor.name : typeof err);
      console.error('[NN] Error message:', err instanceof Error ? err.message : String(err));
      console.error('[NN] Error stack:', err instanceof Error ? err.stack : 'No stack');

      // Try to show alert before throwing
      const errMsg = err instanceof Error ? err.message : String(err);
      setTimeout(() => {
        alert(`ONNX Session Creation Failed!\n\n${errMsg}\n\nModel: thai_checkers_v3_mobile.onnx\nOpset: 17`);
      }, 100);

      throw new Error(`ONNX session creation failed: ${errMsg}`);
    }

    isInitializing = false;
  } catch (error) {
    isInitializing = false;
    initFailed = true;
    console.error('[NN] ========================================');
    console.error('[NN] ❌❌❌ NN INITIALIZATION FAILED');
    console.error('[NN] Error:', error);
    console.error('[NN] Error message:', error instanceof Error ? error.message : String(error));
    console.error('[NN] Stack:', error instanceof Error ? error.stack : 'No stack');
    console.error('[NN] ========================================');

    // Show alert to user
    const errMsg = error instanceof Error ? error.message : String(error);
    setTimeout(() => {
      alert(`NN Initialization Failed!\n\n${errMsg}\n\nThe app will use Minimax AI instead.`);
    }, 100);

    throw error;
  }
}

/**
 * Evaluate position using NN
 * Returns { value, policyLogits }
 */
export async function evaluateNN(pos: Position): Promise<{ value: number; policyLogits: Float32Array }> {
  if (!session) {
    throw new Error('NN inference not initialized. Call initNNInference() first.');
  }

  // Extract features (320-dim)
  const features = extractNNFeatures(pos);

  // Create input tensor
  const inputTensor = new ort.Tensor('float32', features, [1, 320]);

  // Run inference
  let outputs;
  try {
    console.log('[NN] Running inference...');
    console.log('[NN] Input tensor shape:', inputTensor.dims);
    console.log('[NN] Input data length:', features.length);
    outputs = await session.run({ features: inputTensor });
    console.log('[NN] ✅ Inference complete');
  } catch (err) {
    console.error('[NN] ❌ Inference failed:', err);
    console.error('[NN] Error type:', err instanceof Error ? err.constructor.name : typeof err);
    console.error('[NN] Error message:', err instanceof Error ? err.message : String(err));
    console.error('[NN] Error stack:', err instanceof Error ? err.stack : 'No stack');
    console.error('[NN] Input shape:', inputTensor.dims);
    console.error('[NN] Input data length:', features.length);
    const errMsg = err instanceof Error ? err.message : String(err);
    throw new Error(`NN inference failed: ${errMsg}`);
  }

  // Extract outputs
  const policyLogits = outputs['policy_logits'].data as Float32Array;
  const valueOutput = outputs['value'].data as Float32Array;
  const value = valueOutput[0];

  return { value, policyLogits };
}

/**
 * Select best move from NN policy logits
 */
export function selectBestNNMove(
  policyLogits: Float32Array,
  legalMoves: Move[],
  _pos: Position,
): Move | undefined {
  if (legalMoves.length === 0) return undefined;

  // Find move with highest policy logit
  let bestMove = legalMoves[0];
  let bestLogit = -Infinity;

  for (const move of legalMoves) {
    const moveIdx = move.from * 32 + move.to;
    const logit = policyLogits[moveIdx];
    if (logit > bestLogit) {
      bestLogit = logit;
      bestMove = move;
    }
  }

  return bestMove;
}

/**
 * Close inference session
 */
export function closeNNInference(): void {
  if (session) {
    // ONNX Runtime React Native doesn't have explicit close
    session = null;
  }
}
