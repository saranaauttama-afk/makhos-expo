/**
 * nnInferenceWeb.ts - ONNX Web inference using WebView
 *
 * This uses ONNX Runtime Web in a WebView to run inference
 * Works around onnxruntime-react-native compatibility issues with Expo
 */

import { Position } from './position';
import { Move } from './movegen';

type InferenceCallback = {
  resolve: (result: { value: number; policyLogits: Float32Array }) => void;
  reject: (error: Error) => void;
};

let webViewRef: any = null;
let isInitialized = false;
let pendingInference: InferenceCallback | null = null;

/**
 * Set the WebView reference (called from React component)
 */
export function setWebViewRef(ref: any): void {
  webViewRef = ref;
  console.log('[NN Web] WebView ref set');
}

/**
 * Handle messages from WebView
 */
export function handleWebViewMessage(event: any): void {
  try {
    const data = JSON.parse(event.nativeEvent.data);
    console.log('[NN Web] Received message:', data.type);

    switch (data.type) {
      case 'onnx_ready':
        isInitialized = true;
        console.log('[NN Web] ✅ ONNX Runtime initialized');
        break;

      case 'model_loaded':
        console.log('[NN Web] ✅ Model loaded');
        break;

      case 'inference_result':
        if (pendingInference) {
          const { policy_logits, value } = data;
          const policyLogits = new Float32Array(policy_logits);
          pendingInference.resolve({ value, policyLogits });
          pendingInference = null;
        }
        break;

      case 'onnx_error':
      case 'inference_error':
        console.error('[NN Web] Error:', data.error);
        if (pendingInference) {
          pendingInference.reject(new Error(data.error));
          pendingInference = null;
        }
        break;

      default:
        console.log('[NN Web] Unknown message type:', data.type);
    }
  } catch (error) {
    console.error('[NN Web] Failed to parse message:', error);
  }
}

/**
 * Initialize NN inference (load model in WebView)
 */
export async function initNNInferenceWeb(): Promise<void> {
  if (!webViewRef) {
    throw new Error('WebView ref not set. Call setWebViewRef() first.');
  }

  if (isInitialized) {
    console.log('[NN Web] Already initialized');
    return;
  }

  console.log('[NN Web] Initializing...');

  // Send message to WebView to load model
  webViewRef.injectJavaScript(`
    window.loadONNXModel();
    true;
  `);

  // Wait for initialization
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('ONNX initialization timeout'));
    }, 30000); // 30 second timeout

    const checkInit = setInterval(() => {
      if (isInitialized) {
        clearInterval(checkInit);
        clearTimeout(timeout);
        resolve();
      }
    }, 100);
  });
}

/**
 * Evaluate position using NN in WebView
 */
export async function evaluateNNWeb(pos: Position): Promise<{ value: number; policyLogits: Float32Array }> {
  if (!webViewRef) {
    throw new Error('WebView not initialized');
  }

  if (pendingInference) {
    throw new Error('Inference already in progress');
  }

  // Extract features (same as native inference)
  const { extractNNFeatures } = await import('./nnFeatures');
  const features = extractNNFeatures(pos);

  console.log('[NN Web] Running inference...');

  // Send features to WebView
  const featuresArray = Array.from(features);
  webViewRef.injectJavaScript(`
    window.runInference(${JSON.stringify(featuresArray)});
    true;
  `);

  // Wait for result
  return new Promise((resolve, reject) => {
    pendingInference = { resolve, reject };

    // Timeout after 10 seconds
    setTimeout(() => {
      if (pendingInference) {
        pendingInference = null;
        reject(new Error('Inference timeout'));
      }
    }, 10000);
  });
}

/**
 * Select best move from policy logits (same as native)
 */
export function selectBestNNMoveWeb(
  policyLogits: Float32Array,
  legalMoves: Move[],
  _pos: Position,
): Move | undefined {
  if (legalMoves.length === 0) return undefined;

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
