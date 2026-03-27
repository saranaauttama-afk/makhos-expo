const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Enable Worker thread bundling — Metro creates a separate JS bundle for each
// file loaded via `new Worker(new URL('./file', import.meta.url))`.
// Works on web natively; works on Android/iOS with New Architecture (RN 0.79+).
config.transformer ??= {};
config.transformer.unstable_allowRequireContext = true;

// Allow bundling .onnx model files as static assets
config.resolver.assetExts.push('onnx');

module.exports = config;
