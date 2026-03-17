const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Enable Worker thread bundling — Metro creates a separate JS bundle for each
// file loaded via `new Worker(new URL('./file', import.meta.url))`.
// Works on web natively; works on Android/iOS with New Architecture (RN 0.79+).
config.transformer ??= {};
config.transformer.unstable_allowRequireContext = true;

module.exports = config;
