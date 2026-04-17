import React, { useEffect, useState } from 'react';
import { Alert, ScrollView, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import HomeScreen from './src/ui/HomeScreen';
import HumanVsCodexArenaScreen from './src/ui/HumanVsCodexArenaScreen';
import ArenaScreen from './src/ui/ArenaScreen';
import SetupScreen from './src/ui/SetupScreen';
import AccountScreen from './src/ui/AccountScreen';
import { precomputeEndgameTablebase } from './src/coreClaude/search/endgameTablebase';
import { getActiveAZModelId, getAvailableAZModels, setActiveAZModel } from './src/coreClaude/azNet';
import { GameConfig, MonetizationState } from './src/ui/types';

type ErrorBoundaryState = { error: Error | null };
export type AppLanguage = 'th' | 'en';

class ErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error('Root render failed', error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, justifyContent: 'center', gap: 12 }}>
        <Text style={{ fontSize: 22, fontWeight: '700' }}>Web render error</Text>
        <Text selectable style={{ fontSize: 14, lineHeight: 20 }}>{this.state.error.message}</Text>
        <View style={{ height: 1, backgroundColor: '#d4d4d8' }} />
        <Text style={{ fontSize: 12, opacity: 0.75 }}>
          Check the browser console or tell me the message above and I will fix the root cause.
        </Text>
      </ScrollView>
    );
  }
}

export default function App() {
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [draftConfig, setDraftConfig] = useState<GameConfig | null>(null);
  const [language, setLanguage] = useState<AppLanguage>('th');
  const [aiModelId, setAiModelId] = useState<string>(() => getActiveAZModelId());
  const [monetization, setMonetization] = useState<MonetizationState>({
    noAds: false,
    consent: 'unknown',
    coins: 120,
    rewardedHints: 0,
    rewardedUndos: 0,
    interstitialCounter: 0,
    interstitialSeen: 0,
    rewardedSeen: 0,
  });
  const [screen, setScreen] = useState<'home' | 'arena' | 'setup' | 'account'>('home');
  const aiModels = getAvailableAZModels();

  useEffect(() => { precomputeEndgameTablebase(); }, []);

  function handleBuyNoAds() {
    setMonetization(prev => ({ ...prev, noAds: true }));
    Alert.alert('Purchase simulated', 'No Ads is now active.');
  }

  function handleBuyStarterPack() {
    setMonetization(prev => ({
      ...prev,
      noAds: true,
      coins: prev.coins + 500,
      rewardedHints: prev.rewardedHints + 2,
      rewardedUndos: prev.rewardedUndos + 2,
    }));
    Alert.alert('Purchase simulated', 'Starter Pack granted: No Ads + credits.');
  }

  function handleRestorePurchase() {
    setMonetization(prev => ({ ...prev, noAds: true }));
    Alert.alert('Restore complete', 'Restored No Ads entitlement (simulated).');
  }

  return (
    <SafeAreaProvider style={{ flex: 1 }}>
      <ErrorBoundary>
        {screen === 'arena' ? (
          <ArenaScreen onBack={() => setScreen('home')} />
        ) : screen === 'setup' && draftConfig ? (
          <SetupScreen
            initialConfig={draftConfig}
            monetization={monetization}
            onBack={() => setScreen('home')}
            onPlay={config => {
              setGameConfig(config);
              setScreen('home');
            }}
            onOpenAccount={() => setScreen('account')}
          />
        ) : screen === 'account' ? (
          <AccountScreen
            language={language}
            onLanguageChange={setLanguage}
            monetization={monetization}
            onAdConsentChange={consent => setMonetization(prev => ({ ...prev, consent }))}
            aiModels={aiModels}
            aiModelId={aiModelId}
            onAiModelChange={modelId => {
              if (setActiveAZModel(modelId)) setAiModelId(modelId);
            }}
            onBuyNoAds={handleBuyNoAds}
            onBuyStarterPack={handleBuyStarterPack}
            onRestorePurchase={handleRestorePurchase}
            onBack={() => setScreen('home')}
          />
        ) : gameConfig ? (
          <HumanVsCodexArenaScreen
            config={gameConfig}
            onBack={() => {
              setGameConfig(null);
              setScreen('home');
            }}
          />
        ) : (
          <HomeScreen
            language={language}
            onQuickPlay={config => setGameConfig(config)}
            onStart={config => {
              setDraftConfig(config);
              setScreen('setup');
            }}
            onArena={() => setScreen('arena')}
            onAccount={() => setScreen('account')}
          />
        )}
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
