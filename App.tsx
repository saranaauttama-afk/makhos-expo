import React, { useEffect, useState } from 'react';
import { ScrollView, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import HomeScreen from './src/ui/HomeScreen';
import HumanVsCodexArenaScreen from './src/ui/HumanVsCodexArenaScreen';
import ArenaScreen from './src/ui/ArenaScreen';
import ConceptScreen from './src/ui/ConceptScreen';
import SetupScreen from './src/ui/SetupScreen';
import ShopScreen from './src/ui/ShopScreen';
import SettingsScreen from './src/ui/SettingsScreen';
import ResultScreen from './src/ui/ResultScreen';
import { precomputeEndgameTablebase } from './src/coreClaude/search/endgameTablebase';
import { GameConfig } from './src/ui/types';

type ErrorBoundaryState = { error: Error | null };

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
  const [screen, setScreen] = useState<'home' | 'arena' | 'concept' | 'setup' | 'shop' | 'settings' | 'result'>('home');

  useEffect(() => { precomputeEndgameTablebase(); }, []);

  return (
    <SafeAreaProvider style={{ flex: 1 }}>
      <ErrorBoundary>
        {screen === 'arena' ? (
          <ArenaScreen onBack={() => setScreen('home')} />
        ) : screen === 'concept' ? (
          <ConceptScreen onBack={() => setScreen('home')} />
        ) : screen === 'setup' && draftConfig ? (
          <SetupScreen
            initialConfig={draftConfig}
            onBack={() => setScreen('home')}
            onPlay={config => {
              setGameConfig(config);
              setScreen('home');
            }}
            onPreviewResult={() => setScreen('result')}
            onOpenShop={() => setScreen('shop')}
          />
        ) : screen === 'shop' ? (
          <ShopScreen onBack={() => setScreen('home')} />
        ) : screen === 'settings' ? (
          <SettingsScreen onBack={() => setScreen('home')} />
        ) : screen === 'result' ? (
          <ResultScreen
            onBack={() => setScreen('home')}
            onOpenShop={() => setScreen('shop')}
            onReplay={() => setScreen(draftConfig ? 'setup' : 'home')}
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
            onStart={config => {
              setDraftConfig(config);
              setScreen('setup');
            }}
            onArena={() => setScreen('arena')}
            onConcept={() => setScreen('concept')}
            onShop={() => setScreen('shop')}
            onSettings={() => setScreen('settings')}
            onResultPreview={() => setScreen('result')}
          />
        )}
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
