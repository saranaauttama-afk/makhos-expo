import React, { useEffect } from 'react';
import { ScrollView, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import HumanVsCodexArenaScreen from './src/ui/HumanVsCodexArenaScreen';
import { precomputeEndgameTablebase } from './src/coreCodex/search/endgameTablebase';

type ErrorBoundaryState = {
  error: Error | null;
};

class ErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error('Root render failed', error);
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 24, justifyContent: 'center', gap: 12 }}>
        <Text style={{ fontSize: 22, fontWeight: '700' }}>Web render error</Text>
        <Text selectable style={{ fontSize: 14, lineHeight: 20 }}>
          {this.state.error.message}
        </Text>
        <View style={{ height: 1, backgroundColor: '#d4d4d8' }} />
        <Text style={{ fontSize: 12, opacity: 0.75 }}>
          Check the browser console or tell me the message above and I will fix the root cause.
        </Text>
      </ScrollView>
    );
  }
}

export default function App() {
  // Warm up the endgame tablebase in the background so probes during a game
  // are instant Map lookups instead of on-the-fly retrograde analysis.
  useEffect(() => { precomputeEndgameTablebase(); }, []);

  return (
    <SafeAreaProvider style={{ flex: 1 }}>
      <ErrorBoundary>
        <HumanVsCodexArenaScreen />
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
