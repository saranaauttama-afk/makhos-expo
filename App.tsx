import React, { useCallback, useEffect, useState } from 'react';
import { Alert, ScrollView, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { initializeAdService, prepareInterstitialAfterMatch, showRewardedAd } from './src/ads/adService';
import { getActiveAZModelId, getAvailableAZModels, setActiveAZModel } from './src/coreClaude/azNet';
import { precomputeEndgameTablebase } from './src/coreClaude/search/endgameTablebase';
import AccountScreen from './src/ui/AccountScreen';
import ArenaScreen from './src/ui/ArenaScreen';
import HomeScreen from './src/ui/HomeScreen';
import HumanVsCodexArenaScreen from './src/ui/HumanVsCodexArenaScreen';
import { APP_TEXT } from './src/ui/i18n/appText';
import SetupScreen from './src/ui/SetupScreen';
import { GameConfig } from './src/ui/types';
import { MatchOutcome, RewardKind, SpendKind, useWalletStore } from './src/ui/walletStore';

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
  const {
    monetization,
    walletHydrated,
    setAdConsent,
    consumeSpend,
    claimReward,
    applyMatchOutcome,
    setInterstitialCounter,
    markInterstitialShown,
    buyNoAds,
    buyStarterPack,
    restorePurchase,
  } = useWalletStore();
  const [screen, setScreen] = useState<'home' | 'arena' | 'setup' | 'account'>('home');
  const aiModels = getAvailableAZModels();
  const t = APP_TEXT[language];

  useEffect(() => {
    precomputeEndgameTablebase();
    void initializeAdService();
  }, []);

  function handleBuyNoAds() {
    buyNoAds();
    Alert.alert(t.purchaseSimulated, t.noAdsActive);
  }

  function handleBuyStarterPack() {
    buyStarterPack();
    Alert.alert(t.purchaseSimulated, t.starterGranted);
  }

  function handleRestorePurchase() {
    restorePurchase();
    Alert.alert(t.restoreDone, t.restoreBody);
  }

  const runRewarded = useCallback(
    async (kind: RewardKind): Promise<boolean> => {
      const adResult = await showRewardedAd({
        placement: kind,
        adConsent: monetization.adConsent,
      });
      if (!adResult.granted) {
        if (adResult.reason === 'consent_denied') {
          Alert.alert(t.rewardUnavailableTitle, t.rewardUnavailableBody);
        } else {
          Alert.alert(t.adUnavailableTitle, t.adUnavailableBody);
        }
        return false;
      }
      claimReward(kind);
      return true;
    },
    [claimReward, monetization.adConsent, t.adUnavailableBody, t.adUnavailableTitle, t.rewardUnavailableBody, t.rewardUnavailableTitle],
  );

  const handleMatchComplete = useCallback(
    async (outcome: MatchOutcome) => {
      applyMatchOutcome(outcome);

      const interstitial = await prepareInterstitialAfterMatch({
        noAdsUnlocked: monetization.noAdsUnlocked,
        adConsent: monetization.adConsent,
        interstitialEveryMatches: 3,
        completedMatches: monetization.interstitialCounter,
      });

      setInterstitialCounter(interstitial.nextCompletedMatches);
      if (interstitial.shown) markInterstitialShown();
    },
    [applyMatchOutcome, markInterstitialShown, monetization.adConsent, monetization.interstitialCounter, monetization.noAdsUnlocked, setInterstitialCounter],
  );

  const handleConsumeSpend = useCallback((kind: SpendKind) => consumeSpend(kind), [consumeSpend]);

  const screenContent =
    screen === 'arena' ? (
      <ArenaScreen language={language} onBack={() => setScreen('home')} />
    ) : screen === 'setup' && draftConfig ? (
      <SetupScreen
        language={language}
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
        onAdConsentChange={setAdConsent}
        aiModels={aiModels}
        aiModelId={aiModelId}
        onAiModelChange={modelId => {
          if (setActiveAZModel(modelId)) setAiModelId(modelId);
        }}
        onClaimFreeReward={runRewarded}
        onBuyNoAds={handleBuyNoAds}
        onBuyStarterPack={handleBuyStarterPack}
        onRestorePurchase={handleRestorePurchase}
        onBack={() => setScreen('home')}
      />
    ) : gameConfig ? (
      <HumanVsCodexArenaScreen
        language={language}
        config={gameConfig}
        monetization={monetization}
        onConsumeSpend={handleConsumeSpend}
        onWatchRewarded={runRewarded}
        onMatchComplete={handleMatchComplete}
        onBack={() => {
          setGameConfig(null);
          setScreen('home');
        }}
      />
    ) : (
      <HomeScreen
        language={language}
        monetization={monetization}
        onQuickPlay={config => setGameConfig(config)}
        onStart={config => {
          setDraftConfig(config);
          setScreen('setup');
        }}
        onArena={() => setScreen('arena')}
        onAccount={() => setScreen('account')}
      />
    );

  return (
    <SafeAreaProvider style={{ flex: 1 }}>
      <ErrorBoundary>
        {walletHydrated ? (
          screenContent
        ) : (
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 }}>
            <Text style={{ fontSize: 14, opacity: 0.8 }}>{t.loadingWallet}</Text>
          </View>
        )}
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
