import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import Constants from 'expo-constants';
import { useFonts } from 'expo-font';
import { Kanit_400Regular, Kanit_500Medium, Kanit_700Bold, Kanit_800ExtraBold } from '@expo-google-fonts/kanit';
import { initializeAdService, prepareInterstitialAfterMatch, showRewardedAd } from './src/ads/adService';
import { precomputeEndgameTablebase } from './src/coreClaude/search/endgameTablebase';
import AccountScreen from './src/ui/AccountScreen';
import HomeScreen from './src/ui/HomeScreen';
import HumanVsCodexArenaScreen from './src/ui/HumanVsCodexArenaScreen';
import { getDefaultConfig, loadAppPreferences, saveAppPreferences } from './src/ui/appPreferencesPersistence';
import { APP_TEXT } from './src/ui/i18n/appText';
import SetupScreen from './src/ui/SetupScreen';
import { GameConfig } from './src/ui/types';
import { MatchOutcome, RewardKind, SpendKind, useWalletStore } from './src/ui/walletStore';

const GlobalText = Text as typeof Text & { defaultProps?: Record<string, unknown> };
GlobalText.defaultProps = GlobalText.defaultProps ?? {};
GlobalText.defaultProps.style = [
  GlobalText.defaultProps.style,
  { fontFamily: 'Kanit_500Medium' },
];

type ErrorBoundaryState = { error: Error | null };
export type AppLanguage = 'th' | 'en';
const INTERSTITIAL_EVERY_MATCHES = 5;
const COMPANY_NAME = 'Void Light Work & Diehard Monkey';
const APP_VERSION = Constants.expoConfig?.version ?? '1.0.0';
const FALLBACK_CONFIG: GameConfig = getDefaultConfig();

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
        <Text style={{ fontSize: 22, fontFamily: 'Kanit_700Bold' }}>Web render error</Text>
        <Text selectable style={{ fontSize: 14, lineHeight: 20, fontFamily: 'Kanit_500Medium' }}>{this.state.error.message}</Text>
        <View style={{ height: 1, backgroundColor: '#d4d4d8' }} />
        <Text style={{ fontSize: 12, opacity: 0.75, fontFamily: 'Kanit_500Medium' }}>
          Check the browser console or tell me the message above and I will fix the root cause.
        </Text>
      </ScrollView>
    );
  }
}

export default function App() {
  const [fontsLoaded] = useFonts({
    Kanit_400Regular,
    Kanit_500Medium,
    Kanit_700Bold,
    Kanit_800ExtraBold,
  });
  const [bootReady, setBootReady] = useState(false);
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [draftConfig, setDraftConfig] = useState<GameConfig>(FALLBACK_CONFIG);
  const [language, setLanguage] = useState<AppLanguage>('th');
  const [prefsHydrated, setPrefsHydrated] = useState(false);
  const {
    monetization,
    walletHydrated,
    setSoundEnabled,
    setVibrationEnabled,
    consumeSpend,
    claimReward,
    applyMatchOutcome,
    setInterstitialCounter,
    markInterstitialShown,
    buyNoAds,
    buyStarterPack,
  } = useWalletStore();
  const [screen, setScreen] = useState<'home' | 'setup' | 'account'>('home');
  const t = APP_TEXT[language];

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        precomputeEndgameTablebase();
        await initializeAdService();
      } catch (error) {
        console.warn('Boot initialization failed', error);
      } finally {
        if (mounted) setBootReady(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    loadAppPreferences()
      .then(prefs => {
        if (!active) return;
        if (prefs) {
          setLanguage(prefs.language);
          setDraftConfig(prefs.lastConfig);
        }
        setPrefsHydrated(true);
      })
      .catch(() => {
        if (active) setPrefsHydrated(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!prefsHydrated) return;
    const timeoutId = setTimeout(() => {
      void saveAppPreferences({ language, lastConfig: draftConfig });
    }, 120);
    return () => clearTimeout(timeoutId);
  }, [prefsHydrated, language, draftConfig]);

  function handleBuyNoAds() {
    buyNoAds();
    Alert.alert(t.purchaseSimulated, t.noAdsActive);
  }

  function handleBuyStarterPack() {
    buyStarterPack();
    Alert.alert(t.purchaseSimulated, t.starterGranted);
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
        interstitialEveryMatches: INTERSTITIAL_EVERY_MATCHES,
        completedMatches: monetization.interstitialCounter,
      });

      setInterstitialCounter(interstitial.nextCompletedMatches);
      if (interstitial.shown) markInterstitialShown();
    },
    [applyMatchOutcome, markInterstitialShown, monetization.adConsent, monetization.interstitialCounter, monetization.noAdsUnlocked, setInterstitialCounter],
  );

  const handleConsumeSpend = useCallback((kind: SpendKind) => consumeSpend(kind), [consumeSpend]);

  const screenContent =
    screen === 'setup' ? (
      <SetupScreen
        language={language}
        initialConfig={draftConfig}
        monetization={monetization}
        onBack={() => setScreen('home')}
        onPlay={config => {
          setDraftConfig(config);
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
        appVersion={APP_VERSION}
        companyName={COMPANY_NAME}
        onSoundEnabledChange={setSoundEnabled}
        onVibrationEnabledChange={setVibrationEnabled}
        onClaimFreeReward={runRewarded}
        onBuyNoAds={handleBuyNoAds}
        onBuyStarterPack={handleBuyStarterPack}
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
        companyName={COMPANY_NAME}
        appVersion={APP_VERSION}
        onQuickPlay={config => setGameConfig(config)}
        onStart={config => {
          setDraftConfig(config);
          setScreen('setup');
        }}
        defaultSetupConfig={draftConfig}
        onAccount={() => setScreen('account')}
      />
    );

  return (
    <SafeAreaProvider style={{ flex: 1 }}>
      <ErrorBoundary>
        {bootReady && walletHydrated && prefsHydrated && fontsLoaded ? (
          screenContent
        ) : (
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, backgroundColor: '#0f172a' }}>
            <Text style={{ fontSize: 30, fontFamily: 'Kanit_800ExtraBold', color: '#f8fafc', letterSpacing: 0.6 }}>MAKHOS</Text>
            <Text style={{ marginTop: 8, fontSize: 13, color: '#cbd5e1', fontFamily: 'Kanit_500Medium' }}>Thai Checkers AI</Text>
            <ActivityIndicator size="large" color="#f8fafc" style={{ marginTop: 22 }} />
            <Text style={{ marginTop: 14, fontSize: 12, color: '#94a3b8', fontFamily: 'Kanit_500Medium' }}>
              {bootReady ? t.loadingWallet : 'Loading game systems...'}
            </Text>
          </View>
        )}
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}

