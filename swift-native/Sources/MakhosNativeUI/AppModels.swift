import Foundation

public enum GameMode: String, CaseIterable, Codable, Sendable {
    case vsAI = "vs-ai"
    case vsHuman = "vs-human"
}

public enum Difficulty: String, CaseIterable, Codable, Sendable {
    case easy
    case normal
    case hard
    case expert
    case master
    case alpha
}

public enum AdConsentStatus: String, CaseIterable, Codable, Sendable {
    case unknown
    case granted
    case denied
}

public struct GameConfig: Equatable, Codable, Sendable {
    public var mode: GameMode
    public var difficulty: Difficulty
    public var humanSide: Int
    public var unlimitedThink: Bool

    public init(
        mode: GameMode = .vsAI,
        difficulty: Difficulty = .normal,
        humanSide: Int = 1,
        unlimitedThink: Bool = false
    ) {
        self.mode = mode
        self.difficulty = difficulty
        self.humanSide = humanSide
        self.unlimitedThink = unlimitedThink
    }
}

public struct WalletModel: Equatable, Codable, Sendable {
    public var coins: Int
    public var hintCredits: Int
    public var undoCredits: Int
    public var noAdsUnlocked: Bool
    public var adConsent: AdConsentStatus
    public var soundEnabled: Bool
    public var vibrationEnabled: Bool

    public init(
        coins: Int = 0,
        hintCredits: Int = 0,
        undoCredits: Int = 0,
        noAdsUnlocked: Bool = false,
        adConsent: AdConsentStatus = .unknown,
        soundEnabled: Bool = true,
        vibrationEnabled: Bool = true
    ) {
        self.coins = coins
        self.hintCredits = hintCredits
        self.undoCredits = undoCredits
        self.noAdsUnlocked = noAdsUnlocked
        self.adConsent = adConsent
        self.soundEnabled = soundEnabled
        self.vibrationEnabled = vibrationEnabled
    }
}

public struct AppPreferences: Equatable, Codable, Sendable {
    public var language: String
    public var config: GameConfig
    public var wallet: WalletModel

    public init(language: String = "th", config: GameConfig = .init(), wallet: WalletModel = .init()) {
        self.language = language
        self.config = config
        self.wallet = wallet
    }
}
