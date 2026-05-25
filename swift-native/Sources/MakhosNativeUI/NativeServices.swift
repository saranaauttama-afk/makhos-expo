import Foundation

public protocol PreferencesStore {
    func loadPreferences() throws -> AppPreferences?
    func savePreferences(_ preferences: AppPreferences) throws
}

public protocol FeedbackService {
    func playMove()
    func playCapture()
    func playVictory()
    func vibrate()
}

public protocol MonetizationService {
    func showRewardedAd(placement: String) async -> Bool
    func showInterstitial(placement: String) async -> Bool
}

public final class InMemoryPreferencesStore: PreferencesStore {
    private var snapshot: AppPreferences?

    public init() {}

    public func loadPreferences() throws -> AppPreferences? {
        snapshot
    }

    public func savePreferences(_ preferences: AppPreferences) throws {
        snapshot = preferences
    }
}
