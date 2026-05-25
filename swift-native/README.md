# Makhos Swift Native Rewrite

This workspace is the Swift-native rewrite foundation for the Expo app.

Current scope:
- `Sources/MakhosCore`: Thai checkers core rules ported from TypeScript
- `Sources/MakhosNativeUI`: SwiftUI rewrite scaffolding for Home / Setup / Arena / Account flow
- `Tests/MakhosCoreTests`: curated parity coverage based on the TypeScript rule suite

Validation:

```bash
cd /tmp/workspace/saranaauttama-afk/makhos-expo/swift-native
swift test
```

Notes:
- The existing Expo/React Native app remains in the repository as the behavior reference during the rewrite.
- The current Swift milestone ports the board rules and move generation first; AI search, monetization, and native iOS packaging follow next.
