# Makhos — Thai Checkers AI

React Native app for Thai Checkers (หมากฮอส) with two AI agents: hand-crafted Minimax and AlphaZero (self-play neural network).

---

## Game Rules

Thai Checkers on an 8×8 board, dark squares only (32 playable squares).

| Rule | Description |
|---|---|
| **Pieces** | 8 men per side, start on opposite ends |
| **Men move** | Diagonally forward only (1 step) |
| **Men capture** | Diagonally forward, must jump over enemy |
| **Kings move** | Any diagonal direction, any distance (fly) |
| **Kings capture** | Any diagonal direction, fly to first enemy, land immediately behind |
| **Forced capture** | Must capture if any capture is available |
| **Max-capture** | Must take the longest capture chain available |
| **Multi-capture** | Continue capturing from landing square if possible |
| **Promotion** | Man reaching last rank becomes King |
| **Draw** | `halfmoveClock ≥ 20` when each side has ≤ 2 pieces |

### Board Layout

```
row 0 (top screen)     sq  0– 3   ← P2 starts, moves DOWN, promotes at sq 28–31
row 1                  sq  4– 7
row 2                  sq  8–11
row 3                  sq 12–15
row 4                  sq 16–19
row 5                  sq 20–23
row 6                  sq 24–27
row 7 (bottom screen)  sq 28–31   ← P1 starts, moves UP,   promotes at sq  0– 3
```

Squares are indexed row-major on dark squares only (where `(row+col) & 1 == 1`).

---

## Architecture

### Core Engine (`src/coreClaude/`)

| File | Purpose |
|---|---|
| `bitboards.ts` | 32-square bitboard helpers, adjacency table |
| `position.ts` | Position struct, draw detection |
| `movegen.ts` | Move generation (forced capture, max-capture, multi-capture, king fly) |
| `eval.ts` | `handEvaluate` — Texel-tuned static evaluation |
| `search/alphabeta.ts` | Alpha-beta search with move ordering |
| `search/tt.ts` | Transposition table (Zobrist hash) |
| `search/zobrist.ts` | Zobrist hashing |
| `search/openingBook.ts` | Opening book |
| `search/endgameTablebase.ts` | Endgame tablebase |
| `azFeatures.ts` | Feature extraction for AlphaZero (128-dim, board flip) |
| `azMcts.ts` | MCTS with PUCT selection (C_PUCT = 1.5) |
| `azNet.ts` | ONNX model loader (onnxruntime-react-native) |

### Minimax Evaluation (`handEvaluate` v3)

Texel-tuned on 300 games / 18k positions:

| Component | Detail |
|---|---|
| Material | Man = 100 cp, King = 280→380 cp (scales with endgame) |
| PSQT | Row advancement + column safety (men), centrality (kings) |
| Mobility | Step-squares available (fast, no generateMoves call) |
| Back-rank guard | Reward men protecting own promotion rank |
| Simplification | Reward trading when ahead |

---

## AlphaZero Pipeline

### Network Architecture

```
Input: 128-dim float32 (4 planes × 32 squares, current-player relative)
  ↓  Stem: Linear(128→256) + BN + ReLU
  ↓  Trunk: 4 × ResBlock(256)
  ├─ Policy head: Linear(256→128) + BN + ReLU → Linear(128→1024)  [logits]
  └─ Value head:  Linear(256→64)  + BN + ReLU → Linear(64→1) + Tanh
```

~800k parameters. Exported to ONNX for mobile inference.

### Board Flip (Critical)

When P2 is to move, all square indices are remapped `sq → 31-sq` (180° rotation).
This ensures the network always sees the board from the same perspective — "my pieces at the bottom, moving up" — regardless of side. Without this, the network only learns P1's view.

### Feature Encoding

```
x[  0.. 31] = my men
x[ 32.. 63] = my kings
x[ 64.. 95] = enemy men
x[ 96..127] = enemy kings
```

Move encoding: `from_sq * 32 + to_sq` (1024 slots). Flipped for P2: `(31-from)*32 + (31-to)`.

### Training Config

| Parameter | Value |
|---|---|
| Self-play games / iter | 100 |
| MCTS sims / move | 200 |
| Temperature cutoff | 12 plies |
| Replay buffer | 100,000 samples (rolling) |
| Batch size | 256 |
| Train steps / iter | `min(500, buf // 256)` |
| Optimizer | Adam lr=1e-3, wd=1e-4 |
| LR schedule | CosineAnnealing T=200 |
| C_PUCT | 1.5 |
| Eval interval | every 10 iters |
| Win threshold | 55% vs previous best |

### Training Versions

| Version | Drive folder | Notes |
|---|---|---|
| v1 | `makhos_az` | No board flip — P1 only |
| v2 | `makhos_az_v2` | Board flip added, no max-capture |
| **v3** | `makhos_az_v3` | Board flip + **max-capture rule enforced** ← current |

### Running Training (Google Colab)

1. Upload 4 files to `MyDrive/makhos_az_v3/`: `makhos_engine.py`, `network_az.py`, `mcts_az.py`, `train_az.py`
2. Open `train_az.py` as Colab notebook
3. Run Cell 1 → 2 → 3 → 4 → 5
4. Resume: just re-run Cell 1 → 5 — loads `best.pt` + optimizer + replay buffer automatically

### Exporting to ONNX (Cell 7)

```python
EXPORT_ITER = 'iter_0099'   # change to desired checkpoint
```
Run Cell 7 -> export the chosen checkpoint as ONNX.

Model plug-in point for gameplay:
1. Put ONNX file in `assets/models/` (for example `iter_0099.onnx`)
2. Register it in `src/coreClaude/azModelCatalog.ts`
3. Open `Settings` -> `AI Model` and select that model

The active model is reloaded automatically at runtime after selection.
Archived experiment exports stay in `data_old/` and should not be referenced by the mobile app.

---

## Battle Test Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `battleTest.ts` | AZ (with flip) vs Minimax-N |
| `battleTestOld.ts` | OLD AZ (no flip) vs Minimax-5 |
| `battleAZvsAZ.ts` | NEW AZ (flip) vs OLD AZ (no flip) |

```bash
# Compile and run
npx tsc --module commonjs --moduleResolution node --target es2017 \
  --outDir ./tmp_battle --esModuleInterop true --skipLibCheck true \
  scripts/battleTest.ts src/coreClaude/azFeatures.ts \
  src/coreClaude/position.ts src/coreClaude/movegen.ts \
  src/coreClaude/bitboards.ts src/coreClaude/eval.ts \
&& node ./tmp_battle/scripts/battleTest.js
```

### Key Benchmark Results

| Matchup | Result | Notes |
|---|---|---|
| MM5 vs MM5 | P2 wins 100% | P2 structural advantage — not a bug |
| MM3 vs MM3 | P2 wins 100% | Same at any depth |
| OLD v1 (iter_0059) vs MM5 | 50% | P1 strong, P2 broken |
| NEW v2 (iter_0049) vs MM5 | 0% | Needs more training |
| NEW v2 (iter_0049) vs OLD v1 | **100%** | Flip fix works |

---

## Known Issues / Future Work

| Issue | Status | Priority |
|---|---|---|
| `best.pt` overwrite bug | `curr_net` overwrites `best.pt` every iter — use `iter_XXXX.pt` for eval | Medium |
| Separate `latest.pt` / `best.pt` | Resume loads same file into both | Low |
| Python minimax weaker than TS | Python uses pure material; TS uses full eval | Low (eval only) |
| Eval sample size | 20 games → high variance; should be 50–100 | Medium |
| Gmail password in train_az.py | Should be env variable | High (security) |

---

## Development Notes

- P2 always wins vs minimax at any depth tested — this is a **real game-theory property**, not a bug (confirmed by reading Java original source and Thai Checkers references)
- Max-capture is enforced from **v3 onwards** — earlier models were trained without it
- C_PUCT = 1.5 in both training (Python) and inference (TypeScript) from v3 onwards

---

## Product UI Plan

This section is a practical product/UI checklist for turning Makhos into a shippable mobile game with ads plus a paid "remove ads" option.

### Core Difficulty Structure

Recommended difficulty ladder for the new UI:

| Level | Goal |
|---|---|
| Beginner | Very forgiving, fast turns |
| Easy | Casual players |
| Medium | Default mode |
| Hard | Strong AI with more thinking time |
| Master | Strongest practical mobile setting |

### Minimum Screen Set

These are the screens worth planning before final art production:

| Screen | Why it exists |
|---|---|
| Splash / Boot | Load assets, model, and saved settings |
| Home | Main entry point: Play, Settings, Shop, Profile/Future events |
| Mode Select | Human vs AI, Human vs Human, Arena / analysis later |
| Difficulty Select | Choose one of the 5 AI levels |
| Side Select | Let player choose first/second side when playing AI |
| Match Screen | Main board UI, timers, hints, pause, undo rules |
| Pause Sheet | Resume, restart, sound, quit |
| Result Screen | Win/loss/draw, rematch, next action, ad offer if appropriate |
| Shop / No Ads | Buy ad removal, restore purchase, explain benefits clearly |
| Settings | Sound, vibration, language, animation speed, privacy links |
| Privacy / Terms | Needed for store submission and ad SDK compliance |

### Monetization Screens

If the game uses ads but allows players to pay to remove them, these screens are the practical minimum:

| Screen | Purpose |
|---|---|
| No Ads Offer | Explain one-time purchase or subscription to remove forced ads |
| Restore Purchase | Required for iOS if purchases are supported |
| Ad-Free Confirmation | Simple success state after purchase |
| Rewarded Ad Prompt | Optional: watch ad for retry, hint, theme, coins, etc. |
| Ad Error / Unavailable | Fallback when ad inventory is not available |

### Recommended Player Flow

For an ad-supported strategy game, a clean first version can be:

1. Splash
2. Home
3. Mode Select
4. Difficulty Select
5. Side Select
6. Match Screen
7. Result Screen

From Home, Settings and Shop should always be reachable in one tap.

### Ad Strategy

For better retention, prefer this order:

| Ad Type | Recommendation |
|---|---|
| Banner | Avoid on gameplay board if it makes the board feel cheap or cramped |
| Interstitial | Show sparingly after match results, not during active play |
| Rewarded | Best option for voluntary monetization |
| Remove Ads Purchase | Offer permanently from Settings, Result screen, and Shop |

Suggested rule:

- Free players may see interstitial ads after some completed matches
- Players who buy "No Ads" stop seeing forced ads
- Rewarded ads can still remain optional even for paying users if tied to bonuses

### UI Skeleton Guidance

Before final art, it helps to design these reusable blocks:

| Block | Notes |
|---|---|
| Hero header | Logo, subtitle, seasonal/event art zone |
| Large mode cards | Big tap targets for AI / local play / future online mode |
| Difficulty cards | 5 stacked cards with fixed image placeholders |
| Bottom CTA | Consistent primary action button height across screens |
| Modal sheet | Reusable for pause, confirm quit, ad prompts, purchase prompts |
| Result card | Shared win/loss/draw presentation pattern |

### Store / Compliance Notes

If ads or in-app purchases are added, plan for these non-game pages too:

- Privacy Policy
- Terms of Service
- Restore Purchases
- Data / tracking consent if required by the ad SDK or platform rules

### Suggested Next UI Work

If the immediate goal is to build the new interface first, the best order is:

1. Home screen skeleton
2. Difficulty select skeleton with 5 levels
3. Match screen skeleton
4. Result screen skeleton
5. Shop / No Ads screen skeleton
6. Settings screen skeleton

cd "d:\My App\makhos-v2"
$env:REACT_NATIVE_PACKAGER_HOSTNAME="192.168.0.10"
npx expo start --lan -c

cd "D:\My App\PhiKinHuaV6"
$env:REACT_NATIVE_PACKAGER_HOSTNAME="192.168.0.10"
npx expo start --lan -c

cd /d "d:\My App\makhos-v2"
python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v5" --checkpoint iter_0094

cd /d "d:\My App\makhos-v2"
python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v5" --checkpoint iter_0099

---

## Release Checklist

Use this list before shipping a preview or production build.

### Product / UX

- [ ] Home screen text is correct in both TH/EN
- [ ] Language switch persists after app restart
- [ ] Last setup config (mode / side / level) persists after app restart
- [ ] Sound toggle works (OFF = no sound)
- [ ] Vibration toggle works (OFF = no vibration)
- [ ] Hint / Undo / Surrender flows work end-to-end

### Ads / Monetization

- [ ] Interstitial cadence is correct (`every 5 matches`)
- [ ] No Ads purchase hides forced interstitial ads
- [ ] Rewarded actions (coins / hint / undo) are user-triggered only
- [ ] Account footer shows company + app version

### Technical

- [ ] `npx tsc --noEmit` passes
- [ ] Android/iOS open without runtime crash on real device
- [ ] Settings and wallet values persist on mobile (`AsyncStorage` working)

### Store Readiness

- [ ] Privacy Policy URL ready
- [ ] App metadata (title, short/full description, screenshots, icon) ready
- [ ] Build `preview` passed on device
- [ ] Build `production` generated successfully
