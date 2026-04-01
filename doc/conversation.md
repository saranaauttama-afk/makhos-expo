# Makhos AI — Project Overview for Code Review

This document describes the full ML pipeline and engine for **Makhos** (Thai Checkers), written for a code reviewer (Codex) to understand the system and suggest improvements.

---

## 1. The Game — Thai Checkers (Makhos)

- 8×8 board, dark squares only → **32 playable squares** (indexed 0–31, row-major)
- Each side starts with **8 men**
- Men move **diagonally forward only**, capture forward only (unlike Western checkers)
- Kings fly any number of squares diagonally in any direction (like bishops)
- Forced capture: must capture if possible; max-capture rule applies
- Draw: **halfmoveClock ≥ 100** (50 moves without capture or promotion)

### Board Layout

```
row 0 (top)    squares  0– 3   ← P2 starts here, moves DOWN (DL/DR), promotes at 28–31
row 1          squares  4– 7
row 2          squares  8–11
row 3          squares 12–15
row 4          squares 16–19
row 5          squares 20–23
row 6          squares 24–27
row 7 (bottom) squares 28–31   ← P1 starts here, moves UP (UL/UR), promotes at 0–3
```

- **P1 (side=1)** : bottom of screen, moves up, promotes at row 0
- **P2 (side=-1)**: top of screen, moves down, promotes at row 7

### Important: P2 Structural Advantage

In our minimax-vs-minimax tests, **P2 wins 100% of games** at both depth 3 and depth 5. The second player has a structural advantage (possibly related to tempo) in this game's opening theory. This must be kept in mind when interpreting AI vs minimax win rates — a 50% result with alternating sides tells us nothing about AI skill if P2 always wins.

---

## 2. Minimax Engine

**Location:** `src/coreClaude/movegen.ts`, `src/coreClaude/eval.ts`

### Move Generation

- Bitboard-based (`BB = uint32`), stored in `Position`:
  ```
  { side, p1Men, p1Kings, p2Men, p2Kings, halfmoveClock }
  ```
- Pre-computed adjacency table `STEPS[sq]` — 4 directions (UL/UR/DL/DR) × each square
- Multi-capture chains handled recursively
- Forced capture: if any capture exists, only captures are returned

### Evaluation Function — `handEvaluate` (v3)

Texel-tuned on 18k positions, 300 games. Components (all side-relative, negamax):

| Component | Description | Weight |
|---|---|---|
| **Material** | men ×100, kings ×(280→380) | varies by endgame phase |
| **PSQT** | row advancement + column safety for men; centrality for kings | tuned |
| **Mobility** | step-squares available (fast approximation, no generateMoves call) | ×1 |
| **Back rank guard** | reward men on own back rank (promotion stoppers) | ×5, scaled to 0 in endgame |
| **Simplification** | reward trading when ahead | ×6 per pair traded |
| **King endgame proximity** | kings hunt enemy men when winning | ×3 per distance unit |

Notable: `protectedMenBonus` and `kingEndgameScore` were disabled after Texel tuning found weight ~0.

### Alpha-Beta Search

- Standard negamax alpha-beta, no move ordering, no transposition table
- **Depth 3** and **Depth 5** used for evaluation
- Used only for: (a) the `hand` agent in the app, (b) evaluation during AZ training checkpoints

---

## 3. AlphaZero Pipeline

### 3a. Feature Encoding — `get_features` / `getFeatures`

**128-dim float32 vector**, current-player-relative:

```
x[  0.. 31] = 1 if MY man at square i
x[ 32.. 63] = 1 if MY king at square i
x[ 64.. 95] = 1 if ENEMY man at square i
x[ 96..127] = 1 if ENEMY king at square i
```

**Critical fix (board flip):** When P2 is to move, all squares are remapped `sq → 31-sq` (180° rotation). This ensures the network always sees "my pieces at the bottom, moving up" — a single perspective regardless of side.

Without flip: network only learns P1's view → weak as P2 (confirmed: old model wins 0% as P2 vs minimax-5).
With flip: both sides look identical to the network → symmetric learning.

### 3b. Move Encoding

Policy head outputs **1024 logits** for all `from_sq * 32 + to_sq` combinations.

With flip active (P2's turn): `(31 - from_sq) * 32 + (31 - to_sq)` — mirrors the move on the flipped board.

### 3c. Network Architecture

**File:** `colab/network_az.py`, `src/coreClaude/azNet.ts`

```
Input: 128-dim float32
  ↓
Stem: Linear(128→256) + BatchNorm + ReLU
  ↓
Trunk: 4 × ResBlock(256)   [each: Linear→BN→ReLU→Linear→BN + skip]
  ↓
Policy head: Linear(256→128) + BN + ReLU → Linear(128→1024)   [logits]
Value head:  Linear(256→64)  + BN + ReLU → Linear(64→1) + Tanh  [∈ -1,1]
```

Total params: ~800k. Exported to ONNX for inference in the React Native app via `onnxruntime-react-native`.

### 3d. MCTS

**Files:** `colab/mcts_az.py`, `src/coreClaude/azMcts.ts`

- **PUCT selection:** `-child.Q + C_PUCT * child.P * sqrt(parent.N) / (1 + child.N)`
- `C_PUCT = 1.0`
- **Negamax convention:** value is negated going up the tree
- **Terminal detection:** no moves → `-1` (current player loses), draw-by-inactivity → `0`
- **Temperature:** first 12 plies use softmax sampling from visit counts; after → argmax
- **200 simulations per move** during self-play and inference

### 3e. Training Loop

**File:** `colab/train_az.py`

```
For each iteration:
  1. Self-play (100 games, curr_net vs itself, MCTS 200 sims)
     → generates (features, pi, z) tuples
     → pi = MCTS visit count distribution
     → z = game outcome from that position's side (+1 win, -1 loss, 0 draw)

  2. Replay buffer: rolling window of 100,000 most recent samples

  3. Train: 390 steps × batch 256
     Loss = CrossEntropy(policy_logits, pi) + MSE(value, z)
     Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
     Scheduler: CosineAnnealingLR(T_max=200, eta_min=1e-5)

  4. Every 10 iterations — checkpoint eval:
     a. curr vs best net (20 games, alternating P1/P2) → update if win_rate ≥ 55%
     b. curr vs random (20 games)
     c. curr vs minimax-3 (20 games)
     d. curr vs minimax-5 (20 games)
     e. Save iter_XXXX.pt, email notification

  5. Save curr_net + optimizer + scheduler + replay_buffer every iter (for resume)
```

### 3f. Current Training Status

- **Run:** `makhos_az_v2` (Google Drive) — fresh run with board flip
- **GPU:** A100 on Google Colab Pro+ (~60 min/iter, 150 CU remaining ≈ 28hrs)
- **Current iter:** ~49–50 (as of 2026-03-28)
- **v_loss trend:** 0.31 → 0.29 → 0.25 → 0.21 (still decreasing, not plateaued)
- **p_loss trend:** 1.72 → 1.45 → 1.38 → 1.32

### 3g. Evaluation Results (iter_0049, TypeScript battleTest)

| Matchup | Result | Notes |
|---|---|---|
| NEW (iter_0049) vs minimax-3 | 50% | Entirely P2 structural advantage |
| NEW (iter_0049) vs minimax-5 | 0% | Model not strong enough yet |
| OLD (iter_0059, no flip) vs minimax-5 | 50% | P1 only — wins as P1, loses as P2 |
| NEW (iter_0049) vs OLD (iter_0059) | **100%** | New model wins all 100 games |

**Note:** Python minimax (used during training eval) appears weaker than TypeScript minimax (used in the app), so Python checkpoint results (e.g. "75% vs mm5") don't translate 1:1.

---

## 4. App Integration

- **Framework:** React Native + Expo
- **Agents available:** `random`, `hand` (minimax), `az` (AlphaZero MCTS)
- **ONNX inference:** `onnxruntime-react-native`, model bundled as `data/iter_0049.onnx`
- **Preloading:** `preloadAZModel()` called on screen mount to avoid first-move latency

---

## 5. Known Issues / Open Questions for Review

1. **P2 structural advantage** — P2 wins 100% in minimax-vs-minimax at both depth 3 and 5. Is this a bug in the game rules implementation, or real Thai Checkers tempo advantage? Worth investigating `movegen.ts` and initial position.

2. **Eval reliability** — 20-game eval has very high variance (results swing 0%↔100% between checkpoints). Should N_EVAL_GAMES be increased to 50–100?

3. **Python vs TypeScript minimax discrepancy** — Python minimax-5 appears significantly weaker than TypeScript minimax-5 for same depth. Could be `hand_eval` differences. Worth cross-validating.

4. **No move ordering in minimax** — alpha-beta has no move ordering heuristic (no killer moves, no MVV-LVA). Adding capture-first ordering could significantly improve effective depth.

5. **No transposition table** — same positions can be evaluated multiple times in minimax. A simple Zobrist hash table could speed up search substantially.

6. **MCTS sims = 200** — is this enough for a 32-square board? Could we increase sims at the cost of slower self-play?

7. **Self-play quality** — early iterations have random/near-random play. Curriculum-style warmstart (using minimax to generate initial samples) could accelerate early learning.

8. **Network size** — 256 hidden, 4 ResBlocks (~800k params). For 32 squares this might be oversized. A smaller network (128 hidden, 2 ResBlocks) might train faster and overfit less.

9. **best.pt overwrite bug** — at end of each iteration, `curr_net.save(BEST_NET_PATH)` overwrites `best.pt` regardless of whether the new net is better. This means `best.pt` is always `curr_net`, not the actual best checkpoint. The `iter_XXXX.pt` files are the correct checkpoints to use.

10. **King value scaling** — king value scales from 280 (opening) to 380 (endgame). Is this calibrated correctly for Makhos? Thai kings fly freely (unlike western kings) so they may be worth more than 2.8 men even in midgame.
