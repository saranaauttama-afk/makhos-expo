อ่านไฟล์ CODEX_ENGINE_REWRITE_REQUEST.md แล้วเริ่มทำ Phase 0 ก่อน ห้ามแก้ไฟล์อื่นนอก scope

# Codex Request — Makhos AI Engine Cleanup & Rewrite Prep

## Goal

Clean and prepare Thai Checkers AI engine before further training.

Main goals:

- Remove unused training / NN / self-play / MCTS code if not currently used.
- Keep only classical engine path first.
- Improve speed and stability.
- Add profiling and perft before major rewrite.
- Do not change UI.
- Do not add heuristic patch without benchmark proof.

---

## Important Rules

- Work one phase at a time.
- Do not skip phase.
- Before editing, inspect current files and summarize what will change.
- After each phase, report:
  - files changed
  - what was removed
  - what was kept
  - benchmark/test result
  - risk
- If test fails, stop and explain.
- Do not rewrite the whole engine blindly.

---

## Current Engine Files To Inspect First

```text
src/coreClaude/movegen.ts
src/coreClaude/eval.ts
src/coreClaude/position.ts
src/coreClaude/bitboards.ts
src/coreClaude/search/alphabeta.ts
src/coreClaude/search/tt.ts
src/coreClaude/search/zobrist.ts
src/coreClaude/search/repetition.ts
src/coreClaude/search/levelPolicy.ts
scripts/aiBenchmark.ts
scripts/tacticalSuite.ts
scripts/ruleInvariantSuite.ts
package.json

Phase 0 — Clean Training / NN / MCTS Code
Task

Search the repo for unused training-related code.

Search keywords:

train
training
selfplay
self-play
reinforcement
rl
model
nn
neural
weights
checkpoint
dataset
experience
replay
mcts
uct
policy
value
onnx
Remove only if unused by current app / benchmark

Likely removable:

scripts/train*.ts
scripts/selfplay*.ts
scripts/generateDataset*.ts
scripts/replay*.ts
src/**/mcts*.ts
src/**/uct*.ts
src/**/nn*.ts
src/**/model*.ts
src/**/network*.ts
src/**/policy*.ts
src/**/value*.ts
models/
weights/
checkpoints/
training-data/
Keep
src/coreClaude/movegen.ts
src/coreClaude/eval.ts
src/coreClaude/search/alphabeta.ts
src/coreClaude/search/tt.ts
src/coreClaude/search/zobrist.ts
scripts/aiBenchmark.ts
scripts/tacticalSuite.ts
scripts/ruleInvariantSuite.ts
Also clean package.json

Remove unused dependencies only if no remaining import uses them:

onnxruntime-react-native
onnxruntime-node

Do not remove them if any live code still imports them.

Validate

Run:

npm run test:rules
npm run test:tactical
npm run bench:ai:fresh
Phase 1 — Add AI Hotspot Profiler

Create:

scripts/profileAiHotspots.ts

Measure:

elapsedMs
nodes
qnodes
nodes/sec
generateMoves calls
applyMove calls
evaluate calls
hashPosition calls
TT hit/miss
TT hit rate
average branching factor
approx Move object allocation count

Add npm script:

"profile:ai": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/profile-ai scripts/profileAiHotspots.ts && node .tmp/profile-ai/scripts/profileAiHotspots.js"

Do not optimize yet. Just measure.

Phase 2 — Add Perft Correctness Test

Create:

scripts/perftThaiCheckers.ts

Add:

perft(pos, depth)
divide(pos, depth)

Must cover:

initial position
forced capture
max capture rule
men multi-capture
king fly capture
king multi-capture
promotion after capture
side -1 mirror cases

Add npm script:

"test:perft": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/perft scripts/perftThaiCheckers.ts && node .tmp/perft/scripts/perftThaiCheckers.js"

No movegen rewrite until perft exists.

Phase 3 — Rewrite Movegen Allocation Safely

Current problem:

generateMoves creates many Move objects
multi-capture uses array copies like [...caps], [...path]
search calls generateMoves many times
React Native / JS GC becomes bottleneck

Add backward-compatible API:

generateMoves(pos): Move[]
generateMovesInto(pos, out: Move[]): Move[]
generateCapturesInto(pos, out: Move[]): Move[]

Rules:

Existing UI must still work.
Search should gradually use generateMovesInto.
Reuse arrays where possible.
Avoid allocating new arrays inside hot loop.
Avoid STEPS[cur].find inside ray loops if possible.
Precompute directional next table in bitboards.ts if needed.

Do not change game rules.

After rewrite, run:

npm run test:rules
npm run test:tactical
npm run test:perft
npm run profile:ai
npm run bench:ai:fresh

Report before/after:

elapsedMs
nodes/sec
generateMoves calls
allocation estimate
tactical solve rate
blunder rate
Phase 4 — TT / Zobrist Collision Safety

Current TT uses 32-bit hash. Add verification key.

Preferred low-risk approach:

hashPosition(pos) -> keep existing 32-bit index key
verifyHashPosition(pos) -> second 32-bit key
TT stores both key and verifyKey
TT hit only valid when both match

Update:

src/coreClaude/search/zobrist.ts
src/coreClaude/search/tt.ts
src/coreClaude/search/alphabeta.ts

Add tests:

same position same hash
side to move changes hash
different piece placement changes hash
random collision smoke test
Phase 5 — Split Root Safety Heuristics

Do not delete immediately.

Move root override logic from alphabeta.ts into:

src/coreClaude/search/rootSafety.ts

Add feature flags:

interface SearchOptions {
  enableRootSafety?: boolean;
  enableTrapOverride?: boolean;
  enableDiversification?: boolean;
}

Default:

hard/expert:
- diversification off
- trap override off or conservative
- root safety only if benchmark proves useful

Benchmark with:

baseline
noRootSafety
noTrapOverride
noDiversification

Do not keep override if head-to-head gets worse.

Phase 6 — Evaluation Improvement Without Expensive Movegen

Improve eval only with cheap features.

Allowed:

breakthrough / promotion race
trapped king
edge king penalty
king ray mobility using precomputed rays
tempo / opposition approximation
cheap hanging piece approximation
drawish endgame detection

Forbidden:

eval must not call generateMoves
eval must not allocate large arrays
eval must not call deep search

Every new feature must have named weight constants.

Phase 7 — Small Endgame Solver

Start small only.

Implement:

<= 4 pieces exact/lazy cache
root probe only
no tablebase probe in hot negamax nodes

Test positions:

2 kings vs 1 king
king + man vs king
king vs 2 men
men race
all kings drawish cases
Phase 8 — Arena Gate

Create:

scripts/arenaGate.ts

Compare:

baseline engine vs candidate engine

Options:

--games 100
--max-plies 240
--seed fixed

Pass rule:

candidate score >= 52%
tactical severe blunder rate not worse
avg move time not worse by more than 15%

Add npm script:

"gate:ai": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/arena scripts/arenaGate.ts && node .tmp/arena/scripts/arenaGate.js"
Phase 9 — Benchmark Summary Markdown

Update benchmark to write:

.tmp/benchmarks/ai-benchmark-latest.json
.tmp/benchmarks/ai-benchmark-summary.md

Summary must include:

solve rate
blunder rate
p95 ms
nodes/sec
top failed cases
before/after comparison
Phase 10 — Stop

After Phase 10:

Do not add training back yet.

Next possible future work only after engine is stable:

opening book
larger endgame tablebase
NN eval
MCTS / PUCT
self-play training
Final Rule

Do not make the AI better by adding random tactical patches.

Only accept changes that improve:

correctness
speed
benchmark
arena result
stability