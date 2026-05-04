# Makhos Engine - Development Checklist

**Branch:** `tuneGPT`
**Last Updated:** 2026-05-04
**Current Phase:** Phase 0 (Clean unused code)

---

## Phase 0: Clean Training / NN / MCTS Code

**Goal:** ลบโค้ดที่ไม่ใช้งาน (training, NN, MCTS, self-play)

### Tasks

- [ ] **Search for unused training-related code**
  - [ ] Search keywords: train, training, selfplay, self-play, reinforcement, rl
  - [ ] Search keywords: model, nn, neural, weights, checkpoint
  - [ ] Search keywords: dataset, experience, replay
  - [ ] Search keywords: mcts, uct, policy, value, onnx

- [ ] **Remove unused script files**
  - [ ] `scripts/train*.ts`
  - [ ] `scripts/selfplay*.ts`
  - [ ] `scripts/generateDataset*.ts`
  - [ ] `scripts/replay*.ts`

- [ ] **Remove unused source files**
  - [ ] `src/**/mcts*.ts`
  - [ ] `src/**/uct*.ts`
  - [ ] `src/**/nn*.ts`
  - [ ] `src/**/model*.ts`
  - [ ] `src/**/network*.ts`
  - [ ] `src/**/policy*.ts` (ยกเว้น levelPolicy.ts)
  - [ ] `src/**/value*.ts`

- [ ] **Remove unused folders**
  - [ ] `models/`
  - [ ] `weights/`
  - [ ] `checkpoints/`
  - [ ] `training-data/`

- [ ] **Clean package.json dependencies**
  - [ ] Check if `onnxruntime-react-native` is still imported
  - [ ] Check if `onnxruntime-node` is still imported
  - [ ] Remove unused dependencies only if safe

### Validation

- [ ] Run `npm run test:rules` - must pass
- [ ] Run `npm run test:tactical` - must pass
- [ ] Run `npm run bench:ai:fresh` - must not regress

### Report Deliverables

- [ ] List of files removed
- [ ] List of files kept
- [ ] Test results summary
- [ ] Benchmark comparison (before/after)
- [ ] Risk assessment

---

## Phase 1: Add AI Hotspot Profiler

**Goal:** สร้างเครื่องมือวัดประสิทธิภาพของ AI engine

### Tasks

- [ ] **Create profiler script**
  - [ ] Create `scripts/profileAiHotspots.ts`
  - [ ] Measure `elapsedMs`
  - [ ] Measure `nodes` and `qnodes`
  - [ ] Measure `nodes/sec`
  - [ ] Count `generateMoves calls`
  - [ ] Count `applyMove calls`
  - [ ] Count `evaluate calls`
  - [ ] Count `hashPosition calls`
  - [ ] Track `TT hit/miss` and hit rate
  - [ ] Calculate `average branching factor`
  - [ ] Estimate `Move object allocation count`

- [ ] **Add npm script**
  - [ ] Add `"profile:ai"` to package.json scripts

### Validation

- [ ] Run profiler on sample positions
- [ ] Verify all metrics are collected
- [ ] Generate baseline profile report

---

## Phase 2: Add Perft Correctness Test

**Goal:** สร้างเครื่องมือทดสอบความถูกต้องของ move generation

### Tasks

- [ ] **Create perft script**
  - [ ] Create `scripts/perftThaiCheckers.ts`
  - [ ] Implement `perft(pos, depth)` function
  - [ ] Implement `divide(pos, depth)` function

- [ ] **Add test cases**
  - [ ] Initial position
  - [ ] Forced capture positions
  - [ ] Max capture rule positions
  - [ ] Men multi-capture
  - [ ] King fly capture
  - [ ] King multi-capture
  - [ ] Promotion after capture
  - [ ] Side -1 mirror cases

- [ ] **Add npm script**
  - [ ] Add `"test:perft"` to package.json scripts

### Validation

- [ ] Run perft on all test positions
- [ ] Verify node counts match expected values
- [ ] Document expected perft results

---

## Phase 3: Rewrite Movegen Allocation Safely 🎯

**Goal:** Optimize move generation ให้เร็วขึ้น 2-3 เท่า

### Tasks

#### 3.1 Precompute Direction Tables

- [ ] **Update bitboards.ts**
  - [ ] Add `DIRS` constant array
  - [ ] Add `DIR_INDEX` mapping
  - [ ] Create `NEXT: Int8Array[]` table
  - [ ] Create `RAYS: number[][][]` table
  - [ ] Implement `buildFastDirectionalTables()` function

#### 3.2 Add Backward Compatible APIs

- [ ] **Keep existing APIs**
  - [ ] Verify `generateMoves(p: Position): Move[]` signature unchanged
  - [ ] Verify `hasCapturesAvailable(p: Position): boolean` signature unchanged
  - [ ] Verify `applyMove(p: Position, m: Move): Position` signature unchanged

#### 3.3 Add New Faster APIs

- [ ] **Add new functions in movegen.ts**
  - [ ] Add `generateMovesInto(p: Position, out: Move[]): Move[]`
  - [ ] Add `generateCapturesInto(p: Position, out: Move[]): Move[]`
  - [ ] Make `generateMoves()` wrapper call `generateMovesInto(p, [])`

#### 3.4 Optimize Move Generation

- [ ] **Add helper constants**
  - [ ] Add `EMPTY_CAPTURED = Object.freeze([])`
  - [ ] Add direction constants: UL, UR, DL, DR
  - [ ] Add `P1_DIR_A`, `P1_DIR_B`, `P2_DIR_A`, `P2_DIR_B`
  - [ ] Add `LAST_RANK_P1`, `LAST_RANK_P2`

- [ ] **Optimize quiet moves**
  - [ ] Use `EMPTY_CAPTURED` for all quiet moves
  - [ ] Implement `pushQuiet()` helper
  - [ ] Implement `addMenQuietMoves()` using NEXT table
  - [ ] Implement `addKingQuietMoves()` using RAYS table

- [ ] **Optimize capture moves**
  - [ ] Add `MAX_CHAIN` constant
  - [ ] Implement `pushCapture()` helper with buffers
  - [ ] Rewrite `genMenCapturesFromFast()` with `capsBuf`/`pathBuf`
  - [ ] Rewrite `genKingCapturesFromFast()` with `capsBuf`/`pathBuf`
  - [ ] Replace `[...caps]`, `[...path]` with buffer copies

- [ ] **Optimize hasCapturesAvailable()**
  - [ ] Use NEXT/RAYS instead of generator/find
  - [ ] Return immediately when first capture found
  - [ ] No Move object allocation
  - [ ] Implement `hasMenCaptureFrom()` helper

#### 3.5 Optimize applyMove()

- [ ] **Rewrite with explicit branches**
  - [ ] Replace dynamic property access with if/else branches
  - [ ] Separate P1 move handling
  - [ ] Separate P2 move handling
  - [ ] Optimize bitboard operations

#### 3.6 Remove Old Code

- [ ] **Clean up generators**
  - [ ] Remove `function *ray()` generator
  - [ ] Remove `nextInDir()` with STEPS.find()
  - [ ] Update all references to use new tables

### Validation

- [ ] Run `npm run test:rules` - must pass
- [ ] Run `npm run test:tactical` - must pass
- [ ] Run `npm run test:perft` - must pass (if exists)
- [ ] Run `npm run profile:ai` - compare before/after
- [ ] Run `npm run bench:ai:fresh` - must not regress

### Report Deliverables

- [ ] Files changed list
- [ ] What changed summary
- [ ] Correctness results (all tests pass)
- [ ] Performance comparison:
  - [ ] elapsedMs before/after
  - [ ] nodes/sec before/after
  - [ ] generateMoves calls before/after
  - [ ] Allocation estimate before/after
  - [ ] Tactical solve rate before/after
  - [ ] Blunder rate before/after
- [ ] Estimated speedup (target: 2-3x)

### Acceptance Criteria

- [x] Must preserve Thai Checkers rules
- [ ] `npm run test:rules` ผ่าน 100%
- [ ] `npm run test:tactical` ผ่าน 100%
- [ ] `npm run bench:ai:fresh` ไม่แย่ลง
- [ ] `generateMoves(pos)` ให้ผลเหมือนเดิม
- [ ] `generateMovesInto(pos, out)` reuse out array ได้
- [ ] `hasCapturesAvailable` ไม่ allocate Move objects
- [ ] `applyMove` ไม่ใช้ dynamic property access
- [ ] Code เร็วขึ้นอย่างน้อย 2x

---

## Phase 4: TT / Zobrist Collision Safety

**Goal:** ป้องกัน hash collision

### Tasks

- [ ] **Add verification key**
  - [ ] Keep `hashPosition(pos)` as 32-bit index key
  - [ ] Add `verifyHashPosition(pos)` as 32-bit verify key

- [ ] **Update TT storage**
  - [ ] Update TT entry to store both `key` and `verifyKey`
  - [ ] Update TT probe to check both keys match

- [ ] **Update files**
  - [ ] Update `src/coreClaude/search/zobrist.ts`
  - [ ] Update `src/coreClaude/search/tt.ts`
  - [ ] Update `src/coreClaude/search/alphabeta.ts`

- [ ] **Add tests**
  - [ ] Same position → same hash
  - [ ] Side changes → different hash
  - [ ] Piece placement changes → different hash
  - [ ] Random collision smoke test

### Validation

- [ ] Run all hash tests
- [ ] Verify TT hit rate unchanged
- [ ] Run benchmark to check no regression

---

## Phase 5: Split Root Safety Heuristics

**Goal:** แยก root override logic เป็น module แยก

### Tasks

- [ ] **Create new module**
  - [ ] Create `src/coreClaude/search/rootSafety.ts`
  - [ ] Move root override logic from alphabeta.ts

- [ ] **Add feature flags**
  - [ ] Define `SearchOptions` interface
  - [ ] Add `enableRootSafety` flag
  - [ ] Add `enableTrapOverride` flag
  - [ ] Add `enableDiversification` flag

- [ ] **Set default policies**
  - [ ] hard/expert: diversification off
  - [ ] hard/expert: trap override conservative
  - [ ] Enable root safety only if benchmark proves useful

- [ ] **Run benchmarks**
  - [ ] Baseline
  - [ ] noRootSafety
  - [ ] noTrapOverride
  - [ ] noDiversification

### Validation

- [ ] Compare all benchmark variants
- [ ] Keep only features that improve head-to-head
- [ ] Document findings

---

## Phase 6: Evaluation Improvement

**Goal:** ปรับปรุง evaluation ด้วย cheap features

### Tasks

- [ ] **Add allowed features** (cheap only)
  - [ ] Breakthrough / promotion race
  - [ ] Trapped king detection
  - [ ] Edge king penalty
  - [ ] King ray mobility (using RAYS)
  - [ ] Tempo / opposition approximation
  - [ ] Cheap hanging piece approximation
  - [ ] Drawish endgame detection

- [ ] **Ensure constraints**
  - [ ] Verify eval never calls `generateMoves`
  - [ ] Verify no large array allocation
  - [ ] Verify no deep search calls
  - [ ] Add named weight constants for all features

### Validation

- [ ] Run benchmark with each new feature
- [ ] Measure evaluation time cost
- [ ] Compare tactical solve rate
- [ ] Keep only features that improve play

---

## Phase 7: Small Endgame Solver

**Goal:** Implement ตารางท้ายเกมแบบเล็ก

### Tasks

- [ ] **Implement solver**
  - [ ] <= 4 pieces exact/lazy cache
  - [ ] Root probe only
  - [ ] No tablebase probe in negamax hot nodes

- [ ] **Add test positions**
  - [ ] 2 kings vs 1 king
  - [ ] King + man vs king
  - [ ] King vs 2 men
  - [ ] Men race
  - [ ] All kings drawish cases

### Validation

- [ ] Verify all test positions solve correctly
- [ ] Measure endgame performance improvement
- [ ] Run full benchmark

---

## Phase 8: Arena Gate

**Goal:** สร้างระบบทดสอบเปรียบเทียบ engines

### Tasks

- [ ] **Create arena script**
  - [ ] Create `scripts/arenaGate.ts`
  - [ ] Compare baseline vs candidate engine

- [ ] **Add options**
  - [ ] `--games 100`
  - [ ] `--max-plies 240`
  - [ ] `--seed fixed`

- [ ] **Define pass criteria**
  - [ ] Candidate score >= 52%
  - [ ] Tactical blunder rate not worse
  - [ ] Avg move time not worse by > 15%

- [ ] **Add npm script**
  - [ ] Add `"gate:ai"` to package.json

### Validation

- [ ] Run baseline vs baseline (should be ~50%)
- [ ] Test with known weaker/stronger engine
- [ ] Verify all metrics collected

---

## Phase 9: Benchmark Summary Markdown

**Goal:** สร้าง markdown report สำหรับ benchmark

### Tasks

- [ ] **Update benchmark output**
  - [ ] Write `.tmp/benchmarks/ai-benchmark-latest.json`
  - [ ] Write `.tmp/benchmarks/ai-benchmark-summary.md`

- [ ] **Include in summary**
  - [ ] Solve rate
  - [ ] Blunder rate
  - [ ] P95 ms
  - [ ] Nodes/sec
  - [ ] Top failed cases
  - [ ] Before/after comparison

### Validation

- [ ] Generate sample report
- [ ] Verify markdown formatting
- [ ] Check all metrics included

---

## Phase 10: Stop and Review

**Goal:** Review และทำเอกสารสรุป

### Tasks

- [ ] **Review all changes**
  - [ ] Document architecture changes
  - [ ] Update README if needed
  - [ ] Archive old benchmark reports

- [ ] **Final validation**
  - [ ] All tests pass
  - [ ] Benchmark gates pass
  - [ ] Documentation complete

- [ ] **Future work planning**
  - [ ] DO NOT add training back yet
  - [ ] Plan opening book (future)
  - [ ] Plan larger tablebase (future)
  - [ ] Plan NN eval (future)
  - [ ] Plan MCTS/PUCT (future)
  - [ ] Plan self-play training (future)

---

## Important Rules ⚠️

### DO ✅

- Work one phase at a time
- Inspect files before editing
- Report after each phase
- Run all tests after changes
- Measure before/after performance
- Only accept changes that improve: correctness, speed, benchmark, arena, stability

### DON'T ❌

- Skip phases
- Rewrite entire engine blindly
- Change UI
- Change game rules
- Add heuristic patches without benchmark proof
- Optimize without profiling first
- Add expensive eval features (no generateMoves in eval)
- Add training code back (not until Phase 10+)

---

## Test Commands

```bash
# Basic tests
npm run test:rules
npm run test:tactical
npx tsc --noEmit

# Benchmarks
npm run bench:ai                    # Quick benchmark
npm run bench:ai:fresh              # Fresh quick benchmark
npm run bench:ai:full               # Full benchmark with H2H
npm run bench:ai:full:fresh         # Fresh full benchmark
npm run bench:ai:teacher            # Teacher/deep oracle mode
npm run bench:ai:analyze            # Analyze latest results

# Profiling (after Phase 1)
npm run profile:ai

# Perft (after Phase 2)
npm run test:perft

# Arena gate (after Phase 8)
npm run gate:ai
```

---

## Current Status

**Latest Work (tuneAI branch):**
- ✅ Added hanging pieces penalty detection
- ✅ Enhanced quiescence search
- ✅ Fixed delta pruning promotion blind spots
- ✅ Rebalanced difficulty levels

**Next Immediate Task:**
- [ ] Phase 0: Clean unused training/NN/MCTS code
- [ ] Search and document all unused files
- [ ] Create removal plan
- [ ] Get approval before deletion

**Target Phase:** Phase 3 (Movegen Rewrite) - highest priority for performance

---

## Progress Tracking

**Completed Phases:** 0/10
**Current Phase:** Phase 0 (Not started)
**Estimated Time to Phase 3:** ~2-3 days
**Estimated Time to Complete:** ~2-3 weeks

---

## Notes

- เอกสารนี้เป็น working checklist ที่ใช้ track งานจริง
- ROADMAP.md มีรายละเอียดเพิ่มเติมและ context
- new doc/ มี detailed instructions สำหรับแต่ละ phase
- อัพเดทสถานะเมื่อทำแต่ละ task เสร็จ
- เก็บ before/after benchmarks ไว้เปรียบเทียบ
