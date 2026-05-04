# Makhos AI Engine - Roadmap & Development Plan

## Overview
แผนการพัฒนาและปรับปรุง Thai Checkers AI Engine ให้มีความเร็ว ความถูกต้อง และความเสถียรสูงขึ้น

**เป้าหมายหลัก:**
- ลบโค้ดที่ไม่ใช้งาน (training/NN/MCTS)
- เพิ่มความเร็วของ move generation
- ปรับปรุง evaluation และ search quality
- เพิ่ม profiling และ testing infrastructure
- ไม่แก้ไข UI
- ไม่เพิ่ม heuristic patch โดยไม่มี benchmark proof

---

## Phase 0: Clean Training / NN / MCTS Code ✅

**Status:** READY TO START

**เป้าหมาย:** ลบโค้ดที่เกี่ยวกับ training, neural network, self-play ที่ไม่ใช้งาน

**ขอบเขตงาน:**
- ค้นหาไฟล์ที่ไม่ใช้งานด้วย keywords: train, selfplay, reinforcement, rl, model, nn, neural, weights, checkpoint, dataset, experience, replay, mcts, uct, policy, value, onnx
- ลบไฟล์ที่ไม่ใช้งาน:
  - `scripts/train*.ts`
  - `scripts/selfplay*.ts`
  - `scripts/generateDataset*.ts`
  - `scripts/replay*.ts`
  - `src/**/mcts*.ts`
  - `src/**/uct*.ts`
  - `src/**/nn*.ts`
  - `src/**/model*.ts`
  - `src/**/network*.ts`
  - `src/**/policy*.ts` (ยกเว้น levelPolicy.ts)
  - `src/**/value*.ts`
  - folders: `models/`, `weights/`, `checkpoints/`, `training-data/`
- ทำความสะอาด `package.json` (ลบ dependencies ที่ไม่ใช้: onnxruntime-react-native, onnxruntime-node)

**ไฟล์ที่ต้องเก็บไว้:**
- `src/coreClaude/movegen.ts`
- `src/coreClaude/eval.ts`
- `src/coreClaude/search/alphabeta.ts`
- `src/coreClaude/search/tt.ts`
- `src/coreClaude/search/zobrist.ts`
- `src/coreClaude/search/levelPolicy.ts`
- `scripts/aiBenchmark.ts`
- `scripts/tacticalSuite.ts`
- `scripts/ruleInvariantSuite.ts`

**Validation:**
```bash
npm run test:rules
npm run test:tactical
npm run bench:ai:fresh
```

**Deliverables:**
- รายงานไฟล์ที่ลบ
- รายงานไฟล์ที่เก็บไว้
- ผลการรัน test และ benchmark

---

## Phase 1: Add AI Hotspot Profiler

**Status:** PENDING

**เป้าหมาย:** สร้างเครื่องมือวัดประสิทธิภาพของ AI engine

**งานที่ต้องทำ:**
- สร้างไฟล์ `scripts/profileAiHotspots.ts`
- วัดค่าต่าง ๆ:
  - `elapsedMs` - เวลาที่ใช้
  - `nodes` - จำนวน node ที่ search
  - `qnodes` - จำนวน quiescence node
  - `nodes/sec` - ความเร็วในการ search
  - `generateMoves calls` - จำนวนครั้งที่เรียก generateMoves
  - `applyMove calls` - จำนวนครั้งที่เรียก applyMove
  - `evaluate calls` - จำนวนครั้งที่เรียก evaluate
  - `hashPosition calls` - จำนวนครั้งที่ hash
  - `TT hit/miss` - Transposition Table hit rate
  - `average branching factor` - ค่าเฉลี่ย branching
  - `approx Move object allocation count` - ประมาณการ allocation

**เพิ่ม npm script:**
```json
"profile:ai": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/profile-ai scripts/profileAiHotspots.ts && node .tmp/profile-ai/scripts/profileAiHotspots.js"
```

**หมายเหตุ:** ยังไม่ optimize แค่วัดค่าก่อน

---

## Phase 2: Add Perft Correctness Test

**Status:** PENDING

**เป้าหมาย:** สร้างเครื่องมือทดสอบความถูกต้องของ move generation

**งานที่ต้องทำ:**
- สร้างไฟล์ `scripts/perftThaiCheckers.ts`
- Implement:
  - `perft(pos, depth)` - นับจำนวน leaf nodes
  - `divide(pos, depth)` - แสดงรายละเอียดแต่ละ move

**ต้องครอบคลุม test cases:**
- Initial position
- Forced capture
- Max capture rule
- Men multi-capture
- King fly capture
- King multi-capture
- Promotion after capture
- Side -1 mirror cases

**เพิ่ม npm script:**
```json
"test:perft": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/perft scripts/perftThaiCheckers.ts && node .tmp/perft/scripts/perftThaiCheckers.js"
```

**เงื่อนไข:** ห้าม rewrite movegen ก่อนมี perft test

---

## Phase 3: Rewrite Movegen Allocation Safely 🎯

**Status:** PENDING (PRIORITY)

**เป้าหมาย:** Optimize move generation ให้เร็วขึ้น 2-3 เท่า โดยไม่เปลี่ยน behavior

**ปัญหาปัจจุบัน:**
1. `generateMoves()` สร้าง Move object ใหม่เยอะมาก
2. Multi-capture DFS ใช้ `[...caps]`, `[...path]` ทำ allocation หนัก
3. `ray()` เป็น generator function ทำ overhead
4. `nextInDir()` ใช้ `STEPS[from].find(...)` ซ้ำ ๆ
5. `hasCapturesAvailable()` กับ `generateMoves()` มี logic ซ้ำ
6. King movement scan ใช้ iterator/generator เยอะ
7. `applyMove()` ใช้ dynamic property access

**การออกแบบ:**

### 3.1 Keep Backward Compatibility
คงไว้:
```typescript
export interface Move {
  from: number;
  to: number;
  captured: number[];
  promote: boolean;
  path?: number[];
}
export function generateMoves(p: Position): Move[];
export function hasCapturesAvailable(p: Position): boolean;
export function applyMove(p: Position, m: Move): Position;
```

### 3.2 Add Faster APIs
เพิ่ม:
```typescript
export function generateMovesInto(p: Position, out: Move[]): Move[];
export function generateCapturesInto(p: Position, out: Move[]): Move[];
```

### 3.3 Precompute Direction Tables (bitboards.ts)
```typescript
export const DIRS = ['UL', 'UR', 'DL', 'DR'] as const;
export type Dir = typeof DIRS[number];

// NEXT[sq][dirIndex] = next square or -1
export const NEXT: Int8Array[];

// RAYS[sq][dirIndex] = list of squares outward
export const RAYS: number[][][];
```

### 3.4 Optimization Techniques
- ใช้ `EMPTY_CAPTURED = Object.freeze([])` สำหรับ quiet moves
- ใช้ buffer arrays (`capsBuf`, `pathBuf`) แทน array copies ใน DFS
- แทน generator `ray()` ด้วย precomputed `RAYS`
- แทน `STEPS[from].find()` ด้วย `NEXT[from][dir]`
- Rewrite `applyMove()` ใช้ explicit branch แทน dynamic property access

**ไฟล์ที่อนุญาตให้แก้:**
- `src/coreClaude/movegen.ts` (ต้องแก้)
- `src/coreClaude/bitboards.ts` (ต้องแก้)
- `scripts/ruleInvariantSuite.ts` (ถ้าจำเป็น)
- `scripts/tacticalSuite.ts` (ถ้าจำเป็น)
- `scripts/perftThaiCheckers.ts` (ถ้ามี)

**ห้ามแก้:**
- UI code
- Game rules
- `src/coreClaude/eval.ts`
- `src/coreClaude/search/alphabeta.ts` (ยกเว้นการเปลี่ยนจาก generateMoves เป็น generateMovesInto)

**กฎเกมที่ต้องรักษา:**
- Men เดินหน้าเท่านั้น
- Men capture หน้าเท่านั้น
- King/Hos fly ได้
- King capture ต้อง land ช่องว่างหลังตัวที่ถูกกินทันที
- Forced capture
- Max capture rule (เลือก move ที่ captured.length มากสุด)
- Multi-capture chain
- Promotion เมื่อ men ถึง last rank

**Validation:**
```bash
npm run test:rules
npm run test:tactical
npm run test:perft
npm run profile:ai
npm run bench:ai:fresh
```

**Before/After Report ต้องมี:**
- Files changed
- What changed
- Correctness (test results)
- Performance (elapsedMs, nodes/sec, tactical solve rate)
- Speedup estimate

**Acceptance Criteria:**
1. ✅ `npm run test:rules` ผ่าน
2. ✅ `npm run test:tactical` ผ่าน
3. ✅ `npm run bench:ai:fresh` ไม่แย่ลง
4. ✅ `generateMoves(pos)` ให้ผลเหมือนเดิม
5. ✅ `generateMovesInto(pos, out)` ใช้ซ้ำ out ได้
6. ✅ `hasCapturesAvailable` ไม่ allocate Move
7. ✅ `applyMove` ไม่ใช้ dynamic property access

---

## Phase 4: TT / Zobrist Collision Safety

**Status:** PENDING

**เป้าหมาย:** ป้องกัน hash collision ใน Transposition Table

**แนวทาง:**
- เพิ่ม verification key (32-bit ที่สอง)
- `hashPosition(pos)` → ใช้เป็น index key
- `verifyHashPosition(pos)` → ใช้เป็น verify key
- TT เก็บทั้ง `key` และ `verifyKey`
- TT hit ต้อง match ทั้งสองค่า

**ไฟล์ที่ต้องแก้:**
- `src/coreClaude/search/zobrist.ts`
- `src/coreClaude/search/tt.ts`
- `src/coreClaude/search/alphabeta.ts`

**Tests ที่ต้องเพิ่ม:**
- Same position → same hash
- Side to move changes → different hash
- Different piece placement → different hash
- Random collision smoke test

---

## Phase 5: Split Root Safety Heuristics

**Status:** PENDING

**เป้าหมาย:** แยก root override logic ออกมาเป็น module แยก

**งานที่ต้องทำ:**
- สร้างไฟล์ `src/coreClaude/search/rootSafety.ts`
- Move root override logic จาก `alphabeta.ts`
- เพิ่ม feature flags:
```typescript
interface SearchOptions {
  enableRootSafety?: boolean;
  enableTrapOverride?: boolean;
  enableDiversification?: boolean;
}
```

**Default settings:**
- hard/expert: diversification off, trap override conservative
- เก็บ root safety เฉพาะที่ benchmark พิสูจน์ว่าดีขึ้น

**Benchmark variants:**
- baseline
- noRootSafety
- noTrapOverride
- noDiversification

**เงื่อนไข:** ถ้า head-to-head แย่ลง ให้ลบ override นั้นออก

---

## Phase 6: Evaluation Improvement Without Expensive Movegen

**Status:** PENDING

**เป้าหมาย:** ปรับปรุง evaluation ด้วย cheap features เท่านั้น

**Features ที่อนุญาต:**
- ✅ Breakthrough / promotion race
- ✅ Trapped king detection
- ✅ Edge king penalty
- ✅ King ray mobility (precomputed rays)
- ✅ Tempo / opposition approximation
- ✅ Cheap hanging piece approximation
- ✅ Drawish endgame detection

**ห้ามทำ:**
- ❌ eval ห้ามเรียก `generateMoves`
- ❌ eval ห้าม allocate large arrays
- ❌ eval ห้ามเรียก deep search

**เงื่อนไข:** ทุก feature ต้องมี named weight constants

---

## Phase 7: Small Endgame Solver

**Status:** PENDING

**เป้าหมาย:** Implement endgame tablebase แบบเล็ก ๆ

**ขอบเขต:**
- <= 4 pieces exact/lazy cache
- Root probe only
- ไม่ probe ใน negamax nodes ธรรมดา

**Test positions:**
- 2 kings vs 1 king
- King + man vs king
- King vs 2 men
- Men race
- All kings drawish cases

---

## Phase 8: Arena Gate

**Status:** PENDING

**เป้าหมาย:** สร้างระบบทดสอบเปรียบเทียบ engine versions

**งานที่ต้องทำ:**
- สร้างไฟล์ `scripts/arenaGate.ts`
- เปรียบเทียบ baseline engine vs candidate engine

**Options:**
```bash
--games 100
--max-plies 240
--seed fixed
```

**Pass criteria:**
- Candidate score >= 52%
- Tactical blunder rate ไม่แย่ลง
- Avg move time ไม่แย่เกิน 15%

**npm script:**
```json
"gate:ai": "tsc --target ES2022 --module CommonJS --moduleResolution node --strict --esModuleInterop --skipLibCheck --outDir .tmp/arena scripts/arenaGate.ts && node .tmp/arena/scripts/arenaGate.js"
```

---

## Phase 9: Benchmark Summary Markdown

**Status:** PENDING

**เป้าหมาย:** สร้าง markdown report สำหรับ benchmark results

**Output files:**
- `.tmp/benchmarks/ai-benchmark-latest.json`
- `.tmp/benchmarks/ai-benchmark-summary.md`

**Summary ต้องมี:**
- Solve rate
- Blunder rate
- P95 ms
- Nodes/sec
- Top failed cases
- Before/after comparison

---

## Phase 10: Stop and Review

**Status:** PENDING

**เป้าหมาย:** Review และทำเอกสารสรุป

**หลัง Phase 10:**
- ❌ ห้ามเพิ่ม training กลับมาก่อน
- ❌ ห้ามเพิ่ม random tactical patches

**Future work (เมื่อ engine stable):**
- Opening book
- Larger endgame tablebase
- NN eval
- MCTS / PUCT
- Self-play training

---

## Important Rules

1. **ทำทีละ Phase** - ห้ามข้าม phase
2. **ก่อนแก้** - inspect files และสรุปการเปลี่ยนแปลง
3. **หลังแก้** - รายงาน:
   - Files changed
   - What was removed/kept
   - Benchmark/test results
   - Risks
4. **ถ้า test fail** - หยุดและอธิบาย
5. **ห้าม rewrite ทั้ง engine** - ทำแบบค่อย ๆ ทีละส่วน

---

## Only Accept Changes That Improve

✅ **Correctness** - ความถูกต้องตามกฎ
✅ **Speed** - ความเร็วในการ search
✅ **Benchmark** - ผลการทดสอบดีขึ้น
✅ **Arena result** - ชนะ/เสมอมากขึ้น
✅ **Stability** - ไม่มี crash / bug

❌ **Random patches** - ห้ามเพิ่ม heuristic โดยไม่มี proof

---

## Current Status Summary

**Branch:** `tuneGPT`
**Latest commit:** AI tactical improvements
**Priority Phase:** Phase 3 (Movegen Rewrite)

**Recent improvements (tuneAI branch):**
- ✅ Added hanging pieces penalty detection
- ✅ Enhanced quiescence search with recapture prioritization
- ✅ Rebalanced difficulty levels for monotonic progression
- ✅ Fixed delta pruning promotion blind spots

**Next immediate task:** Phase 0 (Clean unused code)
