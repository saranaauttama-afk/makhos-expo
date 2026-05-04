แล้วพรุ่งนี้สั่ง Codex แบบนี้:

```text
อ่าน CODEX_REWRITE_MOVEGEN_FAST.md แล้วทำเฉพาะ movegen rewrite ตามไฟล์นี้
ก่อนแก้ให้สรุปแผนและไฟล์ที่จะเปลี่ยน
หลังแก้ให้รัน test และสรุป before/after

# Codex Task — Rewrite Thai Checkers Movegen for Speed

## Goal

Rewrite `src/coreClaude/movegen.ts` ให้เร็วขึ้นประมาณ 2-3 เท่า โดยยังต้องรักษา behavior เดิม 100%

ห้ามแก้ UI  
ห้ามแก้กฎเกม  
ห้ามแก้ search logic ก่อน  
โฟกัสเฉพาะ move generation / applyMove / helper precompute ที่เกี่ยวข้อง

---

## Current Problem

ไฟล์ปัจจุบัน:

```text
src/coreClaude/movegen.ts

มีจุดช้า:

1. generateMoves() สร้าง Move object ใหม่เยอะมาก
2. multi-capture DFS ใช้ [...caps], [...path] ทำ allocation หนัก
3. ray() เป็น generator function ทำ overhead ใน hot path
4. nextInDir() ใช้ STEPS[from].find(...) ซ้ำ ๆ
5. hasCapturesAvailable() กับ generateMoves() มี logic ซ้ำ
6. king movement scan ใช้ iterator/generator เยอะ
7. applyMove() ใช้ dynamic property access ผ่าน (q as any)[key]

เป้าหมายคือทำให้ search เรียก movegen ได้เร็วขึ้นและลด GC pressure

Files allowed to edit
src/coreClaude/movegen.ts
src/coreClaude/bitboards.ts
scripts/ruleInvariantSuite.ts
scripts/tacticalSuite.ts
scripts/perftThaiCheckers.ts ถ้ามี
package.json ถ้าต้องเพิ่ม script

ถ้าจำเป็นจริง ๆ ค่อยแตะ:

src/coreClaude/search/alphabeta.ts

แต่ phase แรกอย่าเพิ่งแก้ search

Required Design
1. Keep Backward Compatibility

ต้องคง API เดิม:

export interface Move {
  from: number;
  to: number;
  captured: number[];
  promote: boolean;
  path?: number[];
}

export function applyMove(p: Position, m: Move): Position;
export function hasCapturesAvailable(p: Position): boolean;
export function generateMoves(p: Position): Move[];

UI และ benchmark เดิมต้องไม่พัง

2. Add Faster APIs

เพิ่ม API ใหม่:

export function generateMovesInto(p: Position, out: Move[]): Move[];
export function generateCapturesInto(p: Position, out: Move[]): Move[];

เงื่อนไข:

- generateMoves(p) เป็น wrapper ที่เรียก generateMovesInto(p, [])
- search ในอนาคตจะใช้ generateMovesInto เพื่อ reuse array
- function ใหม่ต้องไม่เปลี่ยนผลลัพธ์
3. Remove generator-based ray()

ห้ามใช้:

function *ray(...)

ให้แทนด้วย precomputed ray table ใน bitboards.ts

เพิ่มใน bitboards.ts:

export const DIRS = ['UL', 'UR', 'DL', 'DR'] as const;
export type Dir = typeof DIRS[number];

export const NEXT: Int8Array[]; 
// NEXT[sq][dirIndex] = next square or -1

export const RAYS: Int8Array[][];
// RAYS[sq][dirIndex] = list of squares outward in that direction

หรือโครงสร้างใกล้เคียงได้ แต่ต้องเร็วกว่า STEPS[from].find

4. Replace nextInDir()

ห้ามใช้:

STEPS[from].find(...)

ให้ใช้:

NEXT[from][dirIndex]
5. Reduce capture DFS allocation

ตอนนี้มี:

out.push({ from, to: lastTo, captured: [...caps], promote, path: [...path] });

ให้ลด allocation ให้มากที่สุด

แนวทางที่ยอมรับได้:

- ใช้ reusable fixed arrays เช่น capsBuf[16], pathBuf[16]
- ตอน push result ค่อย slice เฉพาะ final move
- ห้ามสร้าง array ใหม่ทุก recursion
- recursion ต้องส่ง depth แทน push/pop ถ้าเป็นไปได้

ตัวอย่างแนวทาง:

const capsBuf = new Int8Array(16);
const pathBuf = new Int8Array(16);

function dfs(cur, depth, ...) {
  ...
  capsBuf[depth] = capturedSq;
  pathBuf[depth] = landingSq;
  dfs(landing, depth + 1, ...);
  ...
}

function pushCaptureMove(...) {
  const captured = Array.from(capsBuf.subarray(0, depth));
  const path = Array.from(pathBuf.subarray(0, depth));
  out.push({ from, to, captured, promote, path });
}

ถ้าทำได้ดีกว่านี้ เช่น MovePool ให้ทำได้ แต่ห้ามพัง API

6. Optimize applyMove()

ตอนนี้ใช้ dynamic key:

const myMen = p.side === 1 ? 'p1Men' : 'p2Men';
(q as any)[myMen]

ให้ rewrite เป็น branch ชัดเจน:

if (p.side === 1) {
  // update p1 / p2 directly
} else {
  // update p2 / p1 directly
}

เป้าหมาย:

- ลด dynamic property access
- ลด object mutation complexity
- คืน Position object ใหม่เหมือนเดิม

ต้องรักษา behavior:

- capture reset halfmoveClock = 0
- quiet move +1
- promote men เป็น king
- side toggle
7. Optimize hasCapturesAvailable()

ใช้ NEXT / RAYS แทน generator/find

ต้อง return ทันทีเมื่อเจอ capture แรก

ห้ามสร้าง Move object

8. Optimize quiet move generation

Men:

- ใช้ NEXT[sq][UL/UR] สำหรับ P1
- ใช้ NEXT[sq][DL/DR] สำหรับ P2

Kings:

- loop ผ่าน RAYS[sq][dir]
- stop เมื่อเจอ occupied
9. Preserve Thai Checkers Rules

ต้องรักษากฎนี้:

- Men เดินหน้าเท่านั้น
- Men capture หน้าเท่านั้น
- King/Hos fly ได้
- King capture ต้อง land ช่องว่างหลังตัวที่ถูกกินทันที ตาม behavior เดิม
- Forced capture
- ถ้ามีหลาย capture ต้องเลือกเฉพาะ move ที่ captured.length มากสุด
- Multi-capture chain
- Promotion เมื่อ men ถึง last rank

สำคัญ:
ถ้ากฎปัจจุบันใน code ไม่ตรงกติกาจริง ห้ามแก้กฎใน task นี้
task นี้คือ optimize behavior เดิมก่อน

Validation

หลัง rewrite ต้องรัน:

npm run test:rules
npm run test:tactical
npm run bench:ai:fresh

ถ้ามี perft:

npm run test:perft
Required Before/After Report

Codex ต้องสรุป:

Files changed:
- ...

What changed:
- removed generator ray
- added NEXT/RAYS
- added generateMovesInto
- optimized applyMove
- reduced DFS allocation

Correctness:
- test:rules result
- test:tactical result
- test:perft result if available

Performance:
- bench:ai before
- bench:ai after
- estimated speedup
- any tactical regression
Acceptance Criteria

ผ่านได้เมื่อ:

1. npm run test:rules ผ่าน
2. npm run test:tactical ผ่าน
3. npm run bench:ai:fresh ไม่แย่ลง
4. generateMoves(pos) ให้ผลเหมือนเดิม
5. generateMovesInto(pos, out) ใช้ซ้ำ out ได้
6. hasCapturesAvailable ไม่ allocate Move
7. applyMove ไม่ใช้ dynamic property access
Important Warning

ห้ามเพิ่ม heuristic AI
ห้ามแก้ eval
ห้ามแก้ alphabeta แบบสุ่ม
ห้ามแก้ UI
ห้ามเปลี่ยนกฎเกม

นี่คือ task optimization เท่านั้น