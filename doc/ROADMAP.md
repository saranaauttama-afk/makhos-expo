# Makhos v2 — Roadmap

## ✅ Done

### Bug Fixes
- [x] **UI freeze** — `think()` เดิม block main thread; แก้โดยทำ `iterativeDeepening` เป็น async + yield ระหว่าง depth iterations
- [x] **AI เดิน 2 ตาเอง** — cleanup function ใน `useEffect` reset `pendingRef` ระหว่าง React Strict Mode double-invoke; แก้โดยลบ cleanup return ออก
- [x] **`thinking` ค้างที่ `true` ตลอด** — `setThinking(false)` ถูก guard ด้วย `token.cancelled`; แก้ให้ call เสมอ
- [x] **Board click ไม่ได้บน Android** — SVG `onPress` ไม่ reliable; แก้โดยใช้ `Pressable` 8×8 grid overlay แทน
- [x] **SafeAreaView deprecated warning** — เปลี่ยน import จาก `react-native` → `react-native-safe-area-context`
- [x] **Expo Go โหลดไม่ได้** — `react-native` version mismatch; แก้ด้วย `npx expo install --fix` + `--tunnel`

### AI / Engine
- [x] **Codex v3 eval** — เขียน eval ใหม่ทั้งหมด: PSQT tables (advancement + column safety + king centrality), king endgame proximity bonus, ลบ `captureNetScore` ที่เรียก `generateMoves` 2 ครั้งต่อ leaf node
- [x] **Async iterative deepening + CancelToken** — ยกเลิก search ได้กลางคัน, ไม่ block UI
- [x] **Forced capture shortcut** — ถ้ามี move เดียว → เดินทันที ไม่ต้อง search
- [x] **Reverse Futility Pruning (RFP)** — ตัด node ที่ eval สูงกว่า beta มากจนไม่คุ้มจะ search
- [x] **Razoring** — drop ลง quiescence เมื่อ eval ต่ำกว่า alpha มากที่ shallow depth
- [x] **Probcut** — ลอง top-3 captures ที่ reduced depth ก่อน full search (depth ≥ 5)
- [x] **Null Move Pruning (NMP)** — ข้ามตา ถ้า opponent ยัง ≥ beta → prune; only quiet positions, depth ≥ 3
- [x] **Late Move Pruning (LMP)** — ตัด quiet moves ที่ depth 1–2 หลัง search ไปพอแล้ว
- [x] **LMR tactic fix** — ปิด Late Move Reduction เมื่อ opponent จะมี forced captures (ป้องกันพลาด sacrifice tactics)
- [x] **`hasCapturesAvailable()` fast check** — เพิ่ม O(pieces) function แทน `generateMoves(child)` ใน loop → ลด double movegen call ต่อ node ~30–40%
- [x] **PSQT rowBonus tuning** — เพิ่ม front-rank bonus: `[42,34,24,15,8,4,1,0]` (เดิม `[30,25,18,12,7,3,1,0]`)
- [x] **backRankGuard ลดลง** — จาก 10cp → 5cp ต่อตัว (ลด passive play)
- [x] **NMP null position fix** — ลบ pushRepetition/popRepetition สำหรับ synthetic null position

---

## 🚧 To Do

### Strength (เรียงตาม impact)

- [x] **Countermove heuristic**
  จำ move ที่ทำให้เกิด beta cutoff เป็น response ต่อ move ก่อนหน้า → move ordering ดีขึ้น ~10% nodes
  `File: src/coreCodex/search/alphabeta.ts`

- [x] **Protected men bonus**
  ให้ bonus หมากที่มีหมากฝ่ายเดียวกันคุ้มกัน (friendly piece อยู่ทิศ "behind") → AI เล่นหมากหมู่ดีขึ้น
  `File: src/coreCodex/eval.ts`

- [x] **Internal Iterative Deepening (IID)**
  เมื่อไม่มี TT hit ที่ deep node (depth ≥ 4) → ทำ shallow search depth-2 ก่อนเพื่อหา best move → move ordering ดีขึ้นมากใน deep nodes
  `File: src/coreCodex/search/alphabeta.ts`

- [ ] **Opening book** *(ยาก)*
  บันทึก opening ที่ดีของหมากไทย 10–15 ตาแรก → ไม่เสียเวลา search ช่วง opening เลย
  `File: src/coreCodex/search/openingBook.ts` *(new)*

### Speed / UX

- [x] **Adaptive time management**
  หยุด search เร็วขึ้นเมื่อ best move เดิม 3 depth ติดต่อกัน + ใช้เวลา ≥50% → ประหยัดเวลาในตำแหน่งที่ชัดเจน
  `File: src/coreCodex/search/alphabeta.ts`

- [x] **Web Worker**
  Engine ย้ายไปรันใน Worker thread แยก → UI ไม่กระตุกเลยระหว่าง AI คิด; มี fallback กลับ async main-thread สำหรับ Expo Go
  `File: src/workers/engineWorker.ts` *(new)*, `src/ui/useCodexEngine.ts`, `metro.config.js` *(new)*

---

## 📝 Notes

- Engine: Negamax + Alpha-Beta + Iterative Deepening + TT
- Platform: React Native / Expo SDK 54 / Hermes JS (Android) — ~10–20x ช้ากว่า V8
- AI think time: `AI_THINK_MS = Platform.OS === 'web' ? 600 : 1200`
- Board: 8×8, 32 dark squares, bitboard representation (uint32)
- Thai checkers rules: men เดินหน้าเท่านั้น, kings บินได้ทุกทิศ, บังคับกินเสมอ
