# Bible — AlphaZero Makhos (ภาษาไทย)

> ไฟล์นี้อธิบายแนวคิดทั้งหมดที่ใช้ในการเทรน AI หมากฮอส
> ถ้ามีคำถามเกี่ยวกับ concept ใดๆ ให้อ่านที่นี่ก่อน

---

## 1. ภาพรวม AlphaZero

AlphaZero คือวิธีสร้าง AI เกมโดยไม่ต้องมีความรู้จากมนุษย์ (ยกเว้นกติกา)
เรียนรู้จากการ **เล่นกับตัวเอง** ล้วนๆ

```
วนซ้ำทุก iteration:
  1. Self-play    → AI เล่นกับตัวเอง สร้าง training data
  2. Train        → เอา data ไปสอน neural network
  3. Evaluate     → ทดสอบว่า network ใหม่ดีขึ้นไหม
  4. ถ้าดีขึ้น   → อัปเดต best network
```

---

## 2. Neural Network

Network ของเราประกอบด้วย 2 หัว:

```
Input: position (128 ตัวเลข)
         ↓
    FC + 4×ResBlock (256 hidden)
         ↓
   ┌─────┴──────┐
Policy Head   Value Head
(1024 logits) (1 ตัวเลข)
   ↓               ↓
"ควรเดินตาไหน"  "ใครได้เปรียบ"
```

**Policy Head**: บอกความน่าจะเป็นของทุก move ที่เป็นไปได้ (1024 slots = 32×32)
**Value Head**: บอกว่าตำแหน่งนี้ฝ่ายที่กำลังเดินได้เปรียบแค่ไหน (+1=ชนะแน่, -1=แพ้แน่, 0=เสมอ)

### Input Features (128 ตัวเลข)
- 32 ตัวแรก: ตำแหน่ง "ตัวฉัน" (เบี้ย)
- 32 ถัดมา: ตำแหน่ง "ตัวฉัน" (king)
- 32 ถัดมา: ตำแหน่ง "ตัวฝ่ายตรงข้าม" (เบี้ย)
- 32 ถัดมา: ตำแหน่ง "ตัวฝ่ายตรงข้าม" (king)

**Board Flip**: ฝ่าย P2 จะถูก flip board (`sq → 31-sq`) ก่อนส่งเข้า network
→ ทำให้ network เห็นว่า "ตัวฉันอยู่ด้านล่างเสมอ" ไม่ว่าจะเล่นฝ่ายไหน

---

## 3. MCTS (Monte Carlo Tree Search)

ใช้ตอน **เลือก move** ทั้งตอน self-play และตอน inference ใน app

### วิธีทำงาน (ทำซ้ำ N_SIMS ครั้ง):

```
1. SELECT   — เดินลงมาใน tree ตาม PUCT score
             (balance ระหว่าง exploitation กับ exploration)

2. EXPAND   — ถึง node ใหม่ → ส่ง position ให้ network ประเมิน
             → ได้ policy (prior) + value กลับมา

3. BACKPROP — อัปเดตค่า W (win) และ N (visit count)
             กลับขึ้นไปถึง root สลับเครื่องหมายทุกระดับ
             (เพราะคู่แข่งเป็นฝ่ายตรงข้าม)
```

### PUCT Score (สูตรเลือก move):
```
score = -Q + C_PUCT × P × √(parent.N) / (1 + N)
         ↑             ↑
    exploitation    exploration
    (Q สูง=ดี)     (P=prior จาก network, N น้อย=ยังไม่ถูก explore)
```

- **C_PUCT = 1.5** (ทั้ง Python และ TypeScript ต้องตรงกัน)
- Q สูง = move นี้ผ่านการทดสอบแล้วว่าดี
- P สูง + N น้อย = network คิดว่าดี แต่ยังไม่ได้ลอง → ควร explore

### หลังครบ N_SIMS:
เลือก move ที่มี **visit count (N) สูงสุด** ไม่ใช่ Q สูงสุด

### N_SIMS กับความแข็ง:
| N_SIMS | vs MM5 | vs MM7 | เวลา/move |
|---|---|---|---|
| 200 | 50% | — | ~0.03s |
| 400 | 75% | — | ~0.10s |
| 800 | 100% | 25% | ~0.10s |

---

## 4. Self-Play

การสร้าง training data โดย AI เล่นกับตัวเอง

```python
for game in range(N_SELFPLAY):  # 100 games per iter
    pos = initial_position()
    while not terminal:
        moves, probs = mcts(pos, network, N_SIMS=200)
        # เก็บ (features, policy, side) ไว้
        move = sample(moves, probs)  # สุ่มตาม probability (ช่วง opening)
        pos = apply_move(pos, move)
    # กำหนด result (+1/-1/0) ให้ทุก position ในเกมนี้
```

**Temperature**: ช่วง `ply < TEMP_CUTOFF (16)` → สุ่ม move ตาม probability (exploration)
หลังจากนั้น → เลือก move ที่ดีที่สุด (exploitation)

**Dirichlet Noise**: เพิ่ม noise ที่ root ระหว่าง self-play เพื่อให้เล่นหลากหลายขึ้น

---

## 5. Replay Buffer

คลังเก็บ training samples จาก self-play ที่ผ่านมา

- ขนาด: **150,000 samples** (rolling window)
- แต่ละ sample: `(features[128], policy[1024], value)`
- ทุก iter เพิ่ม ~7,000 samples ใหม่ และลบเก่าออก
- ใช้สุ่ม batch จาก buffer มา train

ทำไมไม่ใช้แค่ data ล่าสุด?
→ data เก่าช่วย **stabilize training** ป้องกันการลืม pattern ที่เรียนไปแล้ว

---

## 6. Training

```python
for step in range(TRAIN_STEPS):  # 500 steps per iter
    batch = random.sample(replay_buffer, BATCH_SIZE=256)

    pred_policy, pred_value = network(features)

    p_loss = cross_entropy(pred_policy, target_policy)  # policy loss
    v_loss = mse(pred_value, target_value)              # value loss

    loss = p_loss + v_loss
    loss.backward()
    optimizer.step()
```

### p_loss (Policy Loss):
วัดว่า network เดา distribution ของ moves ได้ถูกแค่ไหน
- สูง = ยังไม่รู้ว่า move ไหนดี
- target = visit counts จาก MCTS

### v_loss (Value Loss):
วัดว่า network ประเมิน "ใครชนะ" ได้แม่นแค่ไหน
- สูง = ยังไม่รู้ว่าตำแหน่งนี้ใครได้เปรียบ
- target = ผลจริงของเกม (+1/-1/0)

### ค่าที่เคยเห็น:
| iter | p_loss | v_loss |
|---|---|---|
| 0 | 6.18 | 0.61 |
| 9 | 1.78 | 0.36 |
| 39 | 1.42 | 0.27 |

---

## 7. Checkpoint System

### best.pt vs latest.pt:
- **latest.pt**: บันทึก `curr_net` **ทุก iter** → ใช้ resume
- **best.pt**: บันทึกเฉพาะเมื่อ `curr_net` ชนะ `best_net` ≥ **55%** ใน eval

### Evaluation (ทุก 10 iter):
1. `curr_net` vs `best_net` (60 games, N_EVAL_SIMS=400)
2. `curr_net` vs random (40 games)
3. `curr_net` vs minimax-3 (40 games, Python MM)
4. `curr_net` vs minimax-5 (40 games, Python MM)

---

## 8. Draw Rules (กติกาจริง)

| กฎ | เงื่อนไข |
|---|---|
| ไม่มีการกิน 32 ตา | `halfmove_clock >= 32` → เสมอ |
| ฮอสล้วน + ไม่กิน 16 ตา | king ทั้งกระดาน + `halfmove_clock >= 16` → เสมอ |
| Stalemate | เดินไม่ได้ → **แพ้** (ไม่ใช่เสมอ) |

`halfmove_clock` reset เป็น 0 ทุกครั้งที่มีการจับหมาก

---

## 9. Max-Capture Rule

กติกาหมากฮอส: ถ้าจับได้หลาย chain ต้องเลือก chain ที่ **ยาวที่สุด** เสมอ

```typescript
const maxCaps = Math.max(...captures.map(m => m.captured.length));
return captures.filter(m => m.captured.length === maxCaps);
```

ผลต่อ training:
- เปลี่ยน game tree ทั้งหมด → ต้อง retrain ใหม่ตั้งแต่ต้น (เหตุผลที่มี v3)
- เกมจบเร็วขึ้น (จับได้เยอะต่อตา) → draw น้อยลง → training signal ชัดขึ้น

---

## 10. P2 Structural Advantage

ในหมากฮอสไทย **P2 ได้เปรียบเชิงโครงสร้าง** เสมอ
→ MM vs MM ทุก depth: P2 ชนะ 100%

ผลต่อการ test:
- `A M A M` pattern = AZ ชนะแค่ตอนเป็น P2
- `A D A D` pattern = AZ ชนะ P2, draw P1 (ดีขึ้น)
- `A A A A` pattern = AZ ชนะทั้ง P1 และ P2 (แข็งจริง)

---

## 11. Python MM vs TypeScript MM

Python minimax (ใน train_az.py) และ TypeScript minimax (ใน app) **ไม่ใช่ตัวเดียวกัน**:

| | Python MM | TypeScript MM |
|---|---|---|
| Eval | Material อย่างเดียว | Full: PSQT + mobility + back-rank |
| ความแข็ง | อ่อนกว่ามาก | แข็งกว่ามาก |
| ใช้เพื่อ | Monitor training trend | Benchmark จริง |

Python MM5 ≈ TypeScript MM1-2 โดยประมาณ

---

## 12. Config สำคัญ

```python
# Self-play
N_SELFPLAY   = 100    # games per iteration
N_SIMS       = 200    # MCTS sims (self-play)
TEMP_CUTOFF  = 16     # ply ก่อน switch to argmax
MAX_GAME_LEN = 250    # hard cap

# Training
REPLAY_SIZE  = 150_000
BATCH_SIZE   = 256
TRAIN_STEPS  = 500
LR           = 1e-3

# Evaluation
N_EVAL_GAMES    = 60
N_MINIMAX_GAMES = 40
N_EVAL_SIMS     = 400   # MCTS sims (eval — แยกจาก self-play)
WIN_THRESHOLD   = 0.55
EVAL_INTERVAL   = 10
```

---

## 13. Benchmark Results (v3)

| Iter | N_SIMS | vs MM5 TS | vs MM7 TS |
|---|---|---|---|
| 0009 | 200 | 0% | — |
| 0019 | 200 | 25% | — |
| 0039 | 200 | 50% | — |
| 0039 | 400 | 75% | — |
| 0039 | 800 | **100%** | 25% |

เป้าหมายต่อไป: iter 49-59 + 800 sims vs MM7, MM9
