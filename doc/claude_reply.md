# Claude Reply — Summary of Changes

## What was done

### 1. Max-capture rule enforced (ทั้ง TypeScript และ Python) ✅

- **`src/coreClaude/movegen.ts`**
- **`colab/makhos_engine.py`**

### 2. best.pt / latest.pt แยกกัน ✅

**`colab/train_az.py`**
- `best.pt` → อัปเดตเฉพาะเมื่อ win rate ≥ 55%
- `latest.pt` → บันทึก curr_net ทุก iter (resume)

### 3. Resume semantics ✅

**`colab/train_az.py`**
```python
curr_net.load(LATEST_NET_PATH)
best_net.load(BEST_NET_PATH if os.path.exists(BEST_NET_PATH) else LATEST_NET_PATH)
```

### 4. Draw rule ตรงกติกาจริง ✅

กติกาเดิม (`≤2 ตัว AND 20 ตา`) ไม่มีในกติกาจริง — แก้แล้วทั้งสองฝั่ง

| กฎ | รายละเอียด |
|---|---|
| ไม่มีการกิน 32 ตา | `halfmove_clock >= 32` → เสมอ |
| ฮอสล้วน + ไม่กิน 16 ตา | king ทั้งกระดาน + `halfmove_clock >= 16` → เสมอ |

- **`colab/makhos_engine.py`** — `is_draw_by_inactivity()`
- **`src/coreClaude/position.ts`** — `isDrawByInactivity()`

### 5. Training config ปรับแล้ว ✅

**`colab/train_az.py`**

| Parameter | เดิม | ใหม่ |
|---|---|---|
| `TEMP_CUTOFF` | 12 | 16 |
| `REPLAY_SIZE` | 100k | 150k |
| `N_EVAL_GAMES` | 20 | 60 |
| `N_MINIMAX_GAMES` | 20 | 40 |
| `N_EVAL_SIMS` | ไม่มี (N_SIMS//2) | 400 |

### 6. GPU info ใน log และ email ✅

**`colab/train_az.py`** — แสดง `nvidia-smi` output ทุก iter และใน checkpoint email

```
GPU: Tesla T4, 1842, 15360, 87, 72
     (name, VRAM used MB, total MB, GPU%, temp°C)
```

หมายเหตุ: ต้อง upload `train_az.py` ใหม่ขึ้น Drive แล้ว resume จึงจะเห็น GPU info

### 7. C_PUCT = 1.5 ✅

**`src/coreClaude/azMcts.ts`**

### 8. Board flip ✅

**`src/coreClaude/azFeatures.ts`** + **`colab/mcts_az.py`**

---

## Training progress (v3)

| Iter | vs best | vs MM3 (Python) | vs MM5 (Python) | vs MM5 (TS) |
|---|---|---|---|---|
| 9 | 100% ✅ | 100% | 50% | 0% |
| 19 | 100% ✅ | 100% | 25% | 25% (draw as P2) |
| 29 | 50% → keep | 50% | 50% | — |
| 36 | — | — | — | — |

หมายเหตุ: Python MM กับ TS MM ต่างกันมาก — Python ใช้ material อย่างเดียว, TS ใช้ full eval (PSQT + mobility + back-rank)

---

## What was NOT done

| เรื่อง | เหตุผล |
|---|---|
| Regression test สำหรับ max-capture | ผู้ใช้ไม่ได้ขอ |
| เอา Gmail password ออก | ผู้ใช้ไม่ได้ขอ |

---

## ไฟล์สำคัญ

### Python (Colab) — ต้อง upload ก่อน train

| File | Path | หน้าที่ |
|---|---|---|
| Engine + draw rule | `colab/makhos_engine.py` | กติกา + max-capture + draw |
| Network | `colab/network_az.py` | PyTorch model |
| MCTS | `colab/mcts_az.py` | MCTS + board flip |
| Training loop | `colab/train_az.py` | main loop + GPU info |

Drive folder: `MyDrive/makhos_az_v3/`

### TypeScript (App)

| File | Path | หน้าที่ |
|---|---|---|
| Draw rule | `src/coreClaude/position.ts` | isDrawByInactivity |
| Max-capture | `src/coreClaude/movegen.ts` | generateMoves |
| Board flip | `src/coreClaude/azFeatures.ts` | getFeatures |
| MCTS | `src/coreClaude/azMcts.ts` | C_PUCT=1.5 |
| Model | `src/coreClaude/azNet.ts` | ONNX loader |
