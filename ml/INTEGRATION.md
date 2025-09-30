# 🤖 ML Model Integration Guide

วิธีใช้ ML model ที่ train เสร็จแล้วในโปรแกรม

---

## 📋 Step 1: Export ONNX (ใน Google Colab)

รันใน Colab notebook:

```python
# Install dependencies
!pip install onnx onnxruntime -q

# Export model
%cd /content/makhos-expo/ml
!python export_onnx.py \
  --model /content/drive/MyDrive/makhos_ml/checkpoints/best_model.pt \
  --output /content/drive/MyDrive/makhos_ml/best_model.onnx \
  --model_type resnet \
  --num_channels 128 \
  --num_res_blocks 6
```

**Output:** `best_model.onnx` ใน Google Drive

---

## 📋 Step 2: Download และวางไฟล์

```python
# Download from Colab
from google.colab import files
files.download('/content/drive/MyDrive/makhos_ml/best_model.onnx')
```

วางไฟล์ที่:
```
D:\MyApp\makhos-expo\ml\best_model.onnx
```

---

## 📋 Step 3: Install Dependencies

```bash
npm install onnxruntime-node
```

---

## 📋 Step 4: Test ML vs AI

รัน tournament 10 เกม:

```bash
npx tsx scripts/testMLvsAI.ts 10 1000
```

**Parameters:**
- `10` = จำนวนเกม
- `1000` = AI time per move (ms)

**Output:**
```
ML vs AI Tournament
Games: 10
ML wins: 6 (60%)
AI wins: 3 (30%)
Draws: 1 (10%)
```

---

## 📋 Step 5: ใช้ใน App

### A) สร้าง ML Player

```typescript
import { MLPlayer } from './src/core/ml/mlPlayer';
import * as path from 'path';

const mlPlayer = new MLPlayer({
  modelPath: path.join(__dirname, 'ml', 'best_model.onnx'),
  temperature: 0.1,  // 0 = greedy, higher = more random
  topK: 10           // Consider top 10 moves
});

await mlPlayer.load();
```

### B) เลือก Move

```typescript
import { Position, initialPosition } from './src/core/position';

const pos = initialPosition();
const move = await mlPlayer.selectMove(pos);

if (move) {
  console.log(`ML plays: ${move.from} → ${move.to}`);
}
```

### C) Evaluate Position

```typescript
const value = await mlPlayer.evaluate(pos);
console.log(`Position evaluation: ${value}`);
// -1 = P2 winning, 0 = equal, +1 = P1 winning
```

---

## 🎮 ตัวอย่าง: AI vs AI Self-Play

```typescript
import { applyMove } from './src/core/movegen';
import { iterativeDeepening } from './src/core/search/alphabeta';
import { TT } from './src/core/search/tt';

async function mlVsAI(numGames: number) {
  const mlPlayer = new MLPlayer({ modelPath: './ml/best_model.onnx' });
  await mlPlayer.load();

  for (let i = 0; i < numGames; i++) {
    let pos = initialPosition();
    const tt = new TT();

    while (true) {
      let move;

      if (pos.side === 1) {
        // ML plays P1
        move = await mlPlayer.selectMove(pos);
      } else {
        // Traditional AI plays P2
        const result = iterativeDeepening(pos, 1000, tt);
        move = result.best;
      }

      if (!move) break;
      pos = applyMove(pos, move);
    }
  }
}

mlVsAI(10);
```

---

## 📊 Model Performance Tips

### ถ้า ML แพ้ AI:
1. **เพิ่ม data** - Gen 5000+ เกมแล้ว train ใหม่
2. **ปรับ temperature** - ลดเหลือ 0 (greedy)
3. **เพิ่ม topK** - พิจารณา moves มากขึ้น
4. **Check overfitting** - ใช้ checkpoint epoch ต้นๆ

### ถ้า ML ชนะ AI:
1. ✓ Model ใช้งานได้!
2. เพิ่ม data เพื่อ improve เพิ่ม
3. ลอง train ResNet ใหญ่ขึ้น

---

## 🔧 Troubleshooting

### Error: "Cannot find module 'onnxruntime-node'"
```bash
npm install onnxruntime-node
```

### Error: "Model not loaded"
```typescript
await mlPlayer.load();  // อย่าลืม await!
```

### ONNX export failed
- เช็คว่า PyTorch model architecture ตรงกับตอน train
- ติดตั้ง: `pip install onnx onnxruntime`

### ML plays illegal moves
- Bug in `getTopMoves()` - legal mask ต้องถูกต้อง
- ตรวจสอบว่า `generateMoves()` ให้ legal moves ที่ถูกต้อง

---

## 📈 Next Steps

1. ✅ Test ML vs AI (10 games)
2. ✅ ประเมินผล (ML ชนะ/แพ้/เสมอ)
3. ถ้าดี → Integrate ใน UI
4. ถ้าแย่ → Gen data เพิ่ม (5000 เกม) → Train ใหม่
5. Production: Train 10,000+ เกม, larger model

**Happy Testing! 🎮🤖**