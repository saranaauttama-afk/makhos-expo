# 📝 ML Integration TODO

## ✅ ทำเสร็จแล้ว
- [x] สร้าง export_onnx.py script
- [x] สร้าง TypeScript ML inference wrapper
- [x] สร้าง MLPlayer class
- [x] สร้าง useMLPlayer React Hook
- [x] อัพเดท inference.ts ให้ใช้ onnxruntime-react-native

---

## 🔲 Step ที่เหลือ (ทำค่ำๆ)

### 1. Export ONNX Model (ใน Google Colab)

```python
# Cell: Install dependencies
!pip install onnx onnxruntime -q

# Cell: Export to ONNX
%cd /content/makhos-expo/ml

!python export_onnx.py \
  --model /content/drive/MyDrive/makhos_ml/checkpoints/best_model.pt \
  --output /content/drive/MyDrive/makhos_ml/best_model.onnx \
  --model_type resnet \
  --num_channels 128 \
  --num_res_blocks 6

# Cell: Download
from google.colab import files
files.download('/content/drive/MyDrive/makhos_ml/best_model.onnx')
```

**Expected output:** ไฟล์ `best_model.onnx` (~25-30 MB)

---

### 2. วาง Model ใน Project

```bash
# วาง best_model.onnx ไว้ที่
D:\MyApp\makhos-expo\assets\best_model.onnx
```

---

### 3. Install Dependencies

```bash
cd D:\MyApp\makhos-expo

npm install onnxruntime-react-native
npx expo install expo-file-system expo-asset
```

---

### 4. Integrate ใน App

**Option A: ใช้ใน UI Component (แนะนำ)**

แก้ไขใน `App.tsx` หรือ game screen:

```tsx
import { useMLPlayer } from './src/core/ml/useMLPlayer';

function GameScreen() {
  const { mlPlayer, loading, error } = useMLPlayer();

  // AI move handler
  const handleAIMove = async (position: Position) => {
    if (!mlPlayer) return null;

    const move = await mlPlayer.selectMove(position);
    return move;
  };

  if (loading) return <Text>Loading AI...</Text>;
  if (error) return <Text>Error: {error}</Text>;

  return <Board onAIMove={handleAIMove} />;
}
```

**Option B: ทดสอบแบบง่าย (ในไฟล์ใหม่)**

สร้าง `scripts/testMLOnDevice.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { View, Text } from 'react-native';
import { useMLPlayer } from '../src/core/ml/useMLPlayer';
import { initialPosition } from '../src/core/position';

export default function TestML() {
  const { mlPlayer, loading, error } = useMLPlayer();
  const [result, setResult] = useState('');

  useEffect(() => {
    if (mlPlayer) {
      testModel();
    }
  }, [mlPlayer]);

  async function testModel() {
    const pos = initialPosition();
    const move = await mlPlayer!.selectMove(pos);
    setResult(`ML suggests: ${move?.from} → ${move?.to}`);
  }

  return (
    <View>
      <Text>{loading ? 'Loading...' : error || result}</Text>
    </View>
  );
}
```

---

### 5. Test บนมือถือ

```bash
npx expo start

# กด 'a' (Android) หรือ 'i' (iOS)
# หรือสแกน QR code ด้วย Expo Go app
```

**Expected:**
- เห็น "Loading AI..." สักพัก
- แล้วขึ้น "ML suggests: 25 → 20" (ตัวอย่าง)
- หรือเล่นเกมได้ โดย AI ใช้ ML model

---

### 6. เช็คว่าใช้งานได้

✅ **Success indicators:**
- Model load เสร็จไม่มี error
- AI เลือก move ได้ (ไม่ช้ามาก ~100-500ms)
- Move ที่เลือกเป็น legal move
- เล่นกับ AI ได้จบเกม

❌ **ถ้ามี error:**
- "Cannot find module 'onnxruntime-react-native'" → ลง npm install อีกรอบ
- "Model not found" → เช็คว่าวางใน `assets/best_model.onnx`
- "Invalid model" → export ONNX ใหม่ (อาจเป็น architecture ผิด)
- แช่มากๆ → ลอง quantize model หรือใช้ SimpleMakhosNet

---

### 7. (Optional) แข่ง ML vs Traditional AI

ถ้าอยากเปรียบเทียบ:

```tsx
// ใน game logic
if (aiType === 'ML') {
  move = await mlPlayer.selectMove(position);
} else {
  // Traditional AI
  const result = iterativeDeepening(position, 1000, tt);
  move = result.best;
}
```

แล้วเล่น 10 เกม ดูว่า ML ชนะหรือแพ้

---

## 📊 Expected Results

**ถ้า ML ชนะ AI เก่า:**
- ✅ Model ใช้งานได้! Deploy เลย
- พิจารณา gen data เพิ่ม (5000 เกม) เพื่อ improve

**ถ้า ML แพ้ AI เก่า:**
- Model อาจ overfit หรือ data น้อยเกินไป
- ลองใช้ checkpoint_epoch_3.pt แทน (val loss ต่ำสุด)
- หรือ gen data เพิ่มเป็น 5000-10000 เกม

---

## 🐛 Known Issues & Solutions

### Issue 1: onnxruntime-react-native ไม่รองรับบาง ops
**Solution:** ใช้ `opset_version=14` ตอน export (ทำไว้แล้ว)

### Issue 2: Model ใหญ่เกิน → app ช้า
**Solution:**
- Quantize model: `torch.quantization.quantize_dynamic()`
- หรือใช้ SimpleMakhosNet (~1.5MB แทน ~12MB)

### Issue 3: iOS ไม่ work
**Solution:** ต้องลง native dependencies:
```bash
cd ios && pod install
```

---

## 📚 Files Created

1. `ml/export_onnx.py` - Export script
2. `src/core/ml/inference.ts` - ONNX inference wrapper
3. `src/core/ml/mlPlayer.ts` - ML AI player
4. `src/core/ml/useMLPlayer.ts` - React Hook
5. `scripts/testMLvsAI.ts` - Node.js test script
6. `ml/INTEGRATION.md` - Full guide
7. `ml/TODO.md` - This file

---

## 🎯 Summary

**ค่ำนี้ทำ:**
1. Export ONNX ใน Colab (5 นาที)
2. Download + วางใน assets/ (1 นาที)
3. npm install dependencies (2 นาที)
4. Integrate useMLPlayer ใน App (5-10 นาที)
5. Test บนมือถือ (5 นาที)

**รวม: ~20-30 นาที**

**Goal:** เห็น AI เล่นด้วย ML model บนมือถือ! 📱🤖

---

**หมายเหตุ:** ถ้าติดตรงไหนให้ดูใน `ml/INTEGRATION.md` หรือถามได้เลย!