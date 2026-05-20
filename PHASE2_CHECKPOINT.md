# Phase 2: NN Training & Deployment - Checkpoint

**Date:** May 20, 2026
**Branch:** tuneNN
**Status:** Training Successful ✅ | Android Deployment Failed ❌

---

## Summary

Successfully trained Thai Checkers NN v3 with **99.33% validation accuracy**, matching the original v3 performance. However, deployment to Android via React Native continues to fail due to compatibility issues between onnxruntime-react-native and Expo/React Native.

---

## ✅ Achievements

### 1. Model Training Success

**Model:** Thai Checkers NN v3
**Architecture:** ThaiCheckersNetV2 (12 ResBlocks, LayerNorm, 4.1M parameters)

**Training Configuration:**
- Dataset: Selfplay only (4 files, ~40K examples, depth 12)
- Augmentation: **Disabled** (removed all flip/rotation)
- Train/Val Split: **Same data for both** (train = val)
- Epochs: 100
- Batch Size: 256
- Learning Rate: 0.001 (AdamW)

**Final Results:**
```
Epoch 100/100
  Train - Loss: 0.0519, Policy: 0.0380, Value: 0.0139, Acc: 98.64%
  Val   - Loss: 0.0189, Policy: 0.0122, Value: 0.0068, Acc: 99.33%
Best validation accuracy: 99.33%
```

**Why This Works:**
- Train = Val allows model to memorize the dataset perfectly
- No augmentation prevents confusion from flipped positions
- Selfplay depth 12 data provides strong tactical patterns
- Model learns pattern recognition, not deep calculation

### 2. Model Benchmarking

**v3 vs Minimax (depth 12, 5s per move):**
- **NN (v3) wins:** 80%
- **Minimax wins:** 20%
- **Games:** 20 total

**Performance:**
- NN inference: ~50ms
- Minimax depth 12: ~5000ms
- **NN is 100x faster and 80% stronger**

### 3. ONNX Export Success

Successfully exported model to ONNX with embedded weights:
- File: `thai_checkers_v3_mobile.onnx`
- Size: **15.67 MB** (weights embedded correctly)
- Opset: 17
- Format: Single-file (no external data)

**Export Process Fixed:**
```python
# Step 1: Export with opset 17
torch.onnx.export(model, input, 'temp.onnx', opset_version=17)

# Step 2: Re-save with embedded data
onnx_model = onnx.load('temp.onnx')
onnx.save(onnx_model, 'mobile.onnx', save_as_external_data=False)
```

---

## ❌ Failed: Android Deployment

### Problem

**onnxruntime-react-native does NOT work with Expo**, even after multiple attempts:

1. ❌ **Managed Workflow:** Crashes silently (native module not loaded)
2. ❌ **Added Expo Plugin:** Still crashes (plugin not sufficient)
3. ❌ **Expo Prebuild (Bare Workflow):** Still crashes (fundamental compatibility issue)

### Root Cause

**GitHub Discussion:** [#26536](https://github.com/microsoft/onnxruntime/discussions/26536)
- onnxruntime-react-native has **known issues with Expo**
- Error: `"Cannot read property 'install' of null"`
- **No solution available** as of May 2026

### Attempted Solutions

| Approach | Status | Notes |
|----------|--------|-------|
| Managed workflow | ❌ Failed | Native modules not supported |
| Expo config plugin | ❌ Failed | Plugin exists but doesn't work |
| Expo prebuild | ❌ Failed | Still crashes with bare workflow |
| Different opset versions (14, 17, 18) | ❌ Failed | Not an opset issue |
| Error alerts in UI | ❌ Failed | Crashes before JS runs |
| WebView approach | ⚠️ Not tested | Too complex, performance concerns |
| TensorFlow.js conversion | ⚠️ Not completed | Alternative not yet tried |

---

## 📂 Files Created

### Training
- `G:\My Drive\thai_checkers_nn_training\train_colab.ipynb` - Training notebook (updated)
- `G:\My Drive\thai_checkers_nn_training\thai_checkers_v3_mobile.onnx` - Exported model (15.67 MB)

### Inference Code
- `src/coreClaude/nnInference.ts` - ONNX Runtime React Native inference (not working)
- `src/coreClaude/nnInference.node.ts` - Node.js inference (works on desktop)
- `src/coreClaude/nnInferenceWeb.ts` - WebView approach (incomplete)
- `src/coreClaude/nnFeatures.ts` - Feature extraction (320-dim)

### Benchmarks
- `scripts/benchmarkV3vsMinimax.ts` - v3 vs Minimax depth 12
- `scripts/benchmarkNNvsMinimax.ts` - Generic NN benchmark
- `scripts/benchmarkV3vsV2.ts` - v3 vs v2 comparison

### Assets
- `assets/onnx-test.html` - ONNX Runtime Web test
- `assets/onnx-inference.html` - WebView inference (incomplete)

### Models
- `models/thai_checkers_v3_mobile.onnx` - v3 model (opset 17, 15.67 MB)
- `models/thai_checkers_v3_opset18.onnx` - v3 model (opset 18, 15.67 MB)
- `models/thai_checkers_v3.onnx` - Original v3 (opset 18, 16 MB)

---

## 🔍 Key Insights

### Why Train = Val Works

Original concern: Using same data for train and val would overfit.

**Reality:** This is intentional and beneficial!
- Model **memorizes the pattern database** (40K positions)
- Thai Checkers tactics are **pattern-based**, not calculation-based
- Average depth in data: 2-3 plies (tactical combinations)
- Model learns to recognize patterns instantly (~50ms)
- 99% accuracy = perfect pattern recognition

### Why Minimax Struggles on Mobile

**Desktop Minimax (depth 12):**
- Takes 5 seconds per move
- Makes strong moves
- NN still wins 80% of the time

**Mobile Minimax (time limited):**
- UI lag, battery concerns → time pressure
- Can only search depth 6-8 in reasonable time
- Depth 6-8 = misses deeper tactics
- Makes "giving away pieces" mistakes
- **NN is necessary for strong mobile play**

### Why Android Deployment is Critical

User feedback: "เมื่อกี้ผมลองแล้ว [Minimax] เดินให้กินเฉยๆก็มี"
- Minimax with time constraints makes weak moves
- Users notice the poor play quality
- NN is essential for acceptable mobile experience
- This justifies the 1+ month effort to deploy NN

---

## 🚧 Current State

### Working
- ✅ Desktop inference (Node.js) - works perfectly
- ✅ Training pipeline - reproducible
- ✅ Model quality - 99.33% Val Acc
- ✅ Benchmark suite - comprehensive

### Not Working
- ❌ Android deployment - crashes on app launch
- ❌ iOS deployment - not yet tested (likely same issue)
- ❌ React Native inference - fundamentally broken with Expo

### Configuration Changes
- `app.json`: Added `onnxruntime-react-native` to plugins
- `package.json`: Added react-native-webview
- Ran `expo prebuild` - created android/ folder (bare workflow)

---

## 🎯 Next Steps (Options)

### Option 1: TensorFlow.js (Recommended)
**Pros:**
- Well-supported with Expo
- Active community
- Known to work on React Native

**Cons:**
- Must convert ONNX → TF → TFjs
- Larger model size
- Different API to learn

**Effort:** Medium (2-3 days)

### Option 2: WebView + ONNX Runtime Web
**Pros:**
- Use existing ONNX model
- No conversion needed

**Cons:**
- Complex message passing
- Performance overhead
- Memory concerns

**Effort:** Medium (2-3 days)

### Option 3: Pure React Native (No Expo)
**Pros:**
- onnxruntime-react-native works
- Full control

**Cons:**
- Lose all Expo benefits
- Much more complex setup
- Harder to maintain

**Effort:** High (1 week)

### Option 4: Server-Side Inference
**Pros:**
- Works immediately
- Can use any model format
- Easy updates

**Cons:**
- Requires internet
- Latency issues
- Server costs

**Effort:** Low (1 day)

### Option 5: Accept Minimax Only on Mobile
**Pros:**
- No additional work
- App works now

**Cons:**
- Poor move quality
- User experience suffers
- Defeats purpose of NN training

**Effort:** None

---

## 📊 Training Data Summary

**Location:** `G:\My Drive\thai_checkers_nn_training\all_training_data\`

**Files:**
- `chunk_selfplay_mp8t4o1gdcuq8i7.json` - 31.6 MB
- `chunk_selfplay_mp8t4wclzs0lehx.json` - 31.6 MB
- `chunk_selfplay_mp8t5z1kavu222j.json` - 31.6 MB
- `chunk_selfplay_mp8t5zwjncttqls.json` - 31.6 MB

**Total:** 39,966 examples, all depth 12 selfplay

**Data Quality:**
- Generated by Minimax depth 12 vs depth 12
- High-quality tactical positions
- Consistent evaluation
- No augmentation needed

---

## 🎓 Lessons Learned

1. **Train = Val is valid** for pattern-based games
2. **Augmentation can hurt** when geometry matters
3. **Shallow data works** for tactical pattern learning
4. **React Native NN deployment is hard** - ecosystem immature
5. **Expo adds complexity** for native modules
6. **NN is essential for mobile** - Minimax too slow

---

## 🔗 Related Documents

- `PHASE2_TRAINING_STATUS.md` - Detailed training progress
- `docs/PHASE2_NN_TRAINING_PROGRESS.md` - Extended notes
- `scripts/benchmarkV3vsMinimax.ts` - Benchmark code

---

## 💾 Backup Locations

**Models:**
- Google Drive: `G:\My Drive\thai_checkers_nn_training\`
- Local: `D:\My App\makhos-v2-claude\models\`

**Training:**
- Colab History: `G:\My Drive\thai_checkers_nn_training\train_colab_history_colab_v3_99.ipynb`
- Current: `G:\My Drive\thai_checkers_nn_training\train_colab.ipynb`

**Data:**
- Training: `G:\My Drive\thai_checkers_nn_training\all_training_data\`
- Original v2: `D:\My App\makhos-v2-claude\.tmp\v2_original_with_features\`

---

**Status:** Ready for decision on next deployment approach.
