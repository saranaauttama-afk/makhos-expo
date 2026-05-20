# Phase 2: Neural Network Training - Progress Report

## Session Summary (2026-05-15)

Successfully completed the PyTorch training pipeline infrastructure for Thai Checkers Neural Network V2.

---

## ✅ Completed Tasks

### 1. Neural Network Architecture (V2)
**File**: [src/coreClaude/nnFeatures.ts](../src/coreClaude/nnFeatures.ts)

Implemented 320-dimensional feature extraction:
- **[0-127]**: Positional features (my/opponent men/kings)
- **[128-191]**: Mobility features (legal moves per square)
- **[192-255]**: Threat maps (squares attacked by pieces)
- **[256-287]**: Hanging pieces (threatened + undefended)
- **[288-319]**: Distance to promotion (normalized 0-1)

**Test Results**:
- Initial position: 44 non-zero features ✅
- Midgame: 44 non-zero features, 2 hanging detected ✅
- Endgame: 32 non-zero features ✅

### 2. PyTorch Model Definition
**File**: [python/model.py](../python/model.py)

Implemented ThaiCheckersNetV2:
```python
- Input: 320 features
- Hidden: 384 with 12 ResBlocks
- Policy Head: 384 → 256 → 1024 logits
- Value Head: 384 → 128 → 1 (tanh)
- Total params: ~2.5M
```

**Key improvements over V1**:
- 5x more parameters (500K → 2.5M)
- 3x deeper (4 → 12 ResBlocks)
- 2.5x richer features (128 → 320)
- LayerNorm + Dropout for better training

### 3. Training Scripts
**Files**:
- [python/train.py](../python/train.py) - Local training
- [python/train_colab.ipynb](../python/train_colab.ipynb) - GPU training on Colab
- [scripts/exportFeaturesForPython.ts](../scripts/exportFeaturesForPython.ts) - Feature extraction bridge

**Configuration**:
- Batch size: 256
- Learning rate: 0.001 (AdamW)
- Loss: Cross-entropy (policy) + MSE (value)
- Scheduler: ReduceLROnPlateau
- Target epochs: 100

### 4. Training Data Generation (In Progress)
**Files**:
- [scripts/generateTrainingDataIncremental.ts](../scripts/generateTrainingDataIncremental.ts)

**Status**: 614/1000 positions (61%) - First chunk
- Estimated time: ~33 minutes per 1K chunk
- Resume-able (handles laptop sleep)
- Minimax depth 12 labels

---

## 📊 Architecture Comparison

| Aspect | V1 (Old) | V2 (New) | Improvement |
|--------|----------|----------|-------------|
| **Features** | 128 binary | 320 float32 | +2.5x richer |
| **Hidden** | 256 | 384 | +50% |
| **ResBlocks** | 4 | 12 | +3x depth |
| **Params** | 500K | 2.5M | +5x capacity |
| **Normalization** | BatchNorm | LayerNorm | Better for RL |
| **Regularization** | None | Dropout 0.1 | Prevents overfit |
| **Feature Quality** | Position only | +Mobility +Threats +Tactics | Domain knowledge |

---

## 🎯 Expected Performance

### After Supervised Learning (Phase 2):
- **Policy accuracy**: 80-85% (match minimax best move)
- **Value MSE**: <0.1
- **ELO**: ~1900-2000
- **Strength**: "Strong amateur" → "Expert level"

### Success Criteria:
- ✅ NN achieves 80% best move accuracy
- ✅ NN reaches Expert 90/10 benchmark
- ✅ ONNX model < 15MB
- ✅ Inference < 100ms on mobile

---

## 📁 File Structure

```
makhos-v2-claude/
├── src/coreClaude/
│   └── nnFeatures.ts          ✅ 320-dim feature extraction
├── scripts/
│   ├── generateTrainingDataIncremental.ts  ⏳ In progress (614/1000)
│   ├── exportFeaturesForPython.ts          ✅ Created
│   └── testNNFeatures.ts                   ✅ Tested
├── python/
│   ├── model.py               ✅ PyTorch model (2.5M params)
│   ├── train.py               ✅ Local training script
│   ├── train_colab.ipynb      ✅ Colab notebook
│   ├── requirements.txt       ✅ Dependencies
│   └── README.md              ✅ Documentation
└── docs/
    ├── NN_ARCHITECTURE_V2.md  ✅ Architecture spec
    └── PHASE2_NN_TRAINING_PROGRESS.md  ✅ This file
```

---

## ⏭️ Next Steps

### Immediate (This Session):
1. ✅ Wait for first chunk to complete (614/1000, ~10 min remaining)
2. ⏳ Run `exportFeaturesForPython.ts` to convert features
3. ⏳ Test training pipeline on first chunk (1K examples)

### Short-term (1-2 Days):
1. Generate remaining chunks to reach 10K-20K examples
2. Train on Colab GPU (~2 hours for 10K examples)
3. Evaluate model performance:
   - Policy accuracy
   - Value prediction
   - Training curves

### Medium-term (1 Week):
1. Generate full 100K dataset (run incremental script ~100 times)
2. Train complete model on Colab GPU (~6-8 hours)
3. Export ONNX model
4. Test NN vs Minimax benchmark:
   - Target: 80-90% best move accuracy
   - Target: 90-95% win rate vs minimax

### Long-term (Phase 3 - Optional):
1. **Self-Play Fine-Tuning**:
   - NN vs NN + minimax opponent pool
   - Discover strategies beyond minimax
   - Target: >50% win rate vs minimax depth 12

2. **Mobile Deployment**:
   - Integrate ONNX model into React Native app
   - Test inference speed on mobile GPU
   - Compare with minimax performance

---

## 🔧 Technical Notes

### Feature Extraction Fix
Fixed type error in [nnFeatures.ts:151](../src/coreClaude/nnFeatures.ts#L151):
```typescript
// WRONG (caused error)
for (const [toSq, dir] of moves) {

// CORRECT
for (const step of moves) {
  const toSq = step.to;
  const dir = step.dir;
```

### Training Data Format
Each example includes:
```json
{
  "position": { ... },
  "features": [320 floats],
  "bestMove": { "from": 24, "to": 20 },
  "positionValue": 0.123,
  "depth": 12,
  "nodes": 45678
}
```

### Workflow Pipeline
```
1. generateTrainingDataIncremental.ts
   ↓ Creates chunk_*.json with positions + minimax labels

2. exportFeaturesForPython.ts
   ↓ Extracts 320-dim features using nnFeatures.ts

3. train.py / train_colab.ipynb
   ↓ Trains PyTorch model with supervised learning

4. Export ONNX
   ↓ Deploy to mobile app
```

---

## 🎓 Lessons Learned

### Why V1 Failed (95% vs minimax 100%):
1. **Too simple features**: Only 128 binary (position only)
2. **Too shallow network**: 4 ResBlocks insufficient for tactics
3. **Poor training data**: Self-play with weak NN → garbage in, garbage out

### V2 Improvements:
1. **Richer features**: 320 dims with mobility, threats, tactical info
2. **Deeper network**: 12 ResBlocks for complex pattern recognition
3. **Expert labels**: Supervised learning from minimax depth 12
4. **Modern techniques**: LayerNorm, Dropout, AdamW optimizer

---

## 📚 References

- **Architecture**: Inspired by Leela Chess Zero
- **Training**: Supervised learning (AlphaGo style)
- **Features**: Domain knowledge (chess engine heuristics)
- **Deployment**: ONNX Runtime for mobile inference

---

## 🚀 Current Branch: `tuneNN`

This work is isolated in the `tuneNN` branch:
- **Stable minimax**: `tuneClaude` branch
- **Experimental NN**: `tuneNN` branch (current)
- **Backup**: `expert-100-perfect` branch

Can always return to `tuneClaude` for stable APK export.

---

**Last Updated**: 2026-05-15 03:40 UTC
**Status**: Training pipeline complete, data generation 61% done
**Next**: Wait for chunk completion, then test training
