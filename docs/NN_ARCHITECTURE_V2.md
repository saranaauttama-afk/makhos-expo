# Neural Network Architecture V2 - Thai Checkers

## Overview
Improved architecture addressing shortcomings of V1 (old NN that achieved only 95% vs minimax 100%).

---

## V1 Problems (Why NN Lost to Minimax)

### 1. **Feature Representation Too Simple (128 dims)**
```
Old features (128 binary):
- [0-31]:   My men positions
- [32-63]:  My kings positions
- [64-95]:  Opponent men positions
- [96-127]: Opponent kings positions
```

**Problems:**
- ❌ No mobility information
- ❌ No hanging piece detection
- ❌ No threat maps
- ❌ No distance to promotion
- ❌ No king safety zones

**Result:** NN must learn everything from scratch, while minimax has built-in knowledge!

### 2. **Network Too Small**
```python
Old architecture:
- Hidden: 256
- ResBlocks: 4  # Too shallow!
- Total params: ~500K
```

**Problems:**
- ❌ Insufficient capacity for complex patterns
- ❌ Cannot model deep tactics
- ❌ Leela Chess Zero uses 30-40 blocks, we only had 4!

### 3. **Training Data Quality**
- Self-play with weak NN → garbage in, garbage out
- No expert guidance (minimax depth 12)
- No tactical puzzle training

---

## V2 Architecture (Improved)

### **Feature Engineering (128 → 320 dims)**

#### **Positional Features (128 dims)** - Same as V1
- [0-31]: My men
- [32-63]: My kings
- [64-95]: Opponent men
- [96-127]: Opponent kings

#### **Mobility Features (64 dims)** - NEW!
- [128-159]: My pieces' mobility (legal moves count per square)
- [160-191]: Opponent pieces' mobility

#### **Threat Maps (64 dims)** - NEW!
- [192-223]: Squares threatened by my pieces
- [224-255]: Squares threatened by opponent

#### **Tactical Features (32 dims)** - NEW!
- [256-287]: Hanging piece flags (undefended + threatened)

#### **Strategic Features (32 dims)** - NEW!
- [288-303]: Distance to promotion (my men)
- [304-319]: Distance to promotion (opponent men)

**Total: 320 float32 features**

---

### **Network Architecture**

```python
class ThaiCheckersNetV2(nn.Module):
    def __init__(self, hidden=384, n_res=12):
        super().__init__()

        # Stem: 320 → 384
        self.stem = nn.Sequential(
            nn.Linear(320, hidden),
            nn.LayerNorm(hidden),  # Better than BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Trunk: 12 Residual Blocks (up from 4!)
        self.trunk = nn.Sequential(
            *[ResBlock(hidden, dropout=0.1) for _ in range(n_res)]
        )

        # Policy Head: 384 → 1024 logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1024),  # 32×32 move space
        )

        # Value Head: 384 → 1 (tanh)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

class ResBlock(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))  # Residual connection
```

---

## Architecture Comparison

| Aspect | V1 (Old) | V2 (New) | Improvement |
|--------|----------|----------|-------------|
| **Input Features** | 128 binary | 320 float32 | +2.5x richer |
| **Hidden Size** | 256 | 384 | +50% capacity |
| **ResBlocks** | 4 | 12 | +3x depth |
| **Total Params** | ~500K | ~2.5M | +5x capacity |
| **Normalization** | BatchNorm | LayerNorm | Better for RL |
| **Regularization** | None | Dropout 0.1 | Prevents overfit |
| **Feature Quality** | Position only | +Mobility +Threats +Tactics | Domain knowledge |

---

## Training Strategy V2

### **Phase 1: Supervised Learning (Week 1-2)**
**Goal:** Learn from Expert minimax (100/0)

1. **Generate 100K positions**
   - 40K from self-play (diverse positions)
   - 30K from opening book variations
   - 20K from tactical positions
   - 10K from endgame positions

2. **Label with minimax depth 12**
   - Best move (policy target)
   - Position value (value target)
   - PV (principal variation)

3. **Train supervised**
   - Loss: Cross-entropy (policy) + MSE (value)
   - Optimizer: AdamW (lr=0.001)
   - Batch size: 256
   - Epochs: 50-100
   - Target: 80% best move accuracy

### **Phase 2: Fine-tuning with Self-Play (Week 3-4)**
**Goal:** Discover new strategies beyond minimax

1. **Self-play with opponent pool**
   - Minimax depth 8, 10, 12
   - NN generation n, n-1, n-2
   - 30% opening book injection

2. **Loss mining**
   - Collect positions where NN loses
   - Re-train on hard examples

3. **Evaluation**
   - NN vs minimax depth 12
   - Target: 90-95% win rate

---

## Expected Performance

### **After Supervised Learning (Phase 1):**
- NN accuracy: 80-85% vs minimax best moves
- ELO: ~1900-2000 (vs minimax 2000 baseline)
- Strength: "Strong amateur"

### **After Self-Play (Phase 2):**
- NN accuracy: 85-90%
- ELO: ~2000-2100 (may surpass minimax!)
- Strength: "Expert level"
- Discovery: New opening/endgame strategies

---

## Mobile Deployment

### **ONNX Export:**
```python
torch.onnx.export(
    model,
    dummy_input,
    "thai_checkers_v2.onnx",
    input_names=['features'],
    output_names=['policy_logits', 'value'],
    dynamic_axes={'features': {0: 'batch'}},
)
```

### **Model Size:**
- V1: ~2MB (500K params)
- V2: ~10MB (2.5M params)
- Still acceptable for mobile!

### **Inference Speed:**
- CPU: ~50-100ms per position
- Mobile GPU: ~20-50ms per position
- Acceptable for 1-second moves

---

## Success Criteria

### **Minimum Viable Product (MVP):**
- ✅ NN achieves 80% best move accuracy
- ✅ NN reaches Expert 90/10 benchmark
- ✅ ONNX model < 15MB
- ✅ Inference < 100ms on mobile

### **Stretch Goals:**
- 🎯 NN achieves 90% best move accuracy
- 🎯 NN reaches Expert 95/5 benchmark
- 🎯 NN discovers new strategies (better than minimax in some positions)
- 🎯 Win rate > 50% vs minimax depth 12

---

## Risks & Mitigation

### **Risk 1: Training time too long**
- **Mitigation:** Use Colab Pro GPU, batch size 256
- **Fallback:** Reduce to 50K positions, 6 ResBlocks

### **Risk 2: Overfitting**
- **Mitigation:** Dropout 0.1, validation set 20%
- **Fallback:** Early stopping, L2 regularization

### **Risk 3: NN still worse than minimax**
- **Mitigation:** Start with supervised learning (guaranteed 80%+)
- **Fallback:** Use NN for move ordering in minimax (hybrid approach)

### **Risk 4: Model too large for mobile**
- **Mitigation:** Quantization (FP32 → FP16)
- **Fallback:** Reduce hidden size 384 → 256

---

## Next Steps

1. ✅ Create feature extraction code (320 dims)
2. ✅ Implement PyTorch model
3. ✅ Create training data generator
4. ✅ Generate 100K labeled positions
5. ✅ Train on Colab GPU
6. ✅ Export ONNX
7. ✅ Test vs minimax benchmark

**Timeline:** 2-3 weeks full-time work

---

## Comparison with Leela Chess Zero

| Aspect | Leela Chess Zero | Thai Checkers V2 | Notes |
|--------|------------------|------------------|-------|
| Input | 112 planes (8×8×14) | 320 floats (32 squares) | Smaller board |
| Hidden | 256 filters | 384 hidden | Proportional |
| ResBlocks | 20-40 | 12 | Simpler game |
| Training | 1000+ GPU-years | ~24 GPU-hours | Much faster! |
| Performance | ELO 3400+ | Target ELO 2100 | Human expert level |

**Key insight:** Thai Checkers is simpler than chess, so we need less compute!

---

## Conclusion

V2 architecture addresses all shortcomings of V1:
- ✅ Richer features (320 vs 128)
- ✅ Deeper network (12 vs 4 blocks)
- ✅ Better training data (supervised from minimax)
- ✅ Modern techniques (LayerNorm, Dropout)

**Expected result:** NN that matches or exceeds minimax Expert 100/0!
