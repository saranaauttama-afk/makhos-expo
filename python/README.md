# Thai Checkers Neural Network V2 - Training Pipeline

PyTorch implementation for training Thai Checkers Neural Network V2.

## Architecture

- **Input**: 320 features (position + mobility + threats + tactical + strategic)
- **Hidden**: 384 with 12 ResBlocks
- **Output**: Policy head (1024 logits) + Value head (1 tanh)
- **Total params**: ~2.5M

## Files

- `model.py` - Neural network architecture definition
- `train.py` - Training script (local CPU/GPU)
- `train_colab.ipynb` - Google Colab notebook (GPU training)
- `requirements.txt` - Python dependencies

## Workflow

### 1. Generate Training Data

First, generate training data using TypeScript scripts:

```bash
# Generate 100K positions in 1K chunks
cd "D:\My App\makhos-v2-claude"
npx tsx scripts/generateTrainingDataIncremental.ts
```

This will create files in `.tmp/training_data/chunk_*.json`

### 2. Extract Features

Convert positions to 320-dim features:

```bash
npx tsx scripts/exportFeaturesForPython.ts
```

This will create files in `.tmp/training_data_with_features/chunk_*.json`

### 3. Train Model

**Option A: Train Locally (if you have GPU)**

```bash
cd python
pip install -r requirements.txt
python train.py
```

**Option B: Train on Google Colab (Recommended)**

1. Upload `train_colab.ipynb` to Google Colab
2. Change runtime to GPU (Runtime → Change runtime type → GPU → T4 or better)
3. Upload training data:
   - Create folder `thai_checkers_data` in Google Drive
   - Upload all `chunk_*.json` from `.tmp/training_data_with_features/`
4. Run all cells in notebook

### 4. Export Results

After training completes:

1. **Download trained model**:
   - `checkpoints/best_model.pt` - Best PyTorch checkpoint
   - `checkpoints/thai_checkers_v2.onnx` - ONNX for mobile deployment

2. **Check performance**:
   - Target: 80-90% best move accuracy
   - Training curves should show convergence

## Model Testing

Test model architecture:

```bash
python model.py
```

Expected output:
```
===============================================================================
Thai Checkers Neural Network V2
===============================================================================
Total parameters: 2,542,977

Input shape: torch.Size([32, 320])
Policy logits shape: torch.Size([32, 1024])
Value shape: torch.Size([32, 1])

Prediction test:
  Best move: (24, 20)
  Probabilities: ['0.250', '0.250', '0.250', '0.250']
  Value: 0.123

✅ Model architecture working correctly!
```

## Training Configuration

Default hyperparameters in `train.py`:

```python
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = cuda  # or cpu
```

**Expected training time**:
- GPU (T4): ~6-8 hours for 100K examples, 100 epochs
- GPU (A100): ~2-3 hours
- CPU: ~48 hours (not recommended)

## Expected Performance

### After Supervised Learning:
- Policy accuracy: 80-85% (match minimax best move)
- Value MSE: <0.1
- ELO: ~1900-2000

### Success Criteria:
- ✅ Validation accuracy > 80%
- ✅ Training converges (loss decreases)
- ✅ ONNX model < 15MB
- ✅ No overfitting (train/val gap < 5%)

## Troubleshooting

**Problem**: Out of memory on GPU

**Solution**: Reduce batch size:
```python
BATCH_SIZE = 128  # or 64
```

**Problem**: Training too slow

**Solution**:
- Use Google Colab Pro (faster GPU)
- Reduce epochs to 50
- Use smaller dataset (10K examples for testing)

**Problem**: Model not converging

**Solution**:
- Check data quality (run `exportFeaturesForPython.ts` again)
- Reduce learning rate: `LEARNING_RATE = 0.0005`
- Increase dropout: `dropout = 0.2`

## Next Steps

After successful training:

1. **Test NN vs Minimax**:
   - Load ONNX model in TypeScript
   - Play 100 games: NN vs Minimax depth 12
   - Target: 90-95% win rate

2. **Deploy to Mobile**:
   - Copy `thai_checkers_v2.onnx` to React Native project
   - Use ONNX Runtime for inference
   - Expected latency: 20-100ms per move

3. **Phase 2: Self-Play Fine-Tuning** (Optional):
   - Train NN vs NN + minimax opponent pool
   - Discover strategies beyond minimax
   - Target: >50% win rate vs minimax

## References

- Architecture inspired by Leela Chess Zero
- Feature engineering based on domain knowledge
- Supervised learning from minimax depth 12 labels
