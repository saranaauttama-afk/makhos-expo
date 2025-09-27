# Makhos ML Training Pipeline

Complete neural network training pipeline for Thai Checkers (Makhos)

## 📋 Overview

3-step pipeline for training a neural network:
1. **Generate Data**: AI vs AI self-play (5,000 games, quality focused)
2. **Train Model**: Neural network training
3. **Export Model**: Save for deployment

## 🚀 Quick Start (Google Colab)

### Phase 1: Generate Training Data

**Recommended: Run on Colab for better performance**

```bash
# Clone repository
!git clone https://github.com/YOUR_USERNAME/makhos-expo.git
%cd makhos-expo/ml

# Install Node.js (for game generation)
!apt-get update && apt-get install -y nodejs npm
!npm install -g tsx

# Install project dependencies
%cd ..
!npm install
%cd ml

# Install Python dependencies
!pip install numpy torch

# Generate 5,000 games in batches of 1,000
# 1000ms per move (matches manual AI vs AI quality)
# Takes ~2-3 hours on Colab
!python gen_data.py --total_games 5000 --batch_size 1000 --time_per_move 1000
```

**Output**: `training_data.npz` (~1.2-1.5 GB)

**Progress tracking example**:
```
============================================================
GENERATION PLAN
============================================================
Total games: 5000
Batch size: 1000
Total batches: 5
Starting from batch: 0
Time per move: 1000ms
============================================================

============================================================
Batch 0: Generating 1000 games...
============================================================
✓ Batch 0 complete in 30.2 minutes

────────────────────────────────────────────────────────────
PROGRESS: 1000/5000 games (20.0%)
Elapsed: 30.2 min
ETA: 120.8 min (2.0 hours)
────────────────────────────────────────────────────────────
```

**Resume support**: If disconnected, re-run the same command and choose 'y' when asked to resume.

### Phase 2: Train Neural Network

```bash
# Simple MLP model (faster, good baseline)
!python train.py \
    --data training_data.npz \
    --model_type simple \
    --hidden_size 512 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001

# Or ResNet model (slower, potentially better)
!python train.py \
    --data training_data.npz \
    --model_type resnet \
    --num_channels 128 \
    --num_res_blocks 6 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

**Output**: `checkpoints/best_model.pt` and `checkpoints/final_model_scripted.pt`

### Phase 3: Download Models

```python
from google.colab import files

# Download the trained models
files.download('training_data.npz')
files.download('checkpoints/best_model.pt')
files.download('checkpoints/final_model_scripted.pt')
```

## 📊 Data Format

### Input (6 planes × 32 squares)
- Plane 0: P1 men
- Plane 1: P1 kings
- Plane 2: P2 men
- Plane 3: P2 kings
- Plane 4: Side to move
- Plane 5: Halfmove clock

### Output
- **Policy**: (32 × 32) move probabilities from square i to square j
- **Value**: [-1, 1] position evaluation

### Dataset Statistics (5,000 games @ 1000ms/move)
- **Positions**: ~200,000
- **File size**: 1.2-1.5 GB (compressed)
- **Uncompressed**: ~4.5 GB
- **Avg legal moves**: 6-8 per position
- **Generation time**: ~2-3 hours on Colab

## 🔧 Advanced Usage

### Generate smaller dataset for testing
```bash
# Quick test (100 games, ~5 minutes)
python gen_data.py --total_games 100 --batch_size 100 --time_per_move 500
```

### Generate high-quality dataset (slower but better)
```bash
# 5,000 games @ 1200ms/move (matches manual AI vs AI)
python gen_data.py --total_games 5000 --batch_size 1000 --time_per_move 1200
```

### Generate large dataset (10,000 games)
```bash
# Takes 5-6 hours on Colab
python gen_data.py --total_games 10000 --batch_size 2000 --time_per_move 1000
```

### Resume interrupted generation
```bash
# Just re-run the same command
python gen_data.py --total_games 5000 --batch_size 1000 --time_per_move 1000
# When prompted: y (to resume)
```

### Process existing game batches without regenerating
```bash
python gen_data.py --skip_generation --batch_dir game_batches
```

### Train with custom parameters
```bash
python train.py \
    --data training_data.npz \
    --model_type simple \
    --hidden_size 1024 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.0005 \
    --policy_weight 1.0 \
    --value_weight 1.5 \
    --save_every 5
```

## 🎯 Time per Move Recommendations

| Setting | Quality | Speed | Use Case |
|---------|---------|-------|----------|
| 500ms | Basic | Fast | Quick testing |
| 1000ms | **Good** | **Balanced** | **Recommended** |
| 1200ms | High | Slow | Matches manual AI |
| 1500ms+ | Best | Very slow | Production |

## 📁 File Structure

```
ml/
├── gen_data.py          # Step 1: Data generation
├── model.py             # Neural network architectures
├── train.py             # Step 2: Training script
├── requirements.txt     # Python dependencies
├── colab_example.ipynb  # Colab notebook (step-by-step)
├── README.md           # This file
├── game_batches/       # Generated game batches (created by gen_data.py)
│   ├── games_batch_0000.json
│   ├── games_batch_0001.json
│   └── ...
└── checkpoints/        # Trained models (created by train.py)
    ├── best_model.pt
    ├── final_model.pt
    └── final_model_scripted.pt
```

## ⚙️ Model Architectures

### SimpleMakhosNet (Recommended for prototyping)
- MLP-based architecture
- 3 hidden layers (512 units each)
- ~1.5M parameters
- Fast training (~30 min for 50 epochs on Colab GPU)

### MakhosNet (Advanced)
- ResNet-style with 1D convolutions
- 6 residual blocks (128 channels)
- ~3M parameters
- Slower training (~1 hour for 50 epochs on Colab GPU)

## 📈 Training Tips

1. **Start small**: Test with 100 games first to verify pipeline
2. **Use GPU**: Enable GPU in Colab (Runtime → Change runtime type → GPU)
3. **Monitor overfitting**: Check val_loss vs train_loss
4. **Adjust learning rate**: If loss plateaus, reduce learning rate
5. **Balance losses**: Tune `--policy_weight` and `--value_weight`

## 🐛 Troubleshooting

### "No games were generated"
- Check Node.js installation: `node --version`
- Check tsx installation: `npx tsx --version`
- Verify project dependencies: `npm install` in root directory

### "Out of memory" during training
- Reduce `--batch_size` (try 16 or 32)
- Use SimpleMakhosNet instead of MakhosNet
- Reduce `--hidden_size` (try 256)

### Training too slow
- Enable GPU in Colab
- Reduce `--num_res_blocks` (try 4)
- Use SimpleMakhosNet

### Disconnected during generation
- Re-run the same command
- Choose 'y' when asked to resume
- Previous batches are preserved

## 📝 Citation

If you use this code, please cite:
```
Makhos (Thai Checkers) ML Training Pipeline
https://github.com/YOUR_USERNAME/makhos-expo
```

## 🤝 Contributing

Improvements welcome! Areas to explore:
- AlphaZero-style self-play training
- Monte Carlo Tree Search integration
- Faster data generation
- Better evaluation metrics

## 📄 License

[Your License Here]