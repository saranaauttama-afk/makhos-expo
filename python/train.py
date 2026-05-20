"""
Training script for Thai Checkers Neural Network V2

Supervised learning from minimax depth 12 labels
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import ThaiCheckersNetV2


class ThaiCheckersDataset(Dataset):
    """Dataset for Thai Checkers training data (with pre-extracted features)"""

    def __init__(self, data_files: List[str]):
        """
        Args:
            data_files: List of JSON file paths containing training data with features
        """
        self.examples = []

        # Load all data files
        for file_path in data_files:
            print(f'Loading {file_path}...')
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.examples.extend(data)

        print(f'Total examples: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Features already extracted by TypeScript
        features = example['features'][:]  # Make a copy

        # Policy target (best move as one-hot)
        best_move = example['bestMove']
        from_sq = best_move['from']
        to_sq = best_move['to']
        move_idx = from_sq * 32 + to_sq

        # Value target
        value = example['positionValue']

        # Data augmentation: 50% chance to flip P1/P2 (side symmetry)
        if np.random.rand() < 0.5:
            features = self._flip_features(features)
            from_sq, to_sq = self._flip_move(from_sq, to_sq)
            move_idx = from_sq * 32 + to_sq
            value = -value  # Flip value for opponent's perspective

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'move_idx': move_idx,
            'value': torch.tensor([value], dtype=torch.float32),
        }

    def _flip_features(self, features):
        """Flip board features for P1/P2 side symmetry augmentation"""
        flipped = features[:]  # Make a copy

        # Flip positional features [0-127]: my_men, my_kings, op_men, op_kings
        # Swap my pieces <-> opponent pieces
        for layer in range(4):
            for sq in range(32):
                # Swap layers: 0↔2 (my_men↔op_men), 1↔3 (my_kings↔op_kings)
                target_layer = 2 + layer if layer < 2 else layer - 2
                flipped[target_layer * 32 + (31 - sq)] = features[layer * 32 + sq]

        # Flip mobility features [128-191]: my_mobility, op_mobility
        for layer in range(2):
            for sq in range(32):
                # Swap layers: 0↔1 (my_mobility↔op_mobility)
                target_layer = 1 - layer
                flipped[128 + target_layer * 32 + (31 - sq)] = features[128 + layer * 32 + sq]

        # Flip threat maps [192-255]: my_threats, op_threats
        for layer in range(2):
            for sq in range(32):
                # Swap layers: 0↔1 (my_threats↔op_threats)
                target_layer = 1 - layer
                flipped[192 + target_layer * 32 + (31 - sq)] = features[192 + layer * 32 + sq]

        # Flip hanging flags [256-287]: hanging pieces
        for sq in range(32):
            flipped[256 + (31 - sq)] = features[256 + sq]

        # Flip distance to promotion [288-319]: promotion distance
        for sq in range(32):
            # Distance to promotion also inverts (distance from bottom → distance from top)
            # 7 - distance because board height is 8
            original_dist = features[288 + sq]
            flipped[288 + (31 - sq)] = 7 - original_dist if original_dist > 0 else 0

        return flipped

    def _flip_move(self, from_sq, to_sq):
        """Flip move indices for P1/P2 side symmetry"""
        return (31 - from_sq, 31 - to_sq)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        features = batch['features'].to(device)
        move_idx = batch['move_idx'].to(device)
        value_target = batch['value'].to(device)

        # Forward pass
        policy_logits, value_pred = model(features)

        # Policy loss (cross-entropy)
        policy_loss = nn.functional.cross_entropy(policy_logits, move_idx)

        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value_pred, value_target)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

        # Accuracy
        pred_moves = torch.argmax(policy_logits, dim=1)
        correct += (pred_moves == move_idx).sum().item()
        total += len(move_idx)

    return {
        'loss': total_loss / len(dataloader),
        'policy_loss': total_policy_loss / len(dataloader),
        'value_loss': total_value_loss / len(dataloader),
        'accuracy': correct / total,
    }


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            move_idx = batch['move_idx'].to(device)
            value_target = batch['value'].to(device)

            # Forward pass
            policy_logits, value_pred = model(features)

            # Policy loss
            policy_loss = nn.functional.cross_entropy(policy_logits, move_idx)

            # Value loss
            value_loss = nn.functional.mse_loss(value_pred, value_target)

            # Combined loss
            loss = policy_loss + value_loss

            # Statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            # Accuracy
            pred_moves = torch.argmax(policy_logits, dim=1)
            correct += (pred_moves == move_idx).sum().item()
            total += len(move_idx)

    return {
        'loss': total_loss / len(dataloader),
        'policy_loss': total_policy_loss / len(dataloader),
        'value_loss': total_value_loss / len(dataloader),
        'accuracy': correct / total,
    }




def main():
    # Configuration
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 80)
    print('Thai Checkers NN V2 - Supervised Training')
    print('=' * 80)
    print(f'Device: {DEVICE}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'Epochs: {EPOCHS}')
    print('')

    # Find training data files (with features)
    data_dir = Path('.tmp/training_data_with_features')
    train_files = list(data_dir.glob('chunk_*.json'))

    if not train_files:
        print('ERROR: No training data with features found!')
        print(f'Expected files in: {data_dir}')
        print('Please run:')
        print('  1. generateTrainingDataIncremental.ts')
        print('  2. exportFeaturesForPython.ts')
        return

    print(f'Found {len(train_files)} data chunks')
    print('')

    # Create dataset (80/20 train/val split)
    split_idx = int(len(train_files) * 0.8)
    train_dataset = ThaiCheckersDataset(
        [str(f) for f in train_files[:split_idx]]
    )
    val_dataset = ThaiCheckersDataset(
        [str(f) for f in train_files[split_idx:]]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f'Train examples: {len(train_dataset)}')
    print(f'Val examples: {len(val_dataset)}')
    print('')

    # Create model
    model = ThaiCheckersNetV2().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('')

    # Training loop
    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')

        # Train
        train_stats = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f'  Train - Loss: {train_stats["loss"]:.4f}, '
              f'Policy: {train_stats["policy_loss"]:.4f}, '
              f'Value: {train_stats["value_loss"]:.4f}, '
              f'Acc: {train_stats["accuracy"]:.2%}')

        # Validate
        val_stats = validate(model, val_loader, DEVICE)
        print(f'  Val   - Loss: {val_stats["loss"]:.4f}, '
              f'Policy: {val_stats["policy_loss"]:.4f}, '
              f'Value: {val_stats["value_loss"]:.4f}, '
              f'Acc: {val_stats["accuracy"]:.2%}')

        # Learning rate scheduling
        scheduler.step(val_stats['accuracy'])

        # Save best model
        if val_stats['accuracy'] > best_val_acc:
            best_val_acc = val_stats['accuracy']
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print(f'  ✅ New best model! Accuracy: {best_val_acc:.2%}')

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_stats': train_stats,
                'val_stats': val_stats,
            }, f'checkpoints/checkpoint_epoch_{epoch}.pt')

        print('')

    print('=' * 80)
    print('Training Complete!')
    print('=' * 80)
    print(f'Best validation accuracy: {best_val_acc:.2%}')
    print('')

    # Export ONNX
    print('Exporting to ONNX...')
    model.eval()
    dummy_input = torch.randn(1, 320).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        'checkpoints/thai_checkers_v2.onnx',
        input_names=['features'],
        output_names=['policy_logits', 'value'],
        dynamic_axes={'features': {0: 'batch'}},
    )
    print('✅ ONNX model saved to checkpoints/thai_checkers_v2.onnx')
    print('')


if __name__ == '__main__':
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    main()
