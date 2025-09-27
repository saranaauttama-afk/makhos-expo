"""
Step 2 & 3: Train neural network and save model

Train a neural network on self-play data for Makhos evaluation and move prediction
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import create_model

class MakhosDataset(Dataset):
    """PyTorch dataset for Makhos training data"""

    def __init__(self, states: np.ndarray, policy_targets: np.ndarray, legal_masks: np.ndarray, values: np.ndarray):
        """
        Args:
            states: (N, 6, 32) board states
            policy_targets: (N, 32, 32) policy distributions from search
            legal_masks: (N, 32, 32) legal move masks
            values: (N,) game outcomes
        """
        self.states = torch.from_numpy(states).float()
        self.policy_targets = torch.from_numpy(policy_targets).float()
        self.legal_masks = torch.from_numpy(legal_masks).float()
        self.values = torch.from_numpy(values).float().unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policy_targets[idx], self.legal_masks[idx], self.values[idx]

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from npz file"""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    states = data['states']
    policy_targets = data['policy_targets']
    legal_masks = data['legal_masks']
    values = data['values']
    print(f"  Loaded {len(states)} training examples")
    print(f"  Avg legal moves per position: {legal_masks.sum(axis=(1,2)).mean():.1f}")
    return states, policy_targets, legal_masks, values

def create_dataloaders(
    states: np.ndarray,
    policy_targets: np.ndarray,
    legal_masks: np.ndarray,
    values: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    dataset = MakhosDataset(states, policy_targets, legal_masks, values)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")

    return train_loader, val_loader

def policy_loss_fn(policy_logits: torch.Tensor, policy_targets: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute policy loss (cross-entropy with legal move masking)

    Args:
        policy_logits: (batch, 32, 32) raw logits
        policy_targets: (batch, 32, 32) target policy distribution
        legal_masks: (batch, 32, 32) legal move mask

    Returns:
        scalar loss
    """
    batch_size = policy_logits.size(0)

    # Mask illegal moves with large negative value
    masked_logits = policy_logits.clone()
    masked_logits = torch.where(legal_masks > 0, masked_logits, torch.tensor(-1e9))

    # Flatten for softmax
    logits_flat = masked_logits.view(batch_size, -1)
    targets_flat = policy_targets.view(batch_size, -1)

    # Log softmax + negative log likelihood
    log_probs = torch.log_softmax(logits_flat, dim=1)
    loss = -(targets_flat * log_probs).sum(dim=1).mean()

    return loss

def value_loss_fn(value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    """
    Compute value loss (MSE)

    Args:
        value_pred: (batch, 1) predicted values
        value_target: (batch, 1) target values

    Returns:
        scalar loss
    """
    return nn.functional.mse_loss(value_pred, value_target)

def train_epoch(model, train_loader, optimizer, device, policy_weight=1.0, value_weight=1.0):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for states, policy_targets, legal_masks, values in train_loader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        legal_masks = legal_masks.to(device)
        values = values.to(device)

        # Forward pass
        policy_logits, value_pred = model(states)

        # Compute losses
        p_loss = policy_loss_fn(policy_logits, policy_targets, legal_masks)
        v_loss = value_loss_fn(value_pred, values)
        loss = policy_weight * p_loss + value_weight * v_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_policy_loss += p_loss.item()
        total_value_loss += v_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }

def evaluate(model, val_loader, device, policy_weight=1.0, value_weight=1.0):
    """Evaluate on validation set"""
    model.eval()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    with torch.no_grad():
        for states, policy_targets, legal_masks, values in val_loader:
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            legal_masks = legal_masks.to(device)
            values = values.to(device)

            # Forward pass
            policy_logits, value_pred = model(states)

            # Compute losses
            p_loss = policy_loss_fn(policy_logits, policy_targets, legal_masks)
            v_loss = value_loss_fn(value_pred, values)
            loss = policy_weight * p_loss + value_weight * v_loss

            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }

def save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, filepath):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }, filepath)
    print(f"  Checkpoint saved to {filepath}")

def save_model_for_inference(model, filepath):
    """Save model for inference (model only, no optimizer)"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, filepath)
    print(f"  Model saved to {filepath}")

    # Also save as TorchScript for deployment
    model.eval()
    example_input = torch.randn(1, 6, 32)
    traced_model = torch.jit.trace(model, example_input)
    torchscript_path = filepath.replace('.pt', '_scripted.pt')
    traced_model.save(torchscript_path)
    print(f"  TorchScript model saved to {torchscript_path}")

def main():
    import sys

    # Jupyter/Colab compatibility: filter out kernel arguments
    if any('kernel' in arg.lower() for arg in sys.argv):
        sys.argv = [sys.argv[0]]  # Keep only script name

    parser = argparse.ArgumentParser(description="Train Makhos neural network")

    # Data
    parser.add_argument("--data", type=str, default="training_data.npz", help="Training data file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    # Model
    parser.add_argument("--model_type", type=str, default="simple", choices=["simple", "resnet"], help="Model architecture")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size (for simple model)")
    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels (for resnet model)")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Number of residual blocks (for resnet model)")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--policy_weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Value loss weight")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    states, policy_targets, legal_masks, values = load_data(args.data)
    train_loader, val_loader = create_dataloaders(states, policy_targets, legal_masks, values, args.batch_size, args.val_split)

    # Create model
    if args.model_type == "simple":
        model = create_model("simple", hidden_size=args.hidden_size)
    else:
        model = create_model("resnet", num_channels=args.num_channels, num_res_blocks=args.num_res_blocks)

    model = model.to(device)
    print(f"\nModel: {args.model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.policy_weight, args.value_weight)

        # Validate
        val_metrics = evaluate(model, val_loader, device, args.policy_weight, args.value_weight)

        epoch_time = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Policy: {train_metrics['policy_loss']:.4f}, Value: {train_metrics['value_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Policy: {val_metrics['policy_loss']:.4f}, Value: {val_metrics['value_loss']:.4f}")

        # Save checkpoints
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, checkpoint_path)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, best_model_path)
            print(f"  *** New best model (val_loss: {best_val_loss:.4f}) ***")

    # Save final model
    print("\nTraining complete!")
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    save_model_for_inference(model, final_model_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {args.output_dir}/")
    print(f"  - best_model.pt (best validation performance)")
    print(f"  - final_model.pt (last epoch)")
    print(f"  - final_model_scripted.pt (TorchScript for deployment)")

def train_model(
    data_path="training_data.npz",
    model_type="simple",
    hidden_size=512,
    num_channels=128,
    num_res_blocks=6,
    epochs=50,
    batch_size=32,
    lr=0.001,
    policy_weight=1.0,
    value_weight=1.0,
    output_dir="checkpoints",
    save_every=10
):
    """
    Helper function for Jupyter/Colab - call directly without argparse

    Example:
        train_model(data_path="training_data.npz", epochs=30, batch_size=64)
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    states, policy_targets, legal_masks, values = load_data(data_path)
    train_loader, val_loader = create_dataloaders(states, policy_targets, legal_masks, values, batch_size, val_split=0.1)

    # Create model
    if model_type == "simple":
        model = create_model("simple", hidden_size=hidden_size)
    else:
        model = create_model("resnet", num_channels=num_channels, num_res_blocks=num_res_blocks)

    model = model.to(device)
    print(f"\nModel: {model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...\n")
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, policy_weight, value_weight)

        # Validate
        val_metrics = evaluate(model, val_loader, device, policy_weight, value_weight)

        epoch_time = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Policy: {train_metrics['policy_loss']:.4f}, Value: {train_metrics['value_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Policy: {val_metrics['policy_loss']:.4f}, Value: {val_metrics['value_loss']:.4f}")

        # Save checkpoints
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, checkpoint_path)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, best_model_path)
            print(f"  *** New best model (val_loss: {best_val_loss:.4f}) ***")

    # Save final model
    print("\nTraining complete!")
    final_model_path = os.path.join(output_dir, "final_model.pt")
    save_model_for_inference(model, final_model_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {output_dir}/")
    print(f"  - best_model.pt (best validation performance)")
    print(f"  - final_model.pt (last epoch)")
    print(f"  - final_model_scripted.pt (TorchScript for deployment)")

if __name__ == "__main__":
    main()