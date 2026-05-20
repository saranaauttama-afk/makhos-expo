"""
Thai Checkers Neural Network V2

Architecture:
- Input: 320 features (position + mobility + threats + tactical + strategic)
- Hidden: 384 with 12 ResBlocks
- Output: Policy head (1024 logits) + Value head (1 tanh)

Total params: ~2.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with LayerNorm and Dropout"""

    def __init__(self, hidden: int, dropout: float = 0.1):
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
        return F.relu(x + self.net(x))


class ThaiCheckersNetV2(nn.Module):
    """
    Neural Network V2 for Thai Checkers

    Args:
        hidden: Hidden size (default 384)
        n_res: Number of residual blocks (default 12)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(self, hidden: int = 384, n_res: int = 12, dropout: float = 0.1):
        super().__init__()

        # Stem: 320 → 384
        self.stem = nn.Sequential(
            nn.Linear(320, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Trunk: 12 Residual Blocks
        self.trunk = nn.Sequential(
            *[ResBlock(hidden, dropout=dropout) for _ in range(n_res)]
        )

        # Policy Head: 384 → 1024 logits (32×32 move space)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1024),
        )

        # Value Head: 384 → 1 (tanh)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, features):
        """
        Forward pass

        Args:
            features: (batch, 320) float32 tensor

        Returns:
            policy_logits: (batch, 1024) float32 tensor
            value: (batch, 1) float32 tensor in [-1, 1]
        """
        x = self.stem(features)
        x = self.trunk(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict_move(self, features, legal_moves):
        """
        Predict best move given legal moves

        Args:
            features: (320,) numpy array
            legal_moves: list of (from_sq, to_sq) tuples

        Returns:
            best_move: (from_sq, to_sq) tuple
            policy_probs: (len(legal_moves),) numpy array
            value: float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            policy_logits, value = self.forward(features_t)

            # Mask illegal moves
            mask = torch.full((1024,), float('-inf'))
            for from_sq, to_sq in legal_moves:
                idx = from_sq * 32 + to_sq
                mask[idx] = 0

            # Apply mask and softmax
            masked_logits = policy_logits[0] + mask
            policy_probs = F.softmax(masked_logits, dim=0)

            # Get best move
            best_idx = torch.argmax(policy_probs).item()
            best_from = best_idx // 32
            best_to = best_idx % 32

            # Get probabilities for legal moves
            legal_probs = []
            for from_sq, to_sq in legal_moves:
                idx = from_sq * 32 + to_sq
                legal_probs.append(policy_probs[idx].item())

            return (best_from, best_to), legal_probs, value.item()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = ThaiCheckersNetV2()

    print('=' * 80)
    print('Thai Checkers Neural Network V2')
    print('=' * 80)
    print(f'Total parameters: {count_parameters(model):,}')
    print('')

    # Test forward pass
    batch_size = 32
    features = torch.randn(batch_size, 320)

    policy_logits, value = model(features)

    print(f'Input shape: {features.shape}')
    print(f'Policy logits shape: {policy_logits.shape}')
    print(f'Value shape: {value.shape}')
    print('')

    # Test prediction
    single_features = torch.randn(320).numpy()
    legal_moves = [(24, 20), (25, 21), (26, 22), (27, 23)]

    best_move, probs, val = model.predict_move(single_features, legal_moves)

    print('Prediction test:')
    print(f'  Best move: {best_move}')
    print(f'  Probabilities: {[f"{p:.3f}" for p in probs]}')
    print(f'  Value: {val:.3f}')
    print('')
    print('✅ Model architecture working correctly!')
