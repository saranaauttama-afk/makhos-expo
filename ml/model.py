"""
Neural Network model for Makhos (Thai Checkers)

Architecture: ResNet-style with policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class MakhosNet(nn.Module):
    """
    Neural network for Makhos board evaluation and move prediction

    Input: (batch, 6, 32) - 6 feature planes, 32 squares
      - plane 0: P1 men
      - plane 1: P1 kings
      - plane 2: P2 men
      - plane 3: P2 kings
      - plane 4: side to move
      - plane 5: halfmove clock

    Outputs:
      - policy: (batch, 32, 32) - move probabilities from square i to square j
      - value: (batch, 1) - position evaluation [-1, 1]
    """

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 6):
        super().__init__()

        # Initial convolution to expand channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(6, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_channels),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * 32, 32 * 32)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv1d(num_channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(8 * 32, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, 6, 32) input tensor

        Returns:
            policy_logits: (batch, 32, 32) raw move logits
            value: (batch, 1) position value
        """
        # Shared representation
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        policy_logits = policy_logits.view(-1, 32, 32)

        # Value head
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

class SimpleMakhosNet(nn.Module):
    """
    Simpler MLP-based network (faster training, good baseline)

    Good for initial experiments and Colab with limited resources
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()

        input_size = 6 * 32  # 6 planes x 32 squares

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 32)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, 6, 32) input tensor

        Returns:
            policy_logits: (batch, 32, 32) raw move logits
            value: (batch, 1) position value
        """
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Shared representation
        features = self.trunk(x)

        # Policy head
        policy_logits = self.policy_head(features)
        policy_logits = policy_logits.view(batch_size, 32, 32)

        # Value head
        value = torch.tanh(self.value_head(features))

        return policy_logits, value

def create_model(model_type: str = "simple", **kwargs):
    """
    Factory function to create models

    Args:
        model_type: "simple" or "resnet"
        **kwargs: Additional arguments for model constructors

    Returns:
        model instance
    """
    if model_type == "simple":
        return SimpleMakhosNet(**kwargs)
    elif model_type == "resnet":
        return MakhosNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test models
    print("Testing SimpleMakhosNet...")
    model_simple = SimpleMakhosNet()
    x = torch.randn(4, 6, 32)
    policy, value = model_simple(x)
    print(f"  Input: {x.shape}")
    print(f"  Policy: {policy.shape}")
    print(f"  Value: {value.shape}")

    print("\nTesting MakhosNet...")
    model_resnet = MakhosNet(num_channels=64, num_res_blocks=4)
    policy, value = model_resnet(x)
    print(f"  Input: {x.shape}")
    print(f"  Policy: {policy.shape}")
    print(f"  Value: {value.shape}")

    print("\nModel parameter counts:")
    print(f"  SimpleMakhosNet: {sum(p.numel() for p in model_simple.parameters()):,}")
    print(f"  MakhosNet: {sum(p.numel() for p in model_resnet.parameters()):,}")
    print("\nMemory usage (approx):")
    print(f"  SimpleMakhosNet: {sum(p.numel() * 4 for p in model_simple.parameters()) / 1024 / 1024:.2f} MB")
    print(f"  MakhosNet: {sum(p.numel() * 4 for p in model_resnet.parameters()) / 1024 / 1024:.2f} MB")