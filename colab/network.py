"""
network.py — Policy + Value Network สำหรับ AlphaZero Makhos
============================================================

Architecture:
  Input   : 128 features (one-hot board, side-to-move relative)
  Body    : Linear(128→256) → BN → ReLU → Linear(256→256) → BN → ReLU
  Policy  : Linear(256→128) → log-softmax   (128 action logits)
  Value   : Linear(256→64) → ReLU → Linear(64→1) → Tanh  (-1..+1)

Output ของ policy_value() ที่ MCTS เรียก:
  policy : np.ndarray shape (128,)  — raw logits (MCTS จะ softmax+mask เอง)
  value  : float  —  +1 คาดว่าฝั่งนี้ชนะ, -1 คาดว่าแพ้

Usage:
  net = Network()
  net.save('weights.pt')
  net.load('weights.pt')
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class _NetModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared body
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # Policy head
        self.pol_fc = nn.Linear(256, 128)

        # Value head
        self.val_fc1 = nn.Linear(256, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 128)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))

        pol = self.pol_fc(h)          # (B, 128) raw logits
        val = F.tanh(self.val_fc2(F.relu(self.val_fc1(h))))  # (B, 1)

        return pol, val.squeeze(-1)   # (B, 128), (B,)


class Network:
    """Wrapper ให้ MCTS ใช้งานง่าย (numpy in → numpy out)."""

    def __init__(self, weights_path: str | None = None):
        self.model = _NetModule().to(DEVICE)
        if weights_path:
            self.load(weights_path)
        self.model.eval()

    # ── Inference (เรียกจาก MCTS) ────────────────────────────────────────────

    def policy_value(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        x : (128,) float32
        returns: (policy_logits: ndarray[128], value: float)
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)   # (1,128)
            pol, val = self.model(t)
            return pol[0].cpu().numpy(), float(val[0].cpu())

    def policy_value_batch(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        xs : (B, 128) float32
        returns: policy (B,128), value (B,)
        — ใช้ใน training loop
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(xs).to(DEVICE)
            pol, val = self.model(t)
            return pol.cpu().numpy(), val.cpu().numpy()

    # ── Training ──────────────────────────────────────────────────────────────

    def train_batch(
        self,
        xs:  np.ndarray,   # (B, 128)
        pis: np.ndarray,   # (B, 128)  policy targets (visit-count probs)
        zs:  np.ndarray,   # (B,)      value targets  (+1/-1/0)
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        One gradient step.
        Returns (policy_loss, value_loss) as Python floats.
        """
        self.model.train()
        x   = torch.from_numpy(xs).to(DEVICE)
        pi  = torch.from_numpy(pis).to(DEVICE)
        z   = torch.from_numpy(zs).to(DEVICE)

        pol_logits, val = self.model(x)

        # Policy loss: cross-entropy  H(pi, softmax(logits))
        log_softmax = F.log_softmax(pol_logits, dim=-1)
        pol_loss    = -(pi * log_softmax).sum(dim=-1).mean()

        # Value loss: MSE
        val_loss = F.mse_loss(val, z.float())

        loss = pol_loss + val_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(pol_loss), float(val_loss)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()
        print(f"Loaded weights ← {path}")

    def copy_weights_from(self, other: 'Network'):
        """Clone weights from another Network instance (for best-model tracking)."""
        self.model.load_state_dict(other.model.state_dict())
        self.model.eval()
