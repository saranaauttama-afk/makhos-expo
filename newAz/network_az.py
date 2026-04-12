"""
network_az.py — AlphaZero Policy + Value Network for Thai Checkers (Makhos)
===========================================================================
Architecture:
  Input : 128-dim feature vector (4 planes × 32 squares, current-player relative)
  Trunk : FC(128→256) + BN + ReLU → 4 × ResBlock(256)
  Policy: FC(256→128) + BN + ReLU → FC(128→1024)   [logit per from_sq*32+to_sq]
  Value : FC(256→64)  + BN + ReLU → FC(64→1) + Tanh

Move encoding: move_index = from_sq * 32 + to_sq  ∈ [0, 1024)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MOVES = 32 * 32   # 1024


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return F.relu(x + r)


class AlphaZeroNet(nn.Module):
    def __init__(self, hidden: int = 256, n_res: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(128, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])

        # Policy head — outputs logits over all 1024 possible (from, to) slots
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, N_MOVES),
        )

        # Value head — outputs scalar ∈ [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.trunk(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ── Wrapper ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class AZNetwork:
    def __init__(self, hidden: int = 256, n_res: int = 4):
        self.hidden = hidden
        self.n_res  = n_res
        self.net    = AlphaZeroNet(hidden, n_res).to(DEVICE)

    def predict(self, features: np.ndarray, legal_indices: list) -> tuple:
        """Single-position inference used during MCTS.

        Args:
            features:      shape (128,) float32
            legal_indices: list of valid move indices (from_sq*32+to_sq)

        Returns:
            probs : np.ndarray shape (len(legal_indices),)  — policy probs over legal moves
            value : float  — position value in [-1,1] from current player's perspective
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
            p_logits, v = self.net(x)
        p_logits = p_logits[0].cpu().numpy()
        probs = _softmax(p_logits[legal_indices])
        return probs, float(v[0].cpu())

    def predict_batch(self, features: np.ndarray) -> tuple:
        """Batch inference for training.

        Args:
            features: shape (B, 128)
        Returns:
            policy_logits: (B, 1024)
            values:        (B,)
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(DEVICE)
            p, v = self.net(x)
        return p.cpu().numpy(), v.cpu().numpy()

    def save(self, path: str):
        torch.save({
            'state_dict': self.net.state_dict(),
            'hidden':     self.hidden,
            'n_res':      self.n_res,
        }, path)
        print(f'  Saved → {path}')

    def load(self, path: str):
        ck = torch.load(path, map_location=DEVICE)
        self.net.load_state_dict(ck['state_dict'])
        self.net.eval()

    def copy(self) -> 'AZNetwork':
        new = AZNetwork(self.hidden, self.n_res)
        new.net.load_state_dict(self.net.state_dict())
        new.net.eval()
        return new


if __name__ == '__main__':
    net = AZNetwork()
    x = np.zeros(128, dtype=np.float32)
    probs, v = net.predict(x, list(range(10)))
    print(f'OK  probs={probs[:3]}  value={v:.4f}  device={DEVICE}')
