"""PointNet classification network.

Reference: Qi et al., "PointNet: Deep Learning on Point Sets for 3D
Classification and Segmentation", CVPR 2017.

Architecture
------------
Input (B, N, in_features)           default in_features=3 (XYZ)
    │
    ├─ TNet(in_features)  →  (B, in_features, in_features)  [input alignment]
    │
    ├─ SharedMLP [in_features → 64 → 64]
    │
    ├─ TNet(64)  →  (B, 64, 64)                              [feature alignment]
    │
    ├─ SharedMLP [64 → 64 → 128 → 1024]
    │
    ├─ Global Max Pool  →  (B, 1024)
    │
    ├─ FC(1024 → 512) + BN + ReLU + Dropout
    ├─ FC(512  → 256) + BN + ReLU + Dropout
    └─ FC(256  → num_classes)

Loss supplement: λ · ‖I − A·Aᵀ‖²_F  (feature T-Net regularisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Mini-network that regresses a k×k alignment matrix.

    Uses the same shared-MLP + max-pool + FC pattern as the main network.
    Initialised so that the output is close to the identity.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k,    64,  1)
        self.conv2 = nn.Conv1d(64,  128,  1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,  256)
        self.fc3 = nn.Linear(256,  k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialise last layer to zero so output starts as identity
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, N)
        Returns:
            transform: (B, k, k)
        """
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.max(dim=2)[0]                         # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)                              # (B, k*k)

        # Add identity matrix
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).expand(B, -1, -1)
        transform = x.view(B, self.k, self.k) + identity
        return transform


class PointNet(nn.Module):
    """PointNet classifier.

    Args:
        num_classes:  number of output classes.
        in_features:  number of input features per point (3 = XYZ only).
        dropout:      dropout probability in the classification head.
        tnet_reg_weight: λ for the feature T-Net regularisation loss.
    """

    def __init__(self, num_classes: int = 10, in_features: int = 3,
                 dropout: float = 0.3, tnet_reg_weight: float = 0.001):
        super().__init__()
        self.tnet_reg_weight = tnet_reg_weight

        # --- Encoder ---
        self.input_tnet = TNet(k=in_features)

        self.conv1 = nn.Conv1d(in_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(64)

        self.feature_tnet = TNet(k=64)

        self.conv3 = nn.Conv1d(64,   64,   1)
        self.conv4 = nn.Conv1d(64,  128,   1)
        self.conv5 = nn.Conv1d(128, 1024,  1)
        self.bn3   = nn.BatchNorm1d(64)
        self.bn4   = nn.BatchNorm1d(128)
        self.bn5   = nn.BatchNorm1d(1024)

        # --- Classification head ---
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,  256)
        self.fc3 = nn.Linear(256,  num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor,
                x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos:   (total_points, 3)          — XYZ coordinates (PyG convention)
            batch: (total_points,)             — batch index per point
            x:     (total_points, F) or None  — additional per-point features
                   (e.g. surface normals).  If provided, concatenated with pos
                   to form a (total_points, 3+F) input — requires in_features=3+F.
        Returns:
            logits:         (B, num_classes)
            feat_transform: (B, 64, 64)  — used to compute regularisation loss
        """
        # Concatenate pos and extra features if provided
        inp = torch.cat([pos, x], dim=-1) if x is not None else pos

        # Reshape to (B, in_features, N) for Conv1d
        B = int(batch.max().item()) + 1
        N = inp.size(0) // B
        x = inp.view(B, N, -1).permute(0, 2, 1).contiguous()   # (B, C, N)

        # Input transform
        t_in = self.input_tnet(x)                               # (B, C, C)
        x = torch.bmm(t_in, x)                                  # (B, C, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        t_feat = self.feature_tnet(x)                           # (B, 64, 64)
        x = torch.bmm(t_feat, x)                                # (B, 64, N)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))                     # (B, 1024, N)

        # Global max pool over the N dimension — avoids scatter_reduce (not on MPS)
        x = x.max(dim=2)[0]                                     # (B, 1024)

        # Classification head
        x = self.drop1(F.relu(self.bn6(self.fc1(x))))
        x = self.drop2(F.relu(self.bn7(self.fc2(x))))
        logits = self.fc3(x)                                     # (B, num_classes)

        return logits, t_feat

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def tnet_regularisation(transform: torch.Tensor) -> torch.Tensor:
        """‖I − A·Aᵀ‖²_F averaged over the batch."""
        B, k, _ = transform.shape
        I = torch.eye(k, device=transform.device).unsqueeze(0).expand(B, -1, -1)
        diff = I - torch.bmm(transform, transform.transpose(1, 2))
        return (diff ** 2).sum(dim=(1, 2)).mean()

    def loss(self, logits: torch.Tensor, targets: torch.Tensor,
             feat_transform: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets)
        reg  = self.tnet_regularisation(feat_transform)
        return ce + self.tnet_reg_weight * reg
