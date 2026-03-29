"""Point cloud transforms applied at load time.

Each transform is a callable that receives and returns a
``torch_geometric.data.Data`` object with a ``pos`` attribute (N×3 float32).
"""

import torch
from torch_geometric.data import Data


class NormalizePointCloud:
    """Centre the point cloud at the origin and scale to unit sphere.

    Operates on ``data.pos`` in-place (creates a new tensor).
    """

    def __call__(self, data: Data) -> Data:
        pos = data.pos                           # (N, 3)
        pos = pos - pos.mean(dim=0)              # centre
        scale = pos.norm(dim=1).max()
        if scale > 0:
            pos = pos / scale
        data.pos = pos
        return data


class RandomJitter:
    """Add small Gaussian noise — useful for future augmentation experiments.

    Not enabled by default; kept here for reference.
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data: Data) -> Data:
        noise = torch.clamp(
            torch.randn_like(data.pos) * self.sigma, -self.clip, self.clip
        )
        data.pos = data.pos + noise
        return data
