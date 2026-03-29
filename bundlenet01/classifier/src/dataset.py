"""PyG Dataset that reads the .ply files produced by the generator.

Expected layout (ModelNet-style):
    <root>/
        <class_name>/
            train/  *.ply
            test/   *.ply

Each .ply contains N×6 float32 data (XYZ + normals NxNyNz) written by
generator/src/io.py.  Files with only 3 properties (legacy XYZ-only) are
also accepted — in that case data.x will be None.
"""

import os
import glob

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# Canonical class ordering — must match ALL_GENERATORS keys in generator
CLASSES = [
    "airplane", "bookshelf", "bottle", "car", "chair",
    "lamp", "monitor", "mug", "sofa", "table",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a binary little-endian PLY file written by the generator.

    Returns:
        xyz:     (N, 3) float32 — spatial coordinates
        normals: (N, 3) float32 — surface normals, or None if not present
    """
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break

        n_verts = 0
        n_props = 0
        for line in header.split(b"\n"):
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith(b"property float"):
                n_props += 1

        raw = f.read(n_verts * n_props * 4)

    data = np.frombuffer(raw, dtype="<f4").reshape(n_verts, n_props).copy()
    xyz     = data[:, :3]
    normals = data[:, 3:6] if n_props >= 6 else None
    return xyz, normals


class PointCloudDataset(Dataset):
    """Loads point clouds from a ModelNet-style directory tree.

    Each sample is a ``torch_geometric.data.Data`` with:
        - ``pos``:  (N, 3) float32 — XYZ coordinates
        - ``x``:    (N, 3) float32 — surface normals (or None for legacy files)
        - ``y``:    scalar long     — class index

    Args:
        root:      path to the dataset root (e.g. ``data/ModelNet10``).
        split:     ``"train"`` or ``"test"``.
        transform: optional callable applied to each ``Data`` object.
        classes:   subset of class names to load (default: all CLASSES).
    """

    def __init__(self, root: str, split: str = "train",
                 transform=None, classes: list[str] | None = None):
        super().__init__(root=None, transform=transform)
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        self.split = split
        self.class_list = classes if classes is not None else CLASSES

        self._files: list[str] = []
        self._labels: list[int] = []

        for class_name in self.class_list:
            label = CLASS_TO_IDX[class_name]
            pattern = os.path.join(root, class_name, split, "*.ply")
            for path in sorted(glob.glob(pattern)):
                self._files.append(path)
                self._labels.append(label)

    def len(self) -> int:
        return len(self._files)

    def get(self, idx: int) -> Data:
        xyz, normals = _read_ply(self._files[idx])
        pos = torch.from_numpy(xyz)
        x   = torch.from_numpy(normals) if normals is not None else None
        y   = torch.tensor(self._labels[idx], dtype=torch.long)
        return Data(pos=pos, x=x, y=y)
