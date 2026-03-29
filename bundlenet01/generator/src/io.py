"""Dataset I/O: save point clouds as .ply files in ModelNet-style layout.

Layout:
    <root>/
        <class_name>/
            train/   (80 % of samples)
            test/    (20 % of samples)
"""

import os
import numpy as np


def save_point_cloud(cloud: np.ndarray, path: str) -> None:
    """Save a point cloud as a binary little-endian PLY file.

    Args:
        cloud: (N, 3) XYZ  or  (N, 6) XYZ + normals (NxNyNz).
    """
    n, cols = cloud.shape
    assert cols in (3, 6), f"Expected 3 or 6 columns, got {cols}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    props = "property float x\nproperty float y\nproperty float z\n"
    if cols == 6:
        props += "property float nx\nproperty float ny\nproperty float nz\n"

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"{props}"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(cloud.astype("<f4").tobytes())


def class_output_dir(root: str, class_name: str, split: str) -> str:
    return os.path.join(root, class_name, split)


def sample_filename(class_name: str, idx: int) -> str:
    return f"{class_name}_{idx:04d}.ply"


def train_test_split(n: int, test_ratio: float = 0.2, rng: np.random.Generator | None = None):
    """Return (train_indices, test_indices) for n samples."""
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    return indices[n_test:].tolist(), indices[:n_test].tolist()
