"""Point cloud sampling and augmentation."""

import numpy as np
import trimesh


def sample_points_and_normals(mesh: trimesh.Trimesh,
                               n_points: int = 2048) -> np.ndarray:
    """Uniformly sample n_points from the surface of a mesh.

    Returns an (n_points, 6) float32 array: columns 0-2 are XYZ coordinates,
    columns 3-5 are the unit surface normals (Nx, Ny, Nz) at each sample.
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_indices]          # (N, 3), already unit-length
    return np.concatenate([points, normals], axis=1).astype(np.float32)


def normalize(cloud: np.ndarray) -> np.ndarray:
    """Centre XYZ at the origin and scale to unit sphere.

    cloud: (N, 6) — only the XYZ columns (0:3) are shifted/scaled;
    normals (3:6) are unaffected (they are direction vectors, not positions).
    """
    xyz = cloud[:, :3]
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz /= scale
    cloud = cloud.copy()
    cloud[:, :3] = xyz
    return cloud


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random SO(3) rotation matrix (3×3)."""
    H = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def random_rotation(cloud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the same SO(3) rotation to both XYZ and normals.

    Normals are vectors in the tangent space — they must transform identically
    to position vectors under rotation (no translation component).
    """
    Q = _random_rotation_matrix(rng)
    cloud = cloud.copy()
    cloud[:, :3] = (Q @ cloud[:, :3].T).T   # rotate positions
    cloud[:, 3:] = (Q @ cloud[:, 3:].T).T   # rotate normals
    return cloud.astype(np.float32)


def random_scale(cloud: np.ndarray, rng: np.random.Generator,
                 lo: float = 0.8, hi: float = 1.25) -> np.ndarray:
    """Scale XYZ by a random factor; normals are unit vectors and stay unchanged."""
    s = float(rng.uniform(lo, hi))
    cloud = cloud.copy()
    cloud[:, :3] *= s
    return cloud.astype(np.float32)


def augment(cloud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply normalisation → random rotation → random scale."""
    cloud = normalize(cloud)
    cloud = random_rotation(cloud, rng)
    cloud = random_scale(cloud, rng)
    return cloud
