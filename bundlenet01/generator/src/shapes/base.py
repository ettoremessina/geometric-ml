from abc import ABC, abstractmethod
import numpy as np
import trimesh


class ShapeGenerator(ABC):
    """Base class for all procedural shape generators.

    Each subclass must implement `generate()` which returns a trimesh.Trimesh
    built from randomised parameters so that repeated calls produce varied
    instances of the same object category.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def _u(self, lo: float, hi: float) -> float:
        """Uniform sample in [lo, hi]."""
        return float(self.rng.uniform(lo, hi))

    @abstractmethod
    def generate(self) -> trimesh.Trimesh:
        """Return a single mesh instance with randomised proportions."""

    # ------------------------------------------------------------------
    # Helpers shared across generators
    # ------------------------------------------------------------------

    @staticmethod
    def _box(extents, transform=None) -> trimesh.Trimesh:
        m = trimesh.creation.box(extents=extents)
        if transform is not None:
            m.apply_transform(transform)
        return m

    @staticmethod
    def _cylinder(radius, height, sections=32, transform=None) -> trimesh.Trimesh:
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        if transform is not None:
            m.apply_transform(transform)
        return m

    @staticmethod
    def _cone(radius, height, sections=32, transform=None) -> trimesh.Trimesh:
        m = trimesh.creation.cone(radius=radius, height=height, sections=sections)
        if transform is not None:
            m.apply_transform(transform)
        return m

    @staticmethod
    def _sphere(radius, subdivisions=3, transform=None) -> trimesh.Trimesh:
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        if transform is not None:
            m.apply_transform(transform)
        return m

    @staticmethod
    def _translate(tx=0.0, ty=0.0, tz=0.0) -> np.ndarray:
        T = np.eye(4)
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        return T

    @staticmethod
    def _combine(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
        return trimesh.util.concatenate(parts)
