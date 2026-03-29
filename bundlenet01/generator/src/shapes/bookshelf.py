import numpy as np
import trimesh
from .base import ShapeGenerator


class BookshelfGenerator(ShapeGenerator):
    """Bookshelf: outer frame (top + bottom + 2 sides) + 2–4 inner shelves."""

    def generate(self) -> trimesh.Trimesh:
        width  = self._u(0.6, 1.1)
        depth  = self._u(0.22, 0.38)
        height = self._u(0.8, 1.6)
        panel_t = self._u(0.018, 0.032)
        n_shelves = int(self._rng_int(2, 4))

        parts = []

        # Left and right vertical panels
        for sx in (+1, -1):
            T = self._translate(sx * (width / 2 - panel_t / 2), 0, height / 2)
            parts.append(self._box([panel_t, depth, height], T))

        # Bottom and top horizontal panels
        inner_w = width - 2 * panel_t
        for sz_frac in (0.0, 1.0):
            T = self._translate(0, 0, sz_frac * (height - panel_t) + panel_t / 2)
            parts.append(self._box([inner_w, depth, panel_t], T))

        # Shelves evenly distributed
        shelf_spacing = (height - 2 * panel_t) / (n_shelves + 1)
        for i in range(1, n_shelves + 1):
            z = panel_t + i * shelf_spacing
            T = self._translate(0, 0, z)
            parts.append(self._box([inner_w, depth, panel_t], T))

        return self._combine(parts)

    def _rng_int(self, lo: int, hi: int) -> int:
        return int(self.rng.integers(lo, hi + 1))
