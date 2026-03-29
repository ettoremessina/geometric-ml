import numpy as np
import trimesh
from .base import ShapeGenerator


class LampGenerator(ShapeGenerator):
    """Floor/table lamp: disc base + slim pole + conical shade."""

    def generate(self) -> trimesh.Trimesh:
        base_r = self._u(0.08, 0.16)
        base_h = self._u(0.02, 0.05)
        pole_r = self._u(0.008, 0.018)
        pole_h = self._u(0.50, 1.10)
        shade_r = self._u(0.12, 0.22)
        shade_h = self._u(0.15, 0.28)

        base = self._cylinder(base_r, base_h,
                               transform=self._translate(0, 0, base_h / 2))

        pole = self._cylinder(pole_r, pole_h,
                               transform=self._translate(0, 0, base_h + pole_h / 2))

        shade_z = base_h + pole_h + shade_h / 2
        shade = self._cone(shade_r, shade_h,
                            transform=self._translate(0, 0, shade_z))

        return self._combine([base, pole, shade])
