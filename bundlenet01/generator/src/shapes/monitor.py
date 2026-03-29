import numpy as np
import trimesh
from .base import ShapeGenerator


class MonitorGenerator(ShapeGenerator):
    """Monitor: flat screen panel + neck + T-shaped base."""

    def generate(self) -> trimesh.Trimesh:
        scr_w = self._u(0.35, 0.65)
        scr_h = self._u(0.22, 0.40)
        scr_t = self._u(0.015, 0.030)

        neck_r = self._u(0.012, 0.022)
        neck_h = self._u(0.10, 0.20)

        base_w = scr_w * self._u(0.50, 0.75)
        base_d = self._u(0.15, 0.26)
        base_h = self._u(0.015, 0.030)

        # Screen panel — face pointing in Y direction, standing upright
        scr_z = neck_h + base_h + scr_h / 2
        screen = self._box([scr_w, scr_t, scr_h],
                            self._translate(0, 0, scr_z))

        # Neck connecting base to screen bottom
        neck_z = base_h + neck_h / 2
        neck = self._cylinder(neck_r, neck_h,
                               transform=self._translate(0, 0, neck_z))

        # Base disc/box on the ground
        base = self._box([base_w, base_d, base_h],
                          self._translate(0, 0, base_h / 2))

        return self._combine([screen, neck, base])
