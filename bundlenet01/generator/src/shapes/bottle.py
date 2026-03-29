import numpy as np
import trimesh
from .base import ShapeGenerator


class BottleGenerator(ShapeGenerator):
    """Bottle: wide cylinder body + tapered neck + small cap."""

    def generate(self) -> trimesh.Trimesh:
        body_r  = self._u(0.04, 0.075)
        body_h  = self._u(0.15, 0.28)
        neck_r  = body_r * self._u(0.35, 0.55)
        neck_h  = self._u(0.05, 0.10)
        cap_r   = neck_r * self._u(1.05, 1.20)
        cap_h   = self._u(0.015, 0.030)

        body = self._cylinder(body_r, body_h,
                              transform=self._translate(0, 0, body_h / 2))

        # Cone frustum as neck: use a cone and cut — approximate with a short cone
        taper_h = self._u(0.03, 0.06)
        taper = self._cone(body_r, taper_h,
                           transform=self._translate(0, 0, body_h + taper_h / 2))

        neck_z = body_h + taper_h + neck_h / 2
        neck = self._cylinder(neck_r, neck_h,
                              transform=self._translate(0, 0, neck_z))

        cap_z = body_h + taper_h + neck_h + cap_h / 2
        cap = self._cylinder(cap_r, cap_h,
                             transform=self._translate(0, 0, cap_z))

        return self._combine([body, taper, neck, cap])
