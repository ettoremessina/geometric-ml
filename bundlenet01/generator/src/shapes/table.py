import numpy as np
import trimesh
from .base import ShapeGenerator


class TableGenerator(ShapeGenerator):
    """Table: rectangular top + 4 cylindrical legs."""

    def generate(self) -> trimesh.Trimesh:
        top_w  = self._u(0.6, 1.2)
        top_d  = self._u(0.4, 0.8)
        top_h  = self._u(0.03, 0.07)
        leg_h  = self._u(0.55, 0.85)
        leg_r  = self._u(0.025, 0.045)

        table_z = leg_h + top_h / 2
        top = self._box([top_w, top_d, top_h],
                        self._translate(0, 0, table_z))

        ox = top_w / 2 - leg_r * 2
        oy = top_d / 2 - leg_r * 2
        legs = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                T = self._translate(sx * ox, sy * oy, leg_h / 2)
                legs.append(self._cylinder(leg_r, leg_h, transform=T))

        return self._combine([top] + legs)
