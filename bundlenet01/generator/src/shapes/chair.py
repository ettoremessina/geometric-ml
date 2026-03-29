import numpy as np
import trimesh
from .base import ShapeGenerator


class ChairGenerator(ShapeGenerator):
    """Chair: seat + backrest + 4 legs."""

    def generate(self) -> trimesh.Trimesh:
        seat_w = self._u(0.4, 0.6)
        seat_d = self._u(0.4, 0.55)
        seat_h = self._u(0.04, 0.08)
        leg_h  = self._u(0.35, 0.50)
        leg_r  = self._u(0.02, 0.035)
        back_h = self._u(0.35, 0.55)
        back_t = self._u(0.03, 0.06)

        seat_z = leg_h + seat_h / 2
        seat = self._box([seat_w, seat_d, seat_h],
                         self._translate(0, 0, seat_z))

        # backrest centred at rear of seat
        back_z = seat_z + seat_h / 2 + back_h / 2
        back_y = -(seat_d / 2 - back_t / 2)
        back = self._box([seat_w, back_t, back_h],
                         self._translate(0, back_y, back_z))

        # 4 legs at corners
        ox = seat_w / 2 - leg_r
        oy = seat_d / 2 - leg_r
        legs = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                T = self._translate(sx * ox, sy * oy, leg_h / 2)
                legs.append(self._cylinder(leg_r, leg_h, transform=T))

        return self._combine([seat, back] + legs)
