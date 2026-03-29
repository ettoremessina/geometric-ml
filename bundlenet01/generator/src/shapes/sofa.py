import numpy as np
import trimesh
from .base import ShapeGenerator


class SofaGenerator(ShapeGenerator):
    """Sofa: seat cushion + backrest + two armrests + 4 short legs."""

    def generate(self) -> trimesh.Trimesh:
        seat_w = self._u(1.2, 1.9)
        seat_d = self._u(0.55, 0.80)
        seat_h = self._u(0.12, 0.20)
        leg_h  = self._u(0.08, 0.16)
        leg_r  = self._u(0.025, 0.04)

        back_h = self._u(0.45, 0.70)
        back_t = self._u(0.10, 0.18)

        arm_h  = back_h * self._u(0.45, 0.65)
        arm_t  = self._u(0.10, 0.16)

        seat_z = leg_h + seat_h / 2
        seat = self._box([seat_w, seat_d, seat_h],
                          self._translate(0, 0, seat_z))

        # Backrest at the rear
        back_z = seat_z + seat_h / 2 + back_h / 2
        back_y = -(seat_d / 2 - back_t / 2)
        back = self._box([seat_w, back_t, back_h],
                          self._translate(0, back_y, back_z))

        # Armrests on the sides
        arm_x = seat_w / 2 + arm_t / 2
        arm_z = seat_z + seat_h / 2 + arm_h / 2
        arm_d = seat_d + back_t
        arm_y = back_y + (seat_d / 2)  # centred along depth including back
        left_arm  = self._box([arm_t, arm_d, arm_h],
                               self._translate( arm_x, arm_y / 2, arm_z))
        right_arm = self._box([arm_t, arm_d, arm_h],
                               self._translate(-arm_x, arm_y / 2, arm_z))

        # 4 legs
        ox = seat_w / 2 - leg_r
        oy = seat_d / 2 - leg_r
        legs = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                T = self._translate(sx * ox, sy * oy, leg_h / 2)
                legs.append(self._cylinder(leg_r, leg_h, transform=T))

        return self._combine([seat, back, left_arm, right_arm] + legs)
