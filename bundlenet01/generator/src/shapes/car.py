import numpy as np
import trimesh
from .base import ShapeGenerator


class CarGenerator(ShapeGenerator):
    """Car: boxy lower body + cabin + 4 wheels."""

    def generate(self) -> trimesh.Trimesh:
        body_l = self._u(0.8, 1.4)
        body_w = self._u(0.35, 0.55)
        body_h = self._u(0.18, 0.28)

        cab_l  = body_l * self._u(0.40, 0.58)
        cab_w  = body_w * self._u(0.85, 0.95)
        cab_h  = body_h * self._u(0.70, 0.95)

        wheel_r = body_h * self._u(0.42, 0.55)
        wheel_w = body_w * self._u(0.14, 0.22)

        body_z = body_h / 2 + wheel_r * 0.5
        body = self._box([body_l, body_w, body_h],
                          self._translate(0, 0, body_z))

        cab_z = body_z + body_h / 2 + cab_h / 2
        cab_x = body_l * self._u(-0.10, 0.05)   # slightly off-centre rearward
        cab = self._box([cab_l, cab_w, cab_h],
                         self._translate(cab_x, 0, cab_z))

        # 4 wheels
        wx = body_l / 2 - wheel_r * 0.8
        wy = body_w / 2 + wheel_w / 2
        wheel_z = wheel_r * 0.5
        Rw = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        wheels = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                T = np.eye(4)
                T[0, 3] = sx * wx
                T[1, 3] = sy * wy
                T[2, 3] = wheel_z
                w = self._cylinder(wheel_r, wheel_w, sections=24)
                w.apply_transform(Rw)
                w.apply_transform(T)
                wheels.append(w)

        return self._combine([body, cab] + wheels)
