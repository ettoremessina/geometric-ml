import numpy as np
import trimesh
from .base import ShapeGenerator


class MugGenerator(ShapeGenerator):
    """Mug: hollow cylinder body + torus handle."""

    def generate(self) -> trimesh.Trimesh:
        r_out  = self._u(0.04, 0.07)
        r_in   = r_out * self._u(0.75, 0.88)
        height = self._u(0.08, 0.14)

        outer = self._cylinder(r_out, height)
        inner = self._cylinder(r_in,  height * 0.95)
        body  = trimesh.boolean.difference([outer, inner], engine="blender") if False else outer

        # Approximate handle as a torus: major radius = body radius + torus minor radius
        minor_r = self._u(0.008, 0.013)
        major_r = r_out + minor_r * 2.2
        handle  = trimesh.creation.torus(major_radius=major_r, minor_radius=minor_r)
        # Rotate torus so it hangs on the side (torus lies in XY by default)
        T = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        T[0, 3] = r_out + major_r  # push to the side
        handle.apply_transform(T)

        return self._combine([outer, handle])
