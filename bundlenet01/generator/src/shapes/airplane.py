import numpy as np
import trimesh
from .base import ShapeGenerator


class AirplaneGenerator(ShapeGenerator):
    """Airplane: fuselage + main wings + horizontal stabilisers + vertical tail fin."""

    def generate(self) -> trimesh.Trimesh:
        fus_r  = self._u(0.04, 0.07)
        fus_l  = self._u(0.5,  0.9)

        wing_span = fus_l * self._u(0.9, 1.3)
        wing_chord = fus_l * self._u(0.18, 0.28)
        wing_t     = fus_r  * self._u(0.25, 0.40)

        stab_span  = wing_span * self._u(0.30, 0.45)
        stab_chord = wing_chord * self._u(0.50, 0.70)
        stab_t     = wing_t

        fin_h  = fus_r * self._u(2.5, 4.0)
        fin_w  = wing_chord * self._u(0.55, 0.75)
        fin_t  = wing_t

        # Fuselage along X axis, centred at origin
        fuselage = self._cylinder(fus_r, fus_l, sections=24)
        # trimesh cylinder is along Z; rotate to X
        Rx = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        fuselage.apply_transform(Rx)

        # Main wings (flat box spanning Y axis) at centre
        wings = self._box([wing_chord, wing_span, wing_t])

        # Horizontal stabilisers at rear
        rear_x = -(fus_l / 2 - stab_chord / 2)
        stab   = self._box([stab_chord, stab_span, stab_t],
                            self._translate(rear_x, 0, 0))

        # Vertical fin at rear, pointing upward
        fin = self._box([fin_w, fin_t, fin_h],
                         self._translate(rear_x, 0, fin_h / 2))

        return self._combine([fuselage, wings, stab, fin])
