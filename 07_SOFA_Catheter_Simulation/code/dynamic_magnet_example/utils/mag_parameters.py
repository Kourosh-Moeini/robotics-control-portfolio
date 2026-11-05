"""Magnet data ported from the target MATLAB ``mag_parameters.m``.

The target Furlani routine models the external actuator as a uniformly
z-magnetized cuboid of dimensions 50 x 60 x 60 mm.  Set3 is the internally
consistent/default case used by the MATLAB simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAG_SETS = ("Set1", "Set2", "Set3")


@dataclass
class Magnets:
    mag_set: str
    r_c: float
    l_c: float
    M_c: float
    density_c: float
    M_e: float
    external_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.06, 0.06], dtype=float)
    )
    external_moment_volume: float = 0.0
    mu0: float = 4.0 * math.pi * 1e-7

    @property
    def V_c(self) -> float:
        return math.pi * self.r_c**2 * self.l_c

    @property
    def V_e_field(self) -> float:
        return float(np.prod(self.external_dimensions))

    @property
    def m_c_matlab_local(self) -> np.ndarray:
        """Catheter moment in the MATLAB material frame (local -z)."""
        return np.array([0.0, 0.0, -self.V_c * self.M_c])

    @property
    def m_c_sofa_local(self) -> np.ndarray:
        """Same physical moment in BeamAdapter coordinates (local -x)."""
        return np.array([-self.V_c * self.M_c, 0.0, 0.0])

    @property
    def m_e_local(self) -> np.ndarray:
        """External magnet moment in its own frame (local -z), as in MATLAB."""
        return np.array([0.0, 0.0, -self.external_moment_volume * self.M_e])

    @property
    def mass_c(self) -> float:
        return self.density_c * self.V_c

    @property
    def inertia_c_sofa_local(self) -> np.ndarray:
        """Solid-cylinder inertia [kg m^2], centred at the final beam node.

        The cylinder axis is BeamAdapter's local x (MATLAB material z).
        This physical inertia is additional to the catheter beam's own mass.
        """
        axial = self.mass_c * self.r_c**2 / 2.0
        transverse = self.mass_c * (3*self.r_c**2 + self.l_c**2) / 12.0
        return np.diag([axial, transverse, transverse])

    @property
    def remanence_z(self) -> float:
        """Signed local remanence used by the target MATLAB Furlani model."""
        return self.mu0 * self.m_e_local[2] / self.V_e_field

    def summary(self) -> str:
        d = self.external_dimensions * 1e3
        return "\n".join([
            f"magnets ({self.mag_set})",
            f"  catheter: r={self.r_c*1e3:.4f} mm, L={self.l_c*1e3:.4f} mm, "
            f"|m|={np.linalg.norm(self.m_c_matlab_local):.6g} A.m^2, "
            f"mass={self.mass_c*1e3:.4f} g",
            f"  external cuboid: {d[0]:.1f} x {d[1]:.1f} x {d[2]:.1f} mm, "
            f"M={self.M_e:.6g} A/m, Br_z={self.remanence_z:.6g} T",
        ])


def build(mag_set: str = "Set3") -> Magnets:
    if mag_set == "Set1":
        rc, lc, mc, density = 0.001, 0.005, 8.0e6, 7500.0
        me, moment_volume = 8.0e6, math.pi * 0.01**2 * 0.02
    elif mag_set == "Set2":
        rc, lc, mc = 0.0024650, 0.00855, 6.75e5
        density = 1.1e-3 / (lc * math.pi * rc**2)
        me, moment_volume = 6.8e5, math.pi * 0.025**2 * 0.05
    elif mag_set == "Set3":
        rc, lc, mc = 0.0024650, 0.00855, 6.62549e5
        density = 1.1e-3 / (lc * math.pi * rc**2)
        me, moment_volume = 1.055130e6, 0.06 * 0.05 * 0.06
    else:
        raise ValueError(f"unknown magnet set {mag_set!r}; expected {MAG_SETS}")

    return Magnets(
        mag_set=mag_set,
        r_c=rc,
        l_c=lc,
        M_c=mc,
        density_c=density,
        M_e=me,
        external_moment_volume=moment_volume,
    )
