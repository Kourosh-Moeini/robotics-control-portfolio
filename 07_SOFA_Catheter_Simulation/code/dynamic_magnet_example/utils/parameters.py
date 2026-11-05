"""Dynamic material recipes from this folder's MATLAB code; all units are SI.

Default recipe: ``Matlab/parameters.m`` and ``Matlab/mode.m``.
``Trial3`` selects the alternate ``Trial 3-mex-bvp4c/parameters.m`` values.
BeamAdapter is a Kirchhoff beam, not the full shearable MATLAB Cosserat rod.
Its isotropic G is derived from E and nu. Rayleigh stiffness damping applies
one coefficient to every beam stiffness mode; unequal Bse/Bbt are approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

MATTERS = ("Rod1", "Rod2", "PDMS1", "PDMS2")
RECIPES = ("Matlab", "Trial3", "StaticMagnet")


@dataclass
class Params:
    matter: str = "PDMS2"
    recipe: str = "Matlab"
    flag: str = "DynamicMagnet"
    shape: str = "Step"
    mag_set: str = "Set3"
    freq: float = 0.20
    amp: float = 0.10
    rho: float = 1030.0
    r: float = 0.003385
    r0: float = 0.002465
    E: float = 1.888567124751886e6
    nu: float = 0.422429111252843
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    s_length: float = 0.105
    num_points: int = 50
    n_elem: int = 50
    steps: int = 500
    dt: float = 0.01                 # physical OUTPUT interval [s]
    substeps: int = 8                # physical integration steps per interval
    Bse_coef: float = 0.1            # MATLAB constitutive damping reference
    Bbt_coef: float = 0.1
    rayleigh_stiffness: float = 0.1  # beta [s]; C = beta K + alpha M
    rayleigh_mass: float = 0.0       # alpha [1/s]; no extra mass damping by default
    include_tip_magnet_mass: bool = True

    @property
    def A(self):
        return math.pi * (self.r**2 - self.r0**2)

    @property
    def I(self):
        return math.pi * (self.r**4 - self.r0**4) / 4.0

    @property
    def Jp(self):
        return 2.0 * self.I

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def mass(self):
        return self.rho * self.A * self.s_length

    @property
    def time(self):
        return np.arange(self.steps + 1, dtype=float) * self.dt

    @property
    def t_length(self):
        return self.steps * self.dt

    @property
    def integration_dt(self):
        return self.dt / self.substeps

    def summary(self):
        lines = [
            f"{self.matter} ({self.flag} / {self.shape} / {self.mag_set}, recipe={self.recipe})",
            f"  L={self.s_length*1e3:.3f} mm, r={self.r*1e3:.4f} mm, "
            f"r0={self.r0*1e3:.4f} mm, elements={self.n_elem}",
            f"  E={self.E:.6g} Pa, nu={self.nu:.6g}, G={self.G:.6g} Pa, "
            f"rho={self.rho:.3f} kg/m^3, catheter mass={self.mass*1e3:.4f} g",
            f"  EI={self.E*self.I:.6e} N.m^2, EA={self.E*self.A:.6e} N",
            f"  gravity={tuple(float(x) for x in self.gravity)} m/s^2",
            f"  physical time: {self.steps} x {self.dt:g} s = {self.t_length:g} s; "
            f"substeps={self.substeps}, integration dt={self.integration_dt:g} s",
            f"  Rayleigh damping: beta={self.rayleigh_stiffness:g} s, "
            f"alpha={self.rayleigh_mass:g} 1/s",
            f"  tip magnet mass/inertia included: {self.include_tip_magnet_mass}",
            "  coupling: current tip pose -> Furlani wrench -> ONE physical step",
        ]
        if self.Bse_coef != self.Bbt_coef:
            lines.append("  NOTE: unequal MATLAB Bse/Bbt damping cannot be reproduced "
                         "exactly by a single Rayleigh stiffness coefficient.")
        if self.matter == "Rod1":
            lines.append("  NOTE: MATLAB quadratic drag is not implemented by this beam scene.")
        return "\n".join(lines)


def build(matter="PDMS2", *, n_elem=None, recipe="Matlab", **overrides):
    if matter not in MATTERS or recipe not in RECIPES:
        raise ValueError(f"matter must be in {MATTERS}; recipe must be in {RECIPES}")
    p = Params(matter=matter, recipe=recipe)
    if matter == "Rod1":
        p.rho, p.r, p.r0 = 907.0, 0.005, 0.003
        p.nu, p.E = 0.3, 8.0e6
        p.gravity = np.array([-9.81, 0.0, 0.0])
        p.num_points, p.s_length = 20, 0.03
        p.Bse_coef = p.Bbt_coef = 0.0
    elif matter == "Rod2":
        p.rho, p.r, p.r0 = 8000.0, 0.0012, 0.0
        p.nu, p.E = 0.3, 207.0e9
        p.gravity = np.array([0.0, -9.81, 0.0])
        p.num_points, p.s_length = 50, 0.1
        p.Bse_coef = p.Bbt_coef = 0.0
    elif matter == "PDMS1":
        p.rho, p.r, p.r0 = 970.0, 0.0025, 0.0015
        p.nu, p.E = 0.478000878687799, 3.347591925420653e5
        p.num_points, p.s_length = 50, 0.10669
        p.Bse_coef, p.Bbt_coef = 0.0, 0.1
    elif recipe == "Trial3":
        p.nu, p.E = 0.45, 1.05e6
        p.gravity = np.array([9.81, 0.0, 0.0])
        p.Bse_coef, p.Bbt_coef = 0.1, 0.01
    elif recipe == "StaticMagnet":
        # Optional same elastic constants/gravity as the static magnetic package.
        p.nu, p.E = 0.45, 1.05e6

    valid_fields = {f.name for f in fields(p)}
    for name, value in overrides.items():
        if name not in valid_fields:
            raise AttributeError(f"Params has no field {name!r}")
        setattr(p, name, value)
    if "rayleigh_stiffness" not in overrides:
        p.rayleigh_stiffness = p.Bbt_coef
    p.gravity = np.asarray(p.gravity, dtype=float).reshape(3).copy()
    if n_elem is not None:
        p.n_elem = n_elem
    elif "n_elem" not in overrides:
        p.n_elem = p.num_points
    for name in ("steps", "substeps", "n_elem", "num_points"):
        value = getattr(p, name)
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        setattr(p, name, int(value))
    numeric = [p.rho, p.r, p.r0, p.E, p.nu, p.s_length, p.dt,
               p.Bse_coef, p.Bbt_coef, p.rayleigh_stiffness, p.rayleigh_mass]
    if not np.all(np.isfinite(numeric)) or not np.all(np.isfinite(p.gravity)):
        raise ValueError("parameters must be finite")
    if min(p.rho, p.E, p.s_length, p.dt) <= 0 or not 0 <= p.r0 < p.r:
        raise ValueError("positive density/E/length/dt and 0 <= inner radius < radius required")
    if not -1 < p.nu < 0.5 or min(p.Bse_coef, p.Bbt_coef,
                                p.rayleigh_stiffness, p.rayleigh_mass) < 0:
        raise ValueError("require -1 < nu < 0.5 and nonnegative damping")
    if not isinstance(p.include_tip_magnet_mass, (bool, np.bool_)):
        raise ValueError("include_tip_magnet_mass must be True or False")
    return p
