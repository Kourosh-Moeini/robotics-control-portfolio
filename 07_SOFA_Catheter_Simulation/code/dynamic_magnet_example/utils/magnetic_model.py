"""Furlani cuboid field and configuration-dependent tip-magnet wrench.

The field expression is the eight-corner closed form for a uniformly
z-magnetized rectangular prism.  It is ported from the target MATLAB
``Matlab/force_torque.m``.  The analytical spatial Jacobian is used
for ``F = grad(m.B)`` and is checked against finite differences in the tests.

Conventions
-----------
* World vectors are column-vector equivalents represented as NumPy ``(3,)``.
* SOFA rigid quaternions are ``[x, y, z, w]``.
* External trajectory quaternions are MATLAB ``[w, x, y, z]``.
* ``gradient[i,j] = d B_i / d x_j``; hence ``F = gradient.T @ m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MagneticState:
    force: np.ndarray
    torque: np.ndarray
    field: np.ndarray
    gradient: np.ndarray
    catheter_moment: np.ndarray
    tip_position: np.ndarray
    external_position: np.ndarray

    @property
    def wrench(self) -> np.ndarray:
        return np.concatenate([self.force, self.torque])


def rotation_matrix_xyzw(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.all(np.isfinite(q)) or n < 1e-15:
        raise ValueError("finite nonzero SOFA quaternion required")
    x, y, z, w = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def rotation_matrix_wxyz(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    return rotation_matrix_xyzw([q[1], q[2], q[3], q[0]])


def cuboid_field_and_gradient(point_local, dimensions, remanence_z):
    """Return local ``(B, dB/dx)`` outside a z-magnetized cuboid.

    ``dimensions`` contains full x/y/z side lengths.  The point and dimensions
    use metres and the returned field/gradient use tesla and tesla/metre.
    """
    point = np.asarray(point_local, dtype=float).reshape(3)
    dim = np.asarray(dimensions, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(dim)) or np.any(dim <= 0.0):
        raise ValueError("finite point and finite positive cuboid dimensions required")
    if not np.isfinite(remanence_z):
        raise ValueError("finite remanence required")
    half = 0.5 * dim
    if np.all(np.abs(point) <= half + 1e-12):
        raise ValueError("catheter magnet centre lies inside the external magnet")

    x, y, z = point
    U = (x - half[0], x + half[0])
    V = (y - half[1], y + half[1])
    W = (z - half[2], z + half[2])
    bx = by = bz = 0.0
    jxx = jxy = jxz = jyy = jyz = 0.0
    tiny = np.finfo(float).tiny

    # Use one-based loop counters so the sign exactly matches MATLAB.
    for ii, u in enumerate(U, start=1):
        for jj, v in enumerate(V, start=1):
            for kk, wv in enumerate(W, start=1):
                radius = float(np.sqrt(u*u + v*v + wv*wv))
                if radius < 1e-12:
                    raise ValueError("field is singular at an external-magnet corner")
                sign = -1.0 if (ii + jj + kk) % 2 else 1.0
                vr = max(v + radius, tiny)
                ur = max(u + radius, tiny)

                bx += sign * np.log(vr)
                by += sign * np.log(ur)
                bz -= sign * np.arctan2(u * v, wv * radius)

                jxx += sign * u / (radius * vr)
                jyy += sign * v / (radius * ur)
                jxy += sign / radius
                jxz += sign * wv / (radius * vr)
                jyz += sign * wv / (radius * ur)

    coefficient = float(remanence_z) / (4.0 * np.pi)
    field = coefficient * np.array([bx, by, bz])
    jzz = -(jxx + jyy)
    gradient = coefficient * np.array([
        [jxx, jxy, jxz],
        [jxy, jyy, jyz],
        [jxz, jyz, jzz],
    ])
    return field, gradient


def catheter_moment_world(tip_quaternion_xyzw, magnets) -> np.ndarray:
    """Rotate the catheter moment using BeamAdapter's local-x convention."""
    return rotation_matrix_xyzw(tip_quaternion_xyzw) @ magnets.m_c_sofa_local


def magnetic_wrench(tip_position, tip_quaternion_xyzw,
                    external_position, external_quaternion_wxyz,
                    magnets) -> MagneticState:
    """Evaluate Furlani field and the dipole force/torque in world axes."""
    p_tip = np.asarray(tip_position, dtype=float).reshape(3)
    p_ext = np.asarray(external_position, dtype=float).reshape(3)
    r_ext = rotation_matrix_wxyz(external_quaternion_wxyz)
    r_tip = rotation_matrix_xyzw(tip_quaternion_xyzw)

    point_local = r_ext.T @ (p_tip - p_ext)
    b_local, g_local = cuboid_field_and_gradient(
        point_local, magnets.external_dimensions, magnets.remanence_z
    )
    field_world = r_ext @ b_local
    gradient_world = r_ext @ g_local @ r_ext.T
    moment_world = r_tip @ magnets.m_c_sofa_local

    force_world = gradient_world.T @ moment_world
    torque_world = np.cross(moment_world, field_world)
    return MagneticState(
        force=force_world,
        torque=torque_world,
        field=field_world,
        gradient=gradient_world,
        catheter_moment=moment_world,
        tip_position=p_tip,
        external_position=p_ext,
    )


def rotation_angle_xyzw(q0, q1) -> float:
    """Shortest rotation angle between two SOFA quaternions."""
    a = np.asarray(q0, float).reshape(4).copy()
    b = np.asarray(q1, float).reshape(4).copy()
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))
