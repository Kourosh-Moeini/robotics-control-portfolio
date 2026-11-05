"""External cuboid CENTRE poses in physical time, adapted from MATLAB mode.m.

Quaternions use [w,x,y,z]. Default orientation is MATLAB trajectory.m's
[0,1,0,0] (180 degrees about x), NOT the static notebook's identity quaternion.
"""

from __future__ import annotations

import numpy as np

SHAPES = ("Step", "Spiral", "Inf", "Line", "Sine", "Squarewave", "Triangle",
          "Square", "SpiralTilted")


def build(shape, n_steps, freq=0.20, amp=0.10, altitude=0.155, dt=0.01,
          step_position=None, external_quaternion_wxyz=(0.0, 1.0, 0.0, 0.0),
          tilt_deg=1.0):
    key = str(shape).replace("_", "").replace(" ", "").lower()
    if key not in {s.lower() for s in SHAPES}:
        raise ValueError(f"shape must be in {SHAPES}")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not np.all(np.isfinite([dt, freq, amp, altitude, tilt_deg])) or dt <= 0 or freq < 0:
        raise ValueError("finite path settings, dt > 0 and freq >= 0 required")
    t = np.arange(int(n_steps) + 1, dtype=float) * dt
    progress = t / t[-1]
    angle = 2 * np.pi * freq * t
    z = np.full_like(t, altitude)
    if key == "step":
        position = (amp, amp, altitude) if step_position is None else step_position
        p_ext = np.tile(np.asarray(position, float).reshape(3), (t.size, 1))
    else:
        if key in ("spiral", "spiraltilted"):
            x, y = amp * progress * np.cos(2*angle), amp * progress * np.sin(2*angle)
            if key == "spiraltilted":
                theta = np.deg2rad(tilt_deg)
                y0 = y - 0.20
                y, z = y0 * np.cos(theta), altitude + y0 * np.sin(theta)
        elif key == "line":
            x, y = amp * progress, np.zeros_like(t)
        elif key == "inf":
            x, y = amp * np.sin(2*angle), amp * np.sin(angle)
        elif key == "sine":
            x, y = amp * np.sin(angle), amp * progress
        elif key == "squarewave":
            square = np.where(np.mod(freq*t, 1.0) < 0.5, 1.0, -1.0)
            x, y = -amp/4 + amp*square/4, amp * progress
        elif key == "triangle":
            x = amp * (1.0 - 4.0 * np.abs(np.mod(freq*t, 1.0) - 0.5))
            y = amp * progress
        else:
            # One closed square, starting at (0,0), over the simulation duration.
            u = np.minimum(4.0 * progress, 4.0)
            x = amp * np.where(u < 1, u, np.where(u < 2, 1, np.where(u < 3, 3-u, 0)))
            y = amp * np.where(u < 1, 0, np.where(u < 2, u-1, np.where(u < 3, 1, 4-u)))
        p_ext = np.column_stack([x, y, z])
    q = np.asarray(external_quaternion_wxyz, float).reshape(4).copy()
    norm = np.linalg.norm(q)
    if not np.all(np.isfinite(q)) or norm < 1e-15 or not np.all(np.isfinite(p_ext)):
        raise ValueError("finite positions and nonzero finite external quaternion required")
    q_ext = np.tile(q / norm, (t.size, 1))
    return p_ext, q_ext


def interpolate_pose(p0, q0, p1, q1, fraction):
    """Linear position / shortest-arc quaternion SLERP for integration substeps.

    Inputs should be normalized quaternions; run() validates/normalizes them.
    Discontinuous prescribed jumps occur at the sample time: use substeps=1
    or supply a finer explicitly sampled path if you need true step motion.
    """
    a, b = np.asarray(q0, float).copy(), np.asarray(q1, float).copy()
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = a + fraction * (b-a)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1-fraction)*angle)*a + np.sin(fraction*angle)*b) / np.sin(angle)
    q /= np.linalg.norm(q)
    return (1-fraction)*np.asarray(p0) + fraction*np.asarray(p1), q


def describe(p_ext, q_ext, dt=0.01):
    p, q = np.asarray(p_ext), np.asarray(q_ext)
    return "\n".join([
        f"external-magnet path: {len(p)} samples, t=0 .. {(len(p)-1)*dt:g} s",
        *(f"  {axis}: {p[:, i].min():+.4f} .. {p[:, i].max():+.4f} m"
          for i, axis in enumerate("xyz")),
        f"  first quaternion [w,x,y,z] = {tuple(float(x) for x in q[0].round(6))}",
    ])
