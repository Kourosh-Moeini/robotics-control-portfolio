"""Configuration-driven magnetic dynamics, with one step per physical substep.

Solve M dv/dt + C v + f_internal(q) = f_magnetic(q,t) + f_gravity.
Beam mechanics are implicit; the magnetic follower load is sampled at the
beginning of each substep (partitioned, first-order coupling). There is no
equilibrium loop, no under-relaxation and no hidden load ramp. The straight,
stationary initial catheter is NOT assumed to be in magnetic equilibrium.
"""

from __future__ import annotations

import numpy as np

# Import in this order so sofa_model configures the DLL/search paths first.
from . import magnetic_model, sofa_model, trajectory

import Sofa.Simulation  # noqa: E402


def _evaluate(handles, magnets, p_ext, q_ext, time):
    position, quaternion = sofa_model.tip_pose(handles)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise FloatingPointError(f"non-finite SOFA tip state at t={time:.9g} s; reduce integration dt")
    try:
        state = magnetic_model.magnetic_wrench(position, quaternion, p_ext, q_ext, magnets)
    except ValueError as exc:
        r_ext = magnetic_model.rotation_matrix_wxyz(q_ext)
        local = r_ext.T @ (position - p_ext)
        raise RuntimeError(
            f"invalid magnetic geometry at t={time:.9g} s: {exc}; "
            f"tip world={position.tolist()} m; external centre={np.asarray(p_ext).tolist()} m; "
            f"tip in external frame={local.tolist()} m. No magnet contact is modeled."
        ) from exc
    if not np.all(np.isfinite(state.wrench)):
        raise FloatingPointError(f"non-finite Furlani wrench at t={time:.9g} s")
    wrench = state.wrench
    if not handles["params"].include_tip_magnet_mass:
        # MATLAB's tip boundary includes mg but not tip magnet acceleration.
        wrench[:3] += magnets.mass_c * handles["params"].gravity
    # When tip mass is enabled, UniformMass adds its gravity; do NOT add it twice.
    return state, wrench


def run(params, magnets, external_position, external_quaternion_wxyz, verbose_every=100):
    p = params
    p_ext = np.asarray(external_position, dtype=float).copy()
    q_ext = np.asarray(external_quaternion_wxyz, dtype=float).copy()
    expected = p.steps + 1
    if p_ext.shape != (expected, 3) or q_ext.shape != (expected, 4):
        raise ValueError(f"require {expected} physical samples: positions ({expected},3), quaternions ({expected},4)")
    norms = np.linalg.norm(q_ext, axis=1)
    if not np.all(np.isfinite(p_ext)) or not np.all(np.isfinite(q_ext)) or np.any(norms < 1e-15):
        raise ValueError("finite positions and finite nonzero external quaternions required")
    q_ext /= norms[:, None]

    handles = sofa_model.build_scene(p, magnets)
    root = handles["root"]
    frames = np.empty((expected, p.n_elem + 1, 3))
    node_quats = np.empty((expected, p.n_elem + 1, 4))
    node_velocity = np.empty((expected, p.n_elem + 1, 6))
    force, torque, field = (np.empty((expected, 3)) for _ in range(3))
    gradient = np.empty((expected, 3, 3))
    applied = np.empty((expected, 6))
    try:
        Sofa.Simulation.init(root)
        print(f"  nodes={p.n_elem + 1}, physical steps={p.steps}, substeps={p.substeps}")
        initial, _ = sofa_model.tip_pose(handles)
        print(f"  initial tip = {tuple(float(x) for x in (initial*1e3).round(4))} mm")

        # Record states at t_k, including the actual undeformed t=0 and final t=T.
        # The last wrench is re-evaluated at T, never copied from the prior step.
        for k, time in enumerate(p.time):
            sofa_model.set_external_pose(handles, p_ext[k], q_ext[k])
            state, wrench = _evaluate(handles, magnets, p_ext[k], q_ext[k], time)
            frames[k] = sofa_model.backbone(handles)
            node_quats[k] = sofa_model.node_quaternions(handles)
            node_velocity[k] = sofa_model.node_velocities(handles)
            force[k], torque[k], field[k], gradient[k] = state.force, state.torque, state.field, state.gradient
            applied[k] = wrench
            if not np.all(np.isfinite(frames[k])) or not np.all(np.isfinite(node_velocity[k])):
                raise FloatingPointError(f"non-finite SOFA state at t={time:g} s")
            if verbose_every and (k == 0 or k % verbose_every == 0 or k == p.steps):
                tip = frames[k, -1]*1e3
                speed = np.linalg.norm(node_velocity[k, -1, :3])*1e3
                print(f"  step {k:5d}/{p.steps}  t={time:7.3f} s  "
                      f"tip=({tip[0]:8.3f},{tip[1]:8.3f},{tip[2]:8.3f}) mm  "
                      f"|Fmag|={np.linalg.norm(force[k]):.5e} N  |vtip|={speed:.4f} mm/s")
            if k == p.steps:
                break
            for substep in range(p.substeps):
                if substep:
                    position_e, quaternion_e = trajectory.interpolate_pose(
                        p_ext[k], q_ext[k], p_ext[k+1], q_ext[k+1], substep/p.substeps)
                    sofa_model.set_external_pose(handles, position_e, quaternion_e)
                    _, wrench = _evaluate(handles, magnets, position_e, quaternion_e,
                                          time + substep*p.integration_dt)
                sofa_model.update_tip_inertia(handles)
                sofa_model.set_tip_wrench(handles, wrench)
                Sofa.Simulation.animate(root, p.integration_dt)
    finally:
        # Allows repeated notebook runs without keeping live scenes/resources.
        Sofa.Simulation.unload(root)

    tip_quat = node_quats[:, -1].copy()
    # MATLAB material frame is R_sofa * Ry(+90). q_sofa is NOT just reordered.
    material_q = np.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    xyz, scalar = tip_quat[:, :3], tip_quat[:, 3]
    converted_xyz = (scalar[:, None]*material_q[:3] + material_q[3]*xyz
                     + np.cross(xyz, material_q[:3]))
    converted_w = scalar*material_q[3] - xyz @ material_q[:3]
    ang = np.column_stack([converted_w, converted_xyz])
    backbone = frames[-1].copy()
    return {
        "params": p, "magnets": magnets, "time": p.time,
        "step_index": np.arange(expected), "steps": p.steps,
        "external_position": p_ext, "external_quaternion_wxyz": q_ext,
        "p_total": frames, "node_quaternion_xyzw": node_quats,
        "node_velocity": node_velocity, "pos": frames[:, -1].copy(),
        "tip_quaternion_xyzw": tip_quat, "ang": ang,
        "tip_velocity": node_velocity[:, -1, :3].copy(),
        "tip_angular_velocity": node_velocity[:, -1, 3:].copy(),
        "force": force, "torque": torque, "field": field, "gradient": gradient,
        "wrench": applied, "backbone": backbone, "tip": backbone[-1].copy(),
        "arc_length": float(np.linalg.norm(np.diff(backbone, axis=0), axis=1).sum()),
        "integration_steps": p.steps*p.substeps,
    }


def report(result):
    tip = result["tip"]*1e3
    print("--- dynamic magnetic SOFA result")
    print(f"    final time = {result['time'][-1]:g} s; physical integration steps = {result['integration_steps']}")
    print(f"    final tip = ({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}) mm")
    print(f"    final |Fmag| = {np.linalg.norm(result['force'][-1]):.6e} N")
    print(f"    final |Tmag| = {np.linalg.norm(result['torque'][-1]):.6e} N.m")
    print(f"    final |B| = {np.linalg.norm(result['field'][-1]):.6e} T")
    print(f"    final tip speed = {np.linalg.norm(result['tip_velocity'][-1])*1e3:.6g} mm/s")
    print(f"    arc length = {result['arc_length']*1e3:.4f} mm")
    print("    No static-equilibrium iterations: motion is the physical transient.")


def save(result, filename):
    """Save numerical histories and complete run settings (no pickled objects)."""
    import json
    import os
    from dataclasses import asdict

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    settings = asdict(result["params"])
    settings["gravity"] = settings["gravity"].tolist()
    data = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(filename, **data, params_json=json.dumps(settings),
                        params_summary=result["params"].summary(),
                        magnet_summary=result["magnets"].summary(),
                        integration_steps=result["integration_steps"])
    print("  saved data  :", filename)
