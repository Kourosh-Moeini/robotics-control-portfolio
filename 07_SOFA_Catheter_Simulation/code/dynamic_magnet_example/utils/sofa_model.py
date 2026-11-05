"""BeamAdapter FEM catheter with inertia and prescribed external cuboid pose.

Each animate() call is a physical backward-Euler step, not static relaxation.
The magnetic wrench and tip-cylinder world inertia are updated from the current
configuration before each step. A second CFF supplies the tip gyroscopic term;
UniformMass itself provides gravity but no gyroscopic torque.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

SOFA_ROOT_DEFAULT = os.environ.get("SOFA_ROOT", "")

BEAM_PLUGINS = [
    "BeamAdapter",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mass",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Visual",
]

_S = math.sqrt(0.5)
# BeamAdapter local x -> world +z. SOFA quaternion ordering is [x,y,z,w].
BASE_QUAT = [0.0, -_S, 0.0, _S]
_DLL_HANDLES = []


def setup_runtime(sofa_root: str | None = None) -> str:
    sofa_root = sofa_root or os.environ.get("SOFA_ROOT", SOFA_ROOT_DEFAULT)
    if not os.path.isdir(sofa_root):
        raise RuntimeError(f"SOFA_ROOT does not exist: {sofa_root}")
    os.environ["SOFA_ROOT"] = sofa_root
    site = os.path.join(sofa_root, "plugins", "SofaPython3",
                        "lib", "python3", "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)
    if hasattr(os, "add_dll_directory"):
        for directory in (
            os.path.join(sofa_root, "bin"),
            os.path.join(sofa_root, "plugins", "SofaPython3", "bin"),
            os.path.join(sofa_root, "plugins", "BeamAdapter", "bin"),
        ):
            if os.path.isdir(directory):
                _DLL_HANDLES.append(os.add_dll_directory(directory))
    return sofa_root


setup_runtime()

import Sofa  # noqa: E402
import Sofa.Core  # noqa: E402
import Sofa.Simulation  # noqa: E402
import SofaRuntime  # noqa: F401,E402

from . import magnetic_model  # noqa: E402


def straight_rod_nodes(length: float, n_elem: int) -> list:
    return [[0.0, 0.0, i * length / n_elem] + BASE_QUAT
            for i in range(n_elem + 1)]


def build_scene(params, magnets, root=None):
    p = params
    root = root if root is not None else Sofa.Core.Node("root")
    root.gravity = [float(g) for g in p.gravity]
    root.dt = p.integration_dt
    root.addObject("RequiredPlugin", pluginName=BEAM_PLUGINS)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle",
                   displayFlags="showBehaviorModels showForceFields")

    beam = root.addChild("catheter")
    beam.addObject("EulerImplicitSolver", name="odesolver",
                   rayleighStiffness=p.rayleigh_stiffness,
                   rayleighMass=p.rayleigh_mass)
    beam.addObject("BTDLinearSolver", name="linearSolver", verbose=False)

    dofs = beam.addObject(
        "MechanicalObject", template="Rigid3d", name="DOFs",
        position=straight_rod_nodes(p.s_length, p.n_elem),
        showObject=True, showObjectScale=0.004,
    )
    beam.addObject("MeshTopology", name="lines",
                   lines=[[i, i + 1] for i in range(p.n_elem)])
    beam.addObject("FixedProjectiveConstraint", name="clamp", indices=[0])
    beam.addObject(
        "BeamInterpolation", name="interpolation",
        crossSectionShape="circular", radius=[p.r], innerRadius=[p.r0],
        defaultYoungModulus=[p.E], defaultPoissonRatio=[p.nu],
        straight=True, dofsAndBeamsAligned=True,
    )
    beam.addObject(
        "AdaptiveBeamForceFieldAndMass", name="beamForceField",
        interpolation="@interpolation", computeMass=True,
        massDensity=[p.rho], shearStressComputation=True,
    )

    tip_index = p.n_elem
    cff = beam.addObject(
        "ConstantForceField", name="tipMagneticWrench",
        indices=[tip_index], forces=[[0.0] * 6], showArrowSize=0.5,
    )

    tip_mass = None
    gyro_cff = None
    if p.include_tip_magnet_mass:
        r_tip = magnetic_model.rotation_matrix_xyzw(BASE_QUAT)
        inertia_world = r_tip @ magnets.inertia_c_sofa_local @ r_tip.T
        # SOFA RigidMass.inertiaMatrix is inertia PER UNIT MASS, not kg m^2.
        # UniformMass uses this tensor directly in world-axis derivatives.
        # SOFA permits one BaseMass per Node. Keep the extra tip mass in a
        # child, explicitly linked to the same beam DOFs (no extra DOFs/mapping).
        tip_mass_node = beam.addChild("tipMagnet")
        tip_mass = tip_mass_node.addObject(
            "UniformMass", template="Rigid3d", name="tipMagnetMass",
            mstate="@../DOFs", indices=[tip_index],
            vertexMass=[magnets.mass_c, magnets.V_c,
                        (inertia_world / magnets.mass_c).reshape(-1).tolist()],
        )
        gyro_cff = beam.addObject(
            "ConstantForceField", name="tipMagnetGyroscopicTorque",
            indices=[tip_index], forces=[[0.0] * 6],
        )

    # Prescribed kinematic actor. It has no mass or solver because its pose is
    # an input to the dynamic problem, but keeping it in the SOFA scene makes
    # the two-body structure explicit and available to interactive viewers.
    external = root.addChild("externalMagnet")
    external_dofs = external.addObject(
        "MechanicalObject", template="Rigid3d", name="DOFs",
        position=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        showObject=True, showObjectScale=0.02,
    )
    return {
        "root": root,
        "beam": beam,
        "dofs": dofs,
        "cff": cff,
        "tip_mass": tip_mass,
        "gyro_cff": gyro_cff,
        "magnets": magnets,
        "external": external,
        "external_dofs": external_dofs,
        "tip_index": tip_index,
        "params": p,
    }


def set_tip_wrench(handles, wrench) -> None:
    value = np.asarray(wrench, dtype=float).reshape(6)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"non-finite magnetic wrench: {value}")
    handles["cff"].forces.value = [value.tolist()]


def set_external_pose(handles, position, quaternion_wxyz) -> None:
    """Set the prescribed external-magnet Rigid3 pose in the SOFA scene."""
    p = np.asarray(position, dtype=float).reshape(3)
    q = np.asarray(quaternion_wxyz, dtype=float).reshape(4).copy()
    q /= np.linalg.norm(q)
    # MATLAB [w,x,y,z] -> SOFA [x,y,z,w]
    pose = np.concatenate([p, q[1:], q[:1]])
    handles["external_dofs"].position.value = [pose.tolist()]


def tip_pose(handles) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(
        handles["dofs"].position.value[handles["tip_index"]], dtype=float
    )
    return pose[:3].copy(), pose[3:7].copy()


def backbone(handles) -> np.ndarray:
    return np.asarray(handles["dofs"].position.value, dtype=float)[:, :3].copy()


def node_quaternions(handles) -> np.ndarray:
    return np.asarray(handles["dofs"].position.value, dtype=float)[:, 3:7].copy()


def node_velocities(handles) -> np.ndarray:
    """Rigid derivatives in WORLD axes: [vx,vy,vz,wx,wy,wz]."""
    return np.asarray(handles["dofs"].velocity.value, dtype=float).copy()


def tip_inertia_world(quaternion_xyzw, magnets):
    r = magnetic_model.rotation_matrix_xyzw(quaternion_xyzw)
    return r @ magnets.inertia_c_sofa_local @ r.T


def update_tip_inertia(handles):
    """Rotate anisotropic inertia and supply -omega x (I_world omega).

    These are evaluated at the beginning of the physical substep, just like
    the follower magnetic wrench; reducing dt reduces their time-lag error.
    """
    if handles["tip_mass"] is None:
        return
    _, quat = tip_pose(handles)
    magnets = handles["magnets"]
    inertia = tip_inertia_world(quat, magnets)
    handles["tip_mass"].vertexMass.value = [
        magnets.mass_c, magnets.V_c,
        (inertia / magnets.mass_c).reshape(-1).tolist(),
    ]
    omega = node_velocities(handles)[handles["tip_index"], 3:]
    gyro = -np.cross(omega, inertia @ omega)
    handles["gyro_cff"].forces.value = [np.concatenate([np.zeros(3), gyro]).tolist()]
