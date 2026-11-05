"""Dynamic magnetic catheter entry point; mode.ipynb provides the same workflow.

Run from this folder with the Environment_Sofa Python kernel/runtime.
"""

from __future__ import annotations

import time

from utils import mag_parameters, main_code, parameters, paths, trajectory, visualization

# ----------------------------- settings --------------------------------------
MATTER = "PDMS2"
FLAG = "DynamicMagnet"
MATERIAL_RECIPE = "Matlab"     # Matlab | Trial3 | StaticMagnet (see README)
MAG_SET = "Set3"
SHAPE = "Step"                # Step | Spiral | Inf | Line | Sine | ...
FREQ = 0.20
AMP = 0.10                    # external-magnet path amplitude [m]
ALTITUDE = 0.155              # Matlab/mode.m: L + 0.05 [m], for PDMS2
STEP_POSITION = (0.10, 0.10, 0.155)
EXTERNAL_QUATERNION_WXYZ = (0.0, 1.0, 0.0, 0.0)  # Matlab/trajectory.m

STEPS = 500                   # physical output intervals, NOT static poses
DT = 0.01                     # output interval [s]; duration = STEPS * DT
SUBSTEPS = 8                  # mechanics + magnetic wrench every DT/8 = 0.00125 s
N_ELEM = None                 # None -> material's MATLAB num_points
NUMERICAL_OVERRIDES = {}      # e.g. {"rayleigh_stiffness": 0.01}
SAVE_DATA = True
MAKE_PLOT = True
MAKE_VIDEO = False            # notebook cell 3 explicitly creates video
VIDEO_STRIDE = 2
VIDEO_FPS = 50                # at default DT/stride, roughly real-time playback


def main():
    started = time.perf_counter()
    overrides = dict(NUMERICAL_OVERRIDES)
    overrides.update(flag=FLAG, shape=SHAPE, mag_set=MAG_SET, freq=FREQ,
                     amp=AMP, steps=STEPS, dt=DT, substeps=SUBSTEPS)
    params = parameters.build(MATTER, n_elem=N_ELEM,
                              recipe=MATERIAL_RECIPE, **overrides)
    magnets = mag_parameters.build(MAG_SET)
    p_ext, q_ext = trajectory.build(
        SHAPE, params.steps, FREQ, AMP, altitude=ALTITUDE, dt=params.dt,
        step_position=STEP_POSITION, external_quaternion_wxyz=EXTERNAL_QUATERNION_WXYZ,
    )
    print("=" * 78)
    print("DYNAMIC MAGNETIC SOFA CATHETER - configuration-driven coupling")
    print("=" * 78)
    print(params.summary())
    print(magnets.summary())
    print(trajectory.describe(p_ext, q_ext, params.dt))
    result = main_code.run(params, magnets, p_ext, q_ext,
                           verbose_every=max(1, params.steps // 5))
    print()
    main_code.report(result)
    if SAVE_DATA:
        main_code.save(result, paths.dataset(params))
    if MAKE_PLOT:
        visualization.plot_tip_and_wrench(result, paths.plot(params))
    if MAKE_VIDEO:
        visualization.animate_3d(result, paths.video(params),
                                 stride=VIDEO_STRIDE, fps=VIDEO_FPS)
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    return result


if __name__ == "__main__":
    main()
