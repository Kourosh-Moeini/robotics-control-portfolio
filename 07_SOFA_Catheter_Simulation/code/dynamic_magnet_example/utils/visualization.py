"""Plot and animate the coupled catheter/external-magnet simulation."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import magnetic_model


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_tip_and_wrench(result, filename=None, show=False, dpi=140):
    step = result["time"]
    tip = result["pos"] * 1e3
    force = result["force"]
    torque = result["torque"] * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate("xyz"):
        axes[0].plot(step, tip[:, i], label=axis)
        axes[1].plot(step, force[:, i], label=axis)
        axes[2].plot(step, torque[:, i], label=axis)
    axes[0].set_ylabel("tip position [mm]")
    axes[1].set_ylabel("magnetic force [N]")
    axes[2].set_ylabel("magnetic torque [mN m]")
    axes[2].set_xlabel("physical time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.legend(ncol=3)
    fig.suptitle("Configuration-driven dynamic magnetic SOFA simulation")
    fig.tight_layout()
    if filename:
        _ensure_parent(filename)
        fig.savefig(filename, dpi=dpi)
        print("  saved plot  :", filename)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _pole_color(local_z: float, half_length: float) -> tuple:
    """MATLAB red at local -z (N), blue at local +z (S)."""
    blue = float(np.clip(0.5 * (local_z / half_length + 1.0), 0.0, 1.0))
    return (1.0 - blue, 0.0, blue, 0.92)


def _transform_polygons(polygons, rotation, centre):
    r = np.asarray(rotation, float).reshape(3, 3)
    p = np.asarray(centre, float).reshape(3)
    return [np.asarray(poly, float) @ r.T + p for poly in polygons]


def _cylinder_polygons(radius, length, rotation, centre,
                       num_theta=50, num_z=10):
    """Closed cylinder aligned with local z, colored by local z."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_theta + 1)
    z_values = np.linspace(-length / 2.0, length / 2.0, num_z)
    polygons, colors = [], []
    for zi in range(len(z_values) - 1):
        z0, z1 = z_values[zi], z_values[zi + 1]
        for ti in range(num_theta):
            t0, t1 = theta[ti], theta[ti + 1]
            polygons.append(np.array([
                [radius*np.cos(t0), radius*np.sin(t0), z0],
                [radius*np.cos(t1), radius*np.sin(t1), z0],
                [radius*np.cos(t1), radius*np.sin(t1), z1],
                [radius*np.cos(t0), radius*np.sin(t0), z1],
            ]))
            colors.append(_pole_color(0.5 * (z0 + z1), length / 2.0))

    ring = np.column_stack([radius*np.cos(theta[:-1]),
                            radius*np.sin(theta[:-1])])
    polygons.append(np.column_stack([ring[::-1],
                                     np.full(num_theta, -length/2.0)]))
    colors.append(_pole_color(-length/2.0, length/2.0))
    polygons.append(np.column_stack([ring,
                                     np.full(num_theta, length/2.0)]))
    colors.append(_pole_color(length/2.0, length/2.0))
    return _transform_polygons(polygons, rotation, centre), colors


def _cuboid_polygons(dimensions, rotation, centre, num_z=10):
    """Centered cuboid with red/blue pole faces and graded side faces."""
    hx, hy, hz = 0.5 * np.asarray(dimensions, float)
    polygons = [
        np.array([[-hx, -hy, -hz], [hx, -hy, -hz],
                  [hx, hy, -hz], [-hx, hy, -hz]]),
        np.array([[-hx, -hy, hz], [-hx, hy, hz],
                  [hx, hy, hz], [hx, -hy, hz]]),
    ]
    colors = [_pole_color(-hz, hz), _pole_color(hz, hz)]
    z_values = np.linspace(-hz, hz, num_z)
    for z0, z1 in zip(z_values[:-1], z_values[1:]):
        polygons.extend([
            np.array([[-hx, -hy, z0], [hx, -hy, z0],
                      [hx, -hy, z1], [-hx, -hy, z1]]),
            np.array([[hx, -hy, z0], [hx, hy, z0],
                      [hx, hy, z1], [hx, -hy, z1]]),
            np.array([[hx, hy, z0], [-hx, hy, z0],
                      [-hx, hy, z1], [hx, hy, z1]]),
            np.array([[-hx, hy, z0], [-hx, -hy, z0],
                      [-hx, -hy, z1], [-hx, hy, z1]]),
        ])
        colors.extend([_pole_color(0.5 * (z0 + z1), hz)] * 4)
    return _transform_polygons(polygons, rotation, centre), colors


def _set_text_3d(text_artist, position):
    p = np.asarray(position, float)
    text_artist.set_position((p[0], p[1]))
    text_artist.set_3d_properties(p[2], zdir="z")


def animate_3d(result, filename, stride=2, fps=25, dpi=110,
               elev=50, azim=160):
    if int(stride) != stride or stride < 1 or fps <= 0:
        raise ValueError("positive integer stride and positive fps required")
    # Always show the final physical state, even if stride does not divide steps.
    selected = np.unique(np.append(np.arange(0, len(result["time"]), int(stride)),
                                   len(result["time"])-1))
    rods = np.asarray(result["p_total"])[selected] * 1e3
    ext = np.asarray(result["external_position"])[selected] * 1e3
    tip_quat = np.asarray(result["tip_quaternion_xyzw"])[selected]
    ext_quat = np.asarray(result["external_quaternion_wxyz"])[selected]
    tip = rods[:, -1]
    times = np.asarray(result["time"])[selected]
    dim = result["magnets"].external_dimensions * 1e3
    tip_radius = result["magnets"].r_c * 1e3
    tip_length = result["magnets"].l_c * 1e3

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    rod_line, = ax.plot([], [], [], "-o", color="green", lw=2.6, ms=2.5,
                        label="catheter")
    tip_path, = ax.plot([], [], [], "--", color="red", lw=1.3,
                        label="catheter-tip path")
    tip_magnet = Poly3DCollection([], edgecolor="none", label="tip magnet")
    external_magnet = Poly3DCollection(
        [], edgecolor="none", label="external magnet"
    )
    ax.add_collection3d(tip_magnet)
    ax.add_collection3d(external_magnet)
    tip_n = ax.text(0, 0, 0, "N", color="black", fontsize=7,
                    ha="center", va="center")
    tip_s = ax.text(0, 0, 0, "S", color="black", fontsize=7,
                    ha="center", va="center")
    ext_n = ax.text(0, 0, 0, "N", color="black", fontsize=7,
                    ha="center", va="center")
    ext_s = ax.text(0, 0, 0, "S", color="black", fontsize=7,
                    ha="center", va="center")

    # Conservative orientation-independent extents keep the cuboid/cylinder
    # and pole labels inside the camera bounds, including tilted magnets.
    ext_padding = 0.6 * np.linalg.norm(dim)
    tip_padding = 0.7 * tip_length
    points = np.vstack([rods.reshape(-1, 3), ext-ext_padding,
                        ext+ext_padding, tip-tip_padding, tip+tip_padding])
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.max(hi - lo)), 0.5 * float(dim.max())) * 1.08
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=elev, azim=azim)
    # ax.view_init(elev=20, azim=100)
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("dynamic magnetic catheter")

    def update(frame):
        rod = rods[frame]
        rod_line.set_data_3d(rod[:, 0], rod[:, 1], rod[:, 2])
        tip_path.set_data_3d(tip[:frame+1, 0], tip[:frame+1, 1],
                             tip[:frame+1, 2])

        # SOFA local x is the rod tangent; MATLAB magnet local z is the
        # tangent. R_matlab = R_sofa * Ry(+90 degrees).
        r_sofa = magnetic_model.rotation_matrix_xyzw(tip_quat[frame])
        material_conversion = np.array([[0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [-1.0, 0.0, 0.0]])
        r_tip = r_sofa @ material_conversion
        tip_polys, tip_colors = _cylinder_polygons(
            tip_radius, tip_length, r_tip, tip[frame]
        )
        tip_magnet.set_verts(tip_polys)
        tip_magnet.set_facecolor(tip_colors)

        r_ext = magnetic_model.rotation_matrix_wxyz(ext_quat[frame])
        ext_polys, ext_colors = _cuboid_polygons(
            dim, r_ext, ext[frame]
        )
        external_magnet.set_verts(ext_polys)
        external_magnet.set_facecolor(ext_colors)

        tip_delta = 0.1 * tip_length
        _set_text_3d(tip_n, tip[frame] + r_tip @
                     np.array([0.0, 0.0, -tip_length/2.0-tip_delta]))
        _set_text_3d(tip_s, tip[frame] + r_tip @
                     np.array([0.0, 0.0, tip_length/2.0+tip_delta]))
        ext_delta = 0.1 * dim[2]
        _set_text_3d(ext_n, ext[frame] + r_ext @
                     np.array([0.0, 0.0, -dim[2]/2.0-ext_delta]))
        _set_text_3d(ext_s, ext[frame] + r_ext @
                     np.array([0.0, 0.0, dim[2]/2.0+ext_delta]))
        title.set_text(f"dynamic magnetic catheter - t = {times[frame]:.3f} s")
        return (rod_line, tip_path, tip_magnet, external_magnet,
                tip_n, tip_s, ext_n, ext_s, title)

    movie = animation.FuncAnimation(
        fig, update, frames=len(rods), interval=1000/fps, blit=False
    )
    _ensure_parent(filename)
    if filename.lower().endswith(".mp4") and animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2800)
    else:
        filename = os.path.splitext(filename)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    movie.save(filename, writer=writer, dpi=dpi)
    # Let notebook callers embed the actual output, including the GIF fallback.
    movie.saved_path = os.path.abspath(filename)
    plt.close(fig)
    print("  saved video :", filename)
    return movie
