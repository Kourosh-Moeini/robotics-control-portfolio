"""Output paths for the dynamic magnetic SOFA package."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(ROOT, "videos")
PLOTS = os.path.join(ROOT, "plots")
DATA = os.path.join(ROOT, "data")


def ensure_dirs() -> None:
    for directory in (VIDEOS, PLOTS, DATA):
        os.makedirs(directory, exist_ok=True)


def case_stem(params) -> str:
    return (f"{params.flag}_{params.matter}_{params.shape}_{params.mag_set}_"
            f"{params.freq:.2f}_{params.amp:.2f}")


def video(params, suffix="3D", ext="mp4") -> str:
    ensure_dirs()
    return os.path.join(VIDEOS, f"SOFA_{case_stem(params)}_{suffix}.{ext}")


def plot(params, suffix="tip_and_wrench", ext="png") -> str:
    ensure_dirs()
    return os.path.join(PLOTS, f"SOFA_{case_stem(params)}_{suffix}.{ext}")


def dataset(params, suffix="results", ext="npz") -> str:
    ensure_dirs()
    return os.path.join(DATA, f"SOFA_{case_stem(params)}_{suffix}.{ext}")
