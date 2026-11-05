"""Dynamic SOFA catheter with configuration-dependent magnetic actuation.

Submodules are intentionally not imported eagerly so the analytical magnetic
model and its tests can run on machines where SOFA is not installed.
"""

__all__ = [
    "magnetic_model",
    "mag_parameters",
    "main_code",
    "parameters",
    "paths",
    "sofa_model",
    "trajectory",
    "visualization",
]
