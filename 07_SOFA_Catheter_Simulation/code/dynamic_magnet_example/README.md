# Dynamic external-magnet SOFA example

This is the compact original dynamic example, without generated MEX/build files, large datasets or personal runtime paths. Mechanics uses BeamAdapter, with a magnetic wrench recomputed from the current configuration. `mode.py` documents the selected material, external-magnet trajectory, timestep and substeps.

Use a compatible SOFA/SofaPython3/BeamAdapter Python environment, install NumPy and Matplotlib, set `SOFA_ROOT` to the local SOFA installation, and run `python mode.py` from this folder. It runs a simulation and writes data/plots; it does not control hardware. Do not install an unrelated package named `Sofa` from PyPI as a substitute for the SOFA runtime.

The included tests exercise the field model and input construction. The current exported example is checked for syntax and portable configuration; a complete SOFA solve and video regeneration are reported separately in VALIDATION.md only if actually run.
