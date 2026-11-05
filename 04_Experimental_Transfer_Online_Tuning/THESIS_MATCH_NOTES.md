# Experimental source match — final thesis figures

The original embedded PNG outputs were decoded and compared by SHA-256 with the actual figure files referenced by the final thesis chapter. This identifies the notebook/output association without rerunning the experiment.

| Thesis figure | Source notebook | Source cell (zero-based) | SHA-256 |
|---|---|---:|---|
| `Spiral_compare.png` | `Comparisions-Fine Tune.ipynb` | 23 | `a8c3c548c394b24126ddda5404c4167506970ca1df8c1d4c8dd9ca47abb4118d` |
| `Square_compare.png` | `Comparisions-Fine Tune.ipynb` | 26 | `b51c0c120bb62b6a74092b2eee92329bd873801908e29f7bf7cfd2cef929fdfa` |
| `learning_curves.png` | `Experiment_Fine Tune.ipynb` | 33 | `682e3a64a12d2665faa96bf8ccecd984ad9fcdbb04d6a6c8b6953bdc7fc181cc` |
| `reward_components.png` | `Experiment_Fine Tune.ipynb` | 33 | `879f2d54495f3bea287a8b3825b5414fe4a78d6abe3d7e62eabd9c525cd0e28f` |

**Selection:** the unsuffixed `Experiment_Fine Tune.ipynb` for online adaptation and `Comparisions-Fine Tune.ipynb` for comparison. Trial-2 saved plots match a separate thesis archive, but the embedded outputs in the unsuffixed files match the final chapter figures directly, making them the stronger selection.

**Limits:** matching embedded images does not establish the exact final weight file, full execution sequence, data provenance or numerical reproducibility. Notebook source may have been edited after generating an output. Current environment/calibration files are review references, not a certified snapshot of the hardware run. No physical experiment or full training was rerun.
