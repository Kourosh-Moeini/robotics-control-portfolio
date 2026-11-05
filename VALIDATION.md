# Validation performed — 5 November 2026

This file reports checks on the portfolio copy, not new scientific results.

- Syntax parsed successfully for **26 Python source files** and **7 notebooks**. Notebook outputs and execution counts are cleared.
- Four extracted actor/critic implementations passed CPU forward passes, expected state/action shape checks, action-bound checks, critic backpropagation and replay-buffer sampling checks. These are implementation-interface checks, not trained-policy accuracy tests.
- The supplied SOFA analytical magnetic-field/input tests passed, including analytic versus finite-difference field gradients. They run independently of the SOFA dynamics runtime.
- All **87** source artifacts listed in SOURCE_MANIFEST.json exist and match their recorded hashes. HTML links to local files were checked. A basic token/private-key pattern scan found no matching secrets.
- Notebook excerpts preserve the archived learning equations. They are deliberately incomplete review samples: large workspaces, checkpoints, private core implementations and hardware dependencies are not bundled.

## Not rerun

Full model training, the MATLAB simulation experiments, robot/camera operations, a complete SOFA dynamics solve and live LLM experiments were **not** rerun. Existing plots/videos are historical outputs, not newly reproduced results.

## Source identification limits

The five-DOF and 10 Hz notebooks are exactly the versions selected by Kourosh. Embedded outputs in the unsuffixed Experiment_Fine Tune.ipynb and Comparisions-Fine Tune.ipynb match the final thesis chapter learning/reward and spiral/square figures byte for byte. This is a source association, not proof of the exact final checkpoint or a newly reproduced result. See the experimental THESIS_MATCH_NOTES.md. The final static PINN code is not on this laptop; its folder contains the accepted manuscript and explanation only.


## Media revision — 5 November 2026

Replaced the standalone DRL spiral demonstration with two original hybrid DDPG/PID infinity-path tracking views. Visually inspected midpoint and near-end frames and decoded both complete video streams successfully. Added eight existing MPC/adaptive-control figures, visually checked their labels/content and preserved their original image bytes. All media hashes and HTML links were rechecked; the portfolio ZIP was rebuilt. These changes add no newly run controller or LLM experiments.

Four existing notebooks had changed JSON formatting/metadata since the previous export; their source cells were verified unchanged and their current local files were preserved. Video poster images are unaltered frames from the selected clips.

## Public upload scope

This public copy omits the private-core benchmark driver reference and local application/preparation files. Original model code and media are preserved from the curated portfolio; no new training, MATLAB solve or hardware experiment was run for the GitHub upload.
