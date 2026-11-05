# Wrench control: DDPG and PID feedback

Learned a tip-position-to-wrench mapping with a PyTorch actor–critic policy coupled to a MATLAB Cosserat catheter model, then evaluated DDPG/PID hybrid tracking against standalone controllers.

## My contribution

I developed the Python learning/control workflow, linked it to the MATLAB model, and compared trajectory tracking under changing speeds and disturbances.

## Selection and status

Research simulation; accepted ICBME manuscript supplied. IEEE publication is forthcoming, as confirmed by me; this package does not claim an assigned DOI.

## Scope of this sample

The selected 1,000-episode notebook is used by the archived comparison notebook and is consistent with its checkpoint names. Exact regeneration of the accepted paper figures has not been established. MATLAB Engine, MATLAB model/toolboxes, trajectory workspaces and trained checkpoints are needed for the full experiment.

Open `media` for the existing figures/videos and `code` for selected source. The portfolio-wide `SOURCE_MANIFEST.json` records file hashes and extraction details. Full original research folders remain separate.


## Tracking videos

- `media/hybrid_wrench_infinity_tracking.mp4`: hybrid DDPG/PID tracking, showing the actual and desired tip paths closely overlapping.
- `media/hybrid_wrench_infinity_simulation_3D.mp4`: the corresponding saved three-dimensional catheter simulation.

These originals come from `Renders/DRL_PID-Pos Corr/Test/no_disturbance/` and are associated with `Main Code-RL PID-Pos Corr.ipynb`, whose configuration uses the 1,000-episode checkpoint family. The earlier standalone-DRL spiral clip was removed from this presentation because it showed the uncorrected tracking offset. The two new renders use different playback rates; they are not synchronised wall-clock recordings. See `VIDEO_SELECTION.md` for source details.
