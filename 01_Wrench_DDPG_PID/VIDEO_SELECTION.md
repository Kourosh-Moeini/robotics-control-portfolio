# Tracking video selection

Selected from the original `Trial 2-random points` project:

| Portfolio file | Original file under `Renders/DRL_PID-Pos Corr/Test/no_disturbance/` | View |
|---|---|---|
| `hybrid_wrench_infinity_tracking.mp4` | `DRL_PID_Inf_main.mp4` | Actual versus desired XY tip path, hybrid DRL/PID feedback |
| `hybrid_wrench_infinity_simulation_3D.mp4` | `DRL_PID_Inf_setup.mp4` | MATLAB three-dimensional catheter render |

`Main Code-RL PID-Pos Corr.ipynb` defines `render_movement_drl_pid` in zero-based cell 14 and generates these filenames in the no-disturbance test block at cell 19. Its path configuration names the 1,000-episode checkpoint family. Source association was checked; the historical experiment was not rerun.

The previous `DRL-1000 eps/Test_DRL/Test_Last_Spiral_main.mp4` shows standalone DRL tracking with a visible offset. It remains unchanged in the original research directory, but is no longer included as the hybrid-controller demonstration.

Midpoint and near-end frames from the new clips were visually inspected. Both entire streams decoded successfully with FFmpeg. The original files were copied without re-encoding. The XY clip is approximately 16.63 s at 30 fps; the MATLAB view is approximately 5.01 s at 100 fps. Their playback speeds differ. Close visual path overlap is shown; zero numerical tracking error is not inferred from the video.
