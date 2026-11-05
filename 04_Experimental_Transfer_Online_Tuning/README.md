# Physical catheter control and online fine-tuning

Transferred the simulation-trained policy to a magnetic catheter setup with a six-DOF manipulator, an external permanent magnet and stereo-camera tip tracking, then fine-tuned it online.

## My contribution

I worked on the magnetic soft robot, robot-mounted magnet, stereo calibration/3D tracking, learned-policy implementation, experiments and comparisons between the base and fine-tuned policies.

## Selection and status

The strongest source match is the unsuffixed pair: Experiment_Fine Tune.ipynb and Comparisions-Fine Tune.ipynb. Their embedded notebook outputs match the final chapter learning, reward, spiral and square figures byte for byte (SHA-256 checked). The exact final training checkpoint and execution history are not established solely by saved output images.

## Scope of this sample

The presentation includes review-only class/method excerpts and the original demonstration video. No robot startup, motion, camera access or online learning is executed when opening the notebook. RobotSDK/drivers, calibrations, checkpoints, workspaces and the physical setup are omitted. The demonstration is accelerated and is not a real-time timing benchmark. Reproducing the thesis result is not claimed.

Open `media` for the existing figures/videos and `code` for selected source. The portfolio-wide `SOURCE_MANIFEST.json` records file hashes and extraction details. Full original research folders remain separate.
