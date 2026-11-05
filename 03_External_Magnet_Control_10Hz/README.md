# External-magnet control: 10 Hz policy

Trained a DDPG-based catheter-tip controller in Python/PyTorch while a MATLAB Cosserat model simulated the catheter and the effect of an external magnet.

## My contribution

I built the learning and evaluation workflow, linked Python and MATLAB, and tested target reaching and trajectory tracking before experimental transfer.

## Selection and status

Selected exactly from Main Code-500 eps-10 HZ.ipynb, not the 100 Hz alternative.

## Scope of this sample

The historical project is called “2 DOF”, referring to the evaluated tip-tracking task; the saved implementation has a nine-component observation and three external-magnet position commands with fixed magnet orientation. MATLAB mechanics runs at 100 Hz and the policy holds each action for ten steps (10 Hz control). The critic in this archived implementation fits immediate rewards; the source is preserved rather than silently converted to a different DDPG algorithm. The 50 MB sampling dataset, weights and full workspaces are omitted.

Open `media` for the existing figures/videos and `code` for selected source. The portfolio-wide `SOURCE_MANIFEST.json` records file hashes and extraction details. Full original research folders remain separate.
