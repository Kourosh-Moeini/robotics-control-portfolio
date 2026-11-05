# Five-DOF tip control with an external magnet

Extended the learned controller to tip position and direction tracking using external-magnet translation and rotation.

## My contribution

I developed and evaluated the Python/PyTorch policy linked to the MATLAB magnetic-catheter model and selected the final 2,000-episode experiment.

## Selection and status

Selected exactly from Main Code-2000 eps-min v w-Rand-cube-13.ipynb, as specified by me.

## Scope of this sample

Five task DOFs mean three position coordinates plus two independent direction coordinates. The source has 19 observation values and six actions (three linear and three angular magnet-motion commands); it does not independently control catheter roll. This quasi-static environment uses dt=1.0, so the 10 Hz claim from the separate position-control project is not applied here. This archive also uses immediate-reward critic targets. The 70 MB sampling dataset, weights and workspaces are omitted.

Open `media` for the existing figures/videos and `code` for selected source. The portfolio-wide `SOURCE_MANIFEST.json` records file hashes and extraction details. Full original research folders remain separate.
