# hydropower-fatigue-damage-reduction

This repository contains code to model stress in hydraulic machines and trajectory optimization code developed during the EPFL PTMH & SDSC's [paired-hydro project](https://www.datascience.ch/projects/paired-hydro).
### Environment

The provided Dockerfile specifies an environment to run this code in. Alternatively, you can launch a renku session by clicking the `Start` button on this projects [renku page](https://renkulab.io/projects/musertill/hydropower-fatigue-damage-reduction).

### Training the model

To train the model, ensure appropriate training data is present in the `data/normalized` directory, then run `main.py`. Training the model should take a few hours on e.g. a NVIDIA P100 GPU.

### Dijkstra-based trajectory optimization

To run the optimization procedure, execute `optimize_dijkstra.py`. Using a NVIDIA P100 GPU, the optimization procedure can take a few days to complete,
