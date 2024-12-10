# hydropower-fatigue-damage-reduction

This repository contains code to model stress in hydraulic machines and trajectory optimization code developed during the EPFL PTMH & SDSC's [paired-hydro project](https://www.datascience.ch/projects/paired-hydro).

## System Requirements/Environment

The provided Dockerfile specifies an environment to run this code in. Alternatively, you can launch a renku session by clicking the `Start` button on this projects [renku page](https://renkulab.io/projects/musertill/hydropower-fatigue-damage-reduction). Toy data is tracked using git LFS. Please run `git lfs pull` to ensure these files are present. Specific dependencies are otherwise specified in `requirements.txt`. The software has been tested using the associated Docker image.

### Hardware
Due to the underlying deep learning model, this software runs faster if GPUs are present. The demo-script `demo.ipynb` is however designed to run solely CPU resources as well.

### Installation

To install, clone the repository and install the dependencies, e.g. via `pip` by running `pip install -r requirements.txt`. This should not take more than several minutes.

## Use and Demo
A demo notebook is present as `demo.ipynb` and shows the end-to-end process of training a model and running the optimization. On a normal machine with no GPU present, this notebook can take up to several hours to execute.

However, you can also execute standalone scripts `main.py` and `optimize_dijkstra.py` to train and optimize respectively.

### Training the model

To train the model, ensure appropriate training data is present in the input directory (`IN_FOLDER` which can be set in `CONFIG.py`), then run `main.py`. Training the model should take a few hours on e.g. a NVIDIA P100 GPU. We have added some toy data to showcase the expected structure of the input files.

### Dijkstra-based trajectory optimization

To run the optimization procedure, execute `optimize_dijkstra.py`. Using a NVIDIA P100 GPU, the optimization procedure can take a few days to complete on the original grid spacing. On a more coarse, grid, it can be significantly faster. An example of how to adjust the grid size is given in the optimization file.
