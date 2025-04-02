# Confidence Bands for Multiparameter Persistence Landscapes

This repository contains code to compute **confidence bands** using *standard bootstrap* and *multiplier bootstrap* for **multiparameter persistence landscapes**. The repository is a companion for the manuscript ["Confidence Bands for Multiparameter Persistence Landscapes"](link), and also contains the code to reproduce the experiments therein. 

## Structure of the code

The **code** in this repository is organized as follows:
- `confidence_bands.py`: contains python functions to compute confidence bands for a sample of `n` 1- and 2-parameter landscapes, given respectively by `(n, resolution)` and `(n, resolution, resolution)` arrays.
- `plotting.py`: contains auxiliary functions to plot point clouds in $\mathbb{R}^3$ and the confidence bands for 1- and 2-parameter landscapes.
- `sampling_functions.py`: contains functions to obtain samples of points in 3D over the torus, the sphere and the Klein bottle; and functions to add Gaussian and *salt and pepper* noise.
- `utils_PH.py`: contains functions to compute 1-parameter landscapes for the Vietoris--Rips filtration over a point cloud using [Ripser](https://github.com/Ripser/ripser) and [Gudhi](https://gudhi.inria.fr/); and 2-parameter landscapes for the Rips-Density function using a KDE as second parameter in the filtration, using [multipers](https://davidlapous.github.io/multipers/)

We also include two **Jupyter Notebooks**: `tutorial.ipynb`, containing an example showing how to compute confidence bands in 1- and 2-parameter persistence for a torus; and `MBD_classifier.ipynb`, containing the code to reproduce the experiments in the paper above and perform an MDB classification using the confidence bands of lansdcapes from torii, spheres and Klein bottles.

## Dependencies

The following libraries are necessary to run the code in this repository:
- [numpy](https://numpy.org/): to handle arrays
- [matplotlib](https://matplotlib.org/stable/index.html): for plotting functions
- [tqdm](https://github.com/tqdm/tqdm): to have progress bars in the computations within loops
- [scikit-learn](https://scikit-learn.org/stable/): for the Kernel Density Estimator and the K-fold validation in the MBD classifier
- [math](https://docs.python.org/3/library/math.html): for a rounding function to resample with replacement in the bootstrap functions
- [ripser](https://ripser.scikit-tda.org/en/latest/): to compute PH of the Vietoris--Rips filtration
- [multipers](https://davidlapous.github.io/multipers/): for the 2-parameter PH computations

## Citation
