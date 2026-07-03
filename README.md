
# Numerically solving high-dimensional diffusion equation via machine learning


## Overview

traditional solvers - FDM, FEM, multigrid, ... - all perform discretization of the domain - issue in high-dims - curse of dimensionality

sparse grids - strategically construct a collection of grids - hard limit based on available computer memory

sparse grids combination technique = super parallizable

machine learning approaches - mesh-free - strategies to overcome memoty constrains in high dim - but runtime / convergence issues

## thesis text

`PINN/`
- `thesis-en/` - latex folder for the theoretical basis of pinns

`SG/`
- `thesis-en/` - latex folder for the theoretical info bout sparse grids combination technique

`Numerical_experiments/`
- `thesis-en/` - latex folder for discussing the pdes to solve and the results from numerical experiments
