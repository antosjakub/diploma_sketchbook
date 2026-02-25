# My summary

Modern strategies avoid full grids by using randomness, separability, or learning.

Monte Carlo and particle simulations offer simplicity and dimension-independence.
Neural-network approaches (PINNs, deep BSDE/deep splitting) leverage ML to represent high-D functions.
Low-rank tensor methods exploit separability, and sparse grids use specialized quadrature to reduce nodes.

Trade-offs: (e.g. Monte Carlo noise vs. neural network training cost vs. decomposition accuracy)


## feynman kac
- underlying SDE
- brownina motion evolution from x back in time
- convergence 1/sqrt(N)

## deep DBSE
- reformulate as SDE, look at eval
- hundreds of dims 
## deep splitting
- split domains and let a diff NN solve each domain

## PINNS
- can do higher dims if good arch
- use SGDG - sample & not compute full gradient
    - do not compute the full gradient of the PDE residual over all dimensions at each step
    - sample a random subset of dimensions for each gradient update
    - hundreds of dims for HJE and schoedinger on a single gpu
    - https://www.sciencedirect.com/science/article/pii/S0893608024002934#:~:text=arbitrary%20high,equations%20in%20tens%20of%20thousands

## tensor trains
- low rank structure

## sparse grids
- selectively choose subset of grid points
- spectral base
- 5-10 d

## QM variational methods
- diffusive MC
- schroedinger = diffusive like eq. in imag time
- MCTHD - hartree


