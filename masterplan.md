## Roadmap

### Equations to try

time dependent & parabolic
- Heat eq.
- Fokker-Planck eq.

other major if time
- hamolton jacobi?
- belllman?
- finnacial one
- schroedinger

## Problem setup

### A simple case: Poisson eq. (elliptic, no time dependence)
FD (finite differences) / FEM (finite element method) $\to$ get matrix system $A x = b$ $\to$ how to solve? Some other way to discretize?

#### a) standard approaches
- just invert it!
- matrix iteration method - if you can save the matrix
    - Jacobi / Gauss-Seidel
    - CG
        - symmetric, positive definite

#### b) alternative stategies
- multigrid
- FFT
    - solve the PDE in place - no iteration - just compute 1 FT and 1 iFT - but on the entire grid
- sparse grids
    - for large dim
- MC??

#### c) NNs

- PINN
- PDE $\to$ variational form $\to$ minimize functional $J(u)$


### What changes for time dependent eqs?

- explicit time evolution? - nope, issue with stability
- implicit time evolution? - similar as before - solve linear system

apply the NNs from the prev chapeter

perhaps also BSDE?

#### Heat eq. (parabolic)

#### Fokker-Planck

look at kardiakis Fokker-Planck specific paper
