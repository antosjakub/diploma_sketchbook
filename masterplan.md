## Roadmap

### Equations to try

- Poisson eq. (elliptic)
- optionally: Heat eq. (parabolic)
- Fokker-Planck eq. (parabolic)

### 1) Poisson eq. (elliptic)
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

### 2) Heat eq. (parabolic) <- optional

### 3) Fokker-Planck

apply the NNs from the prev chapeter

perhaps also BSDE?


----------------



fix equations, increase dimension
try the same methods




kardiakis Fokker-Planck specific paper





try the major parabolic eqs
- heat
- FP
- hamolton jacobi?
- belllman?
- finnacial one
- if time - schroedinger



pinns?
- just try the most obvious and important things

- main problems in higher dims?
    - grad computation?

- what are the most important parts?
    - parallel
    - sampling parts
    - hyperparams

try diff approaches
    - sampling