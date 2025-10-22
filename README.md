# roadmap

# Poisson eq. (elliptic) / Heat eq. (parabolic)


### standard approach

FD (finite differences) / FEM (finite element method) $\to$ get matrix system $A x = b$ $to$ how to solve?

- just invert it!
- matrix iteration method - if you can save the matrix
    - Jacobi / Gauss-Seidel
    - CG
        - symmetric, positive definite
- multigrid
- FFT
    - solve the PDE in place - no iteration - just compute 1 FT and 1 iFT - but on the entire grid
- sparse grids
    - for large dim
- MC??

### NNs
try:
    - PINN
    - PDE $\to$ variational form $\to$ minimize functional $J(u)$


# Fokker-Planck

### apply the NNs from the prev chapeter

perhaps also BSDE?

