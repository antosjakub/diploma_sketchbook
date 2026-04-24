# Sparse Grid Combination Technique

This project implements the sparse grid combination technique for
high-dimensional linear parabolic PDEs, especially equations of the form

```math
\partial_t p = a \Delta p + b(x) \cdot \nabla p + c(x) p .
```

The main idea is to avoid solving one isotropic full tensor grid problem in
dimension `d`. Instead, we solve many smaller anisotropic tensor-grid problems
and combine their solutions with signed coefficients.

## Tensor Grids

Let the physical domain be

```math
\Omega = [-L,L]^d .
```

For a multi-index

```math
\ell = (\ell_1,\ldots,\ell_d) \in \mathbb N^d ,
```

define an anisotropic tensor grid

```math
\Omega_\ell
= \Omega_{\ell_1}^{(1)} \times \cdots \times \Omega_{\ell_d}^{(d)} ,
```

where direction `j` has mesh width

```math
h_j = 2^{-\ell_j} (2L)
```

up to endpoint convention. Thus large `ell_j` means fine resolution in
coordinate `x_j`, while small `ell_j` means coarse resolution. The number of
unknowns on one component grid scales like

```math
N_\ell \approx \prod_{j=1}^d 2^{\ell_j}
= 2^{|\ell|_1}.
```

The combination technique uses grids with different anisotropies rather than
only the isotropic grid `(n,\ldots,n)`, whose cost scales like `2^{nd}`.

## Sparse Grid Index Set

For sparse-grid level `n` in dimension `d`, the classical combination formula
uses all level vectors satisfying

```math
n \le |\ell|_1 \le n+d-1 .
```

Equivalently, the active component grids are

```math
\mathcal I_{n,d}
= \{\ell \in \mathbb N^d : n \le |\ell|_1 \le n+d-1\}.
```

These are anisotropic full grids close to the sparse-grid boundary. In two
dimensions this corresponds to combining grids along the diagonals
`ell_1 + ell_2 = n` and `ell_1 + ell_2 = n+1`. In higher dimensions there are
`d` such diagonals.

## Combination Formula

Let `u_ell` denote the numerical solution computed on the tensor grid
`Omega_ell`. The sparse-grid approximation at level `n` is

```math
\widehat u_n
=
\sum_{\ell \in \mathcal I_{n,d}}
c_\ell u_\ell ,
```

with coefficients

```math
c_\ell
=
(-1)^{n+d-|\ell|_1-1}
\binom{d-1}{|\ell|_1-n}.
```

So all grids with the same value of `|\ell|_1` have the same coefficient. The
sign alternates by diagonal, and the binomial factor accounts for cancellation
of lower-order tensor contributions.

The formula should be understood after bringing all component-grid solutions
to a common representation. In practice this can mean:

- interpolate each `u_ell` to a common evaluation grid,
- evaluate the combined solution pointwise,
- or store the sparse-grid representation implicitly through the component
  solutions and weights.

## Computing the PDE Solution

For a linear PDE

```math
\partial_t p = \mathcal L p ,
```

the combination technique proceeds as follows.

1. Choose dimension `d`, sparse-grid level `n`, domain `[-L,L]^d`, boundary
   conditions, and final time `T`.
2. Generate all level vectors

   ```math
   \ell \in \mathcal I_{n,d}.
   ```

3. For each `ell`, build the anisotropic tensor grid `Omega_ell`.
4. On each tensor grid, discretize the same PDE operator `L` using standard
   finite differences in each coordinate direction.
5. Evolve the component-grid solution independently:

   ```math
   \frac{d p_\ell}{dt} = L_\ell p_\ell ,
   \qquad
   p_\ell(0) = I_\ell p_0 ,
   ```

   where `I_ell p_0` is the initial condition sampled or projected onto
   `Omega_ell`.
6. Combine the final component solutions:

   ```math
   \widehat p_n(T,x)
   =
   \sum_{\ell \in \mathcal I_{n,d}}
   c_\ell p_\ell(T,x).
   ```

Each component solve is a regular tensor-grid PDE solve, so the implementation
can use standard NumPy/SciPy sparse matrices and time steppers. The component
solves are independent, which makes parallelization over `ell` natural.

## Why It Helps

A full isotropic tensor grid with about `2^n` points per coordinate has cost

```math
O(2^{nd}).
```

The sparse-grid combination technique uses many anisotropic grids whose total
effective size scales like

```math
O(2^n n^{d-1}),
```

up to dimension-dependent constants. For sufficiently smooth solutions with
bounded mixed derivatives, this retains nearly the same second-order spatial
accuracy as the full tensor grid, with the typical sparse-grid error form

```math
O(2^{-2n} n^{d-1}).
```

The price is that the method relies on mixed regularity and may be less
effective for nonsmooth, sharply localized, or strongly non-axis-aligned
features.

## Current Python Layout

The initial implementation is organized as a small reusable package:

- `combination_technique.indices`: sparse-grid level sets and combination
  weights.
- `combination_technique.grid`: endpoint-including anisotropic tensor grids.
- `combination_technique.fd`: one-dimensional and tensor-product finite
  difference matrices.
- `combination_technique.models`: OU, Smoluchowski, and
  convection-diffusion-reaction model definitions.
- `combination_technique.solver`: one-grid theta-method time stepping.
- `combination_technique.combination`: parallel component solves and final
  combination.
- `combination_technique.sgpp`: SG++ sparse-grid interpolation,
  dehierarchisation, batched evaluation, quadrature, and plotting helpers.

Minimal example:

```python
import numpy as np

from combination_technique import (
    OrnsteinUhlenbeck,
    TimeStepper,
    gaussian_density,
    solve_combination,
)

model = OrnsteinUhlenbeck(np.eye(2))
result = solve_combination(
    model,
    level=3,
    initial_condition=gaussian_density,
    final_time=0.1,
    stepper=TimeStepper(dt=0.02, theta=1.0),
    domain_radius=4.0,
    bc="dirichlet",
    max_workers=1,
)
```

To compare assembled and matrix-free component solves, switch the backend:

```python
from combination_technique import LinearSolveConfig

result = solve_combination(
    model,
    level=3,
    initial_condition=gaussian_density,
    final_time=0.1,
    stepper=TimeStepper(dt=0.02, theta=1.0),
    domain_radius=4.0,
    bc="dirichlet",
    max_workers=1,
    operator_backend="linear_operator",
    linear_solve=LinearSolveConfig(
        method="gmres",
        preconditioner="jacobi",
        rtol=1e-8,
        maxiter=200,
    ),
)
```

Use `operator_backend="matrix"` for the original assembled sparse-matrix path.
The `linear_operator` backend avoids materializing the full tensor-grid
operator and is intended for larger high-dimensional component problems.
`preconditioner="jacobi"` uses a diagonal left preconditioner for the implicit
theta-method system and is the recommended first option before trying more
expensive preconditioners.

General convection-diffusion-reaction example:

```python
import numpy as np

from combination_technique import (
    ConvectionDiffusionReaction,
    TimeStepper,
    gaussian_density,
    solve_combination,
)

model = ConvectionDiffusionReaction(
    dimension=3,
    diffusion=0.2,
    drift_fn=lambda x: -0.35 * x,
    reaction_fn=lambda x: 0.15 - 0.05 * np.sum(x * x, axis=0),
)

result = solve_combination(
    model,
    level=4,
    initial_condition=gaussian_density,
    final_time=0.05,
    stepper=TimeStepper(dt=0.01, theta=1.0),
    domain_radius=4.0,
    bc="dirichlet",
    max_workers=1,
)
```

## SG++ Workflow

This repository uses SG++ only for the final sparse-grid surrogate layer.
That is the intended split:

- this package solves the component tensor-grid PDE problems and performs the
  classical combination step,
- SG++ stores the final sparse-grid interpolant, evaluates it at points,
  dehierarchises it when needed, computes quadrature-based mass and moments,
  and generates slice plots.

That keeps the core combination-technique algebra explicit in this package and
uses SG++ only where it adds clear value.

Typical workflow:

1. Solve the PDE with `solve_combination(...)`.
2. Project the resulting combined solution onto an SG++ sparse grid with
   `result_to_sgpp(...)`.
3. Use the returned `SGppInterpolant` for:
   - `evaluate(points)`
   - `nodal_values()`
   - `integral()`
   - `mean()`
   - `covariance()`
   - `save_slice_plot(...)`

Example:

```python
import numpy as np

from combination_technique import (
    ConvectionDiffusionReaction,
    TimeStepper,
    gaussian_density,
    result_to_sgpp,
    solve_combination,
)

model = ConvectionDiffusionReaction(
    dimension=3,
    diffusion=0.2,
    drift_fn=lambda x: -0.35 * x,
    reaction_fn=lambda x: 0.15 - 0.05 * np.sum(x * x, axis=0),
)

result = solve_combination(
    model,
    level=4,
    initial_condition=gaussian_density,
    final_time=0.05,
    stepper=TimeStepper(dt=0.01, theta=1.0),
    domain_radius=4.0,
    bc="dirichlet",
    max_workers=1,
)

sparse = result_to_sgpp(result, level=4, boundary=True)
values = sparse.evaluate(np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]]))
mass = sparse.integral()
mean = sparse.mean()
cov = sparse.covariance()
sparse.save_slice_plot("cdr_sgpp_slice.png", axes=(0, 1), fixed={2: 0.0})
```

### Why `boundary=True` for Quadrature

When the SG++ surrogate is used for mass and moment computations on a bounded
PDE domain, prefer `boundary=True` in `result_to_sgpp(...)` or
`SGppInterpolant.from_function(...)`.

That choice matters for:

- `integral()`
- `mean()`
- `covariance()`

Reason: for bounded-domain density-like quantities, quadrature is more
appropriate on a sparse grid that includes the domain boundary. For pure point
evaluation or qualitative plotting, the interior linear grid may still be
acceptable, but for mass and moments the boundary grid is the safer default.

The current boundary closures are homogeneous Dirichlet and homogeneous
Neumann. Exact reflecting flux conditions such as
`(Sigma grad p + x p) . n = 0` for full-covariance OU should be implemented as
a later model-specific boundary discretization.
