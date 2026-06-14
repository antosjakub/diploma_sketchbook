# General nots

tensor products of one-dimensional multiscale functions

energy norm space grids
- need O(N) dof and 
- order const still depend exp on d

The sparse grid method has been successfully applied to
problems from
- quantum mechanics
- stochastic differential equations
- high-dimensional integration problems from physics and finance
- to the solution of moderately higher-dimensional partial differential equations, mainly of elliptic type [12].

## articles

A sparse grid space-time discretization scheme for parabolic problems
- https://link.springer.com/article/10.1007/s00607-007-0241-3

Sparse grids and related approximation schemes for higher dimensional problems
- https://ins.uni-bonn.de/media/public/publication-media/focm.pdf

Space-time approximation with sparse grids
- https://epubs.siam.org/doi/epdf/10.1137/050629252



# A sparse grid space-time discretization scheme for parabolic problems
- does not really rell how to implement
- use sCN
- adaptability not that hard to implement
- general domain possible
- o(n^(d+1)) -> o(n^d)

## 1. Introduction


## 2. Space-time sparse grids


## 3. Multilevel bases


## 4. Classical regularity theory for parabolic problems


## 5. Sparse grid discretization
- One advantage of our space-time sparse grids over classical sparse grids is the treatment of general domain - see fig 2 - super strange looking 3d like 2d domain discret


## 6. Adaptivity
- helps for non-smooth funs
- use the weighted size of the coefficients in the hierarchical basis representation as an error indicator for local refinement

- sCN method provided the best cost-benefit ratio in experiments

- we directly obtain so-called local time
stepping, i.e., different time steps are used in different parts of the spatial domain

- adaptivity quite easy to implement in space-time sg setting - do not need to invest significant implement. work

## 7. Concluding remarks

- space-time discretization schemes based on discontinuous Galerkin and Crank-Nicolson methods
- additional complexity due to time coordinate is comletely avoided and only the one based on space remains





# Space-time approximation with sparse grids
- a sequel to 'A sparse grid space-time ..'


adaptibility at 717 - stat