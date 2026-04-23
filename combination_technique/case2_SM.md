# The Smoluchowski equation

Consider the motion of an overdamped Brownian particle in an external potential $V(x)$ on $\mathbb{R}^d$, whose stochastic differential equation has the form

$$
dX_t = -\nabla V(X_t)\,dt + \sqrt{2\beta^{-1}}\,dW_t
$$ 


$$
\partial_t \rho = \beta^{-1}\Delta\rho + \nabla\cdot\bigl(\rho\nabla V\bigr) \\
\rho(x,0)=\rho_0(x)
$$ 

This is a linear drift-diffusion equation.

In particular, the diffusion is time-reversible and admits a Gibbs invariant measure.


If $V$ is confining, that is:

$$
\lim_{|x|\to\infty}V(x)=+\infty\quad \text{and} \quad e^{-\beta V}\in L^1
$$


then there exists a unique stationary solution in the form of a Boltzman density
$$
\rho_\infty(x) = \frac{1}{Z} e^{-\beta V(x)} \qquad Z=\int_{\mathbb R^d}e^{-\beta V(x)}dx
$$ 

and under mild smoothness and convexity assumptions on $V$, this convergence is exponentially fast


## The Initial condition
- gaussian with unit variance and centered at origin

## The bc
either dirichlet far from origin or reflecting Neumann to prevent mass leak


## Coubled Quadratic potential

$$
V = 1/2 x^T A x
$$

## Double well

$$
V = 1/4 \sum (x_i^2 - a_i^2)^2
$$

## Rastigin


$$
V = \sum x_i^2 - A\cos(\gamma_i x_i)
$$











