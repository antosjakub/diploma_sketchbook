"""PDE problem definitions for PINN benchmarks.

Each problem is a dataclass with all the information needed to set up
and verify a PINN solver: residual, IC, BC, exact solution, source, etc.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class PDEProblem:
    name: str
    dim: int
    T: float
    domain_type: str             # "box" or "ball"
    domain_bounds: tuple         # e.g. (-1, 1) for box, or radius for ball
    residual_fn: Callable        # (u, du_dt, grad_u, laplacian_u, t, x, params) -> residual
    ic_fn: Callable              # (x, params) -> u0
    bc_fn: Callable              # (t, x, params) -> g
    exact_fn: Optional[Callable] = None  # (t, x, params) -> u_exact
    source_fn: Optional[Callable] = None  # (t, x, params) -> f
    problem_params: dict = field(default_factory=dict)


# ============================================================================
# 1. Heat equation:  u_t - alpha * laplacian_u = 0
# ============================================================================

def make_heat(dim: int, alpha: float = 0.1, T: float = 1.0,
              k: Optional[jax.Array] = None) -> PDEProblem:
    """Heat equation with product-of-sines exact solution."""
    if k is None:
        k = jnp.ones(dim) * jnp.pi
    params = {"alpha": alpha, "k": k}
    k_sq_sum = jnp.sum(k ** 2)

    def residual_fn(u, du_dt, grad_u, laplacian_u, t, x, p):
        return du_dt - p["alpha"] * laplacian_u

    def ic_fn(x, p):
        return jnp.prod(jnp.sin(p["k"] * x))

    def bc_fn(t, x, p):
        return exact_fn(t, x, p)

    def exact_fn(t, x, p):
        return jnp.prod(jnp.sin(p["k"] * x)) * jnp.exp(-p["alpha"] * jnp.sum(p["k"] ** 2) * t)

    return PDEProblem(
        name="heat",
        dim=dim,
        T=T,
        domain_type="box",
        domain_bounds=(-1.0, 1.0),
        residual_fn=residual_fn,
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        exact_fn=exact_fn,
        source_fn=None,
        problem_params=params,
    )


# ============================================================================
# 2. Heat equation II:  u_t - alpha * laplacian_u = f, with cos(beta*t)
# ============================================================================

def make_heat_ii(dim: int, alpha: float = 0.1, beta: float = 2.0,
                 T: float = 1.0, k: Optional[jax.Array] = None) -> PDEProblem:
    """Heat equation with source term and cos(beta*t) modulation."""
    if k is None:
        k = jnp.ones(dim) * jnp.pi
    params = {"alpha": alpha, "beta": beta, "k": k}

    def residual_fn(u, du_dt, grad_u, laplacian_u, t, x, p):
        src = source_fn(t, x, p)
        return du_dt - p["alpha"] * laplacian_u - src

    def source_fn(t, x, p):
        prod_sin = jnp.prod(jnp.sin(p["k"] * x))
        k_sq = jnp.sum(p["k"] ** 2)
        decay = jnp.exp(-p["alpha"] * k_sq * t)
        return -p["beta"] * prod_sin * decay * jnp.sin(p["beta"] * t)

    def ic_fn(x, p):
        return jnp.prod(jnp.sin(p["k"] * x))

    def bc_fn(t, x, p):
        return exact_fn(t, x, p)

    def exact_fn(t, x, p):
        prod_sin = jnp.prod(jnp.sin(p["k"] * x))
        k_sq = jnp.sum(p["k"] ** 2)
        return prod_sin * jnp.exp(-p["alpha"] * k_sq * t) * jnp.cos(p["beta"] * t)

    return PDEProblem(
        name="heat_ii",
        dim=dim,
        T=T,
        domain_type="box",
        domain_bounds=(-1.0, 1.0),
        residual_fn=residual_fn,
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        exact_fn=exact_fn,
        source_fn=source_fn,
        problem_params=params,
    )


# ============================================================================
# 3. Travelling Gaussian:  u_t - delta*laplacian_u + v.grad_u + w*u = f
# ============================================================================

def make_travelling_gaussian(
    dim: int,
    delta: float = 0.01,
    alpha_g: float = 1.0,
    beta_g: float = 0.5,
    T: float = 1.0,
    a: Optional[jax.Array] = None,
    b: Optional[jax.Array] = None,
    c: Optional[jax.Array] = None,
) -> PDEProblem:
    """Advection-diffusion-reaction with a travelling Gaussian packet solution."""
    if a is None:
        a = jnp.ones(dim)
    if b is None:
        b = jnp.zeros(dim)
    if c is None:
        c = jnp.ones(dim) * 0.5

    v = -c / a
    w = -2.0 * delta * alpha_g * jnp.sum(a ** 2)

    params = {
        "delta": delta, "alpha_g": alpha_g, "beta_g": beta_g,
        "a": a, "b": b, "c": c, "v": v, "w": w,
    }

    def _gaussian(t, x, p):
        z = p["a"] * x - p["b"] + p["c"] * t
        return jnp.exp(-p["alpha_g"] * jnp.sum(z ** 2) - p["beta_g"] * t)

    def exact_fn(t, x, p):
        return _gaussian(t, x, p)

    def source_fn(t, x, p):
        z = p["a"] * x - p["b"] + p["c"] * t
        a2 = p["a"] ** 2
        ag = p["alpha_g"]
        term1 = 4.0 * ag**2 * p["delta"] * jnp.sum(a2 * z**2)
        term2 = p["beta_g"]
        return -(term1 + term2) * _gaussian(t, x, p)

    def residual_fn(u, du_dt, grad_u, laplacian_u, t, x, p):
        src = source_fn(t, x, p)
        return du_dt - p["delta"] * laplacian_u + jnp.dot(p["v"], grad_u) + p["w"] * u - src

    def ic_fn(x, p):
        return exact_fn(0.0, x, p)

    def bc_fn(t, x, p):
        return exact_fn(t, x, p)

    return PDEProblem(
        name="travelling_gaussian",
        dim=dim,
        T=T,
        domain_type="box",
        domain_bounds=(-2.0, 2.0),
        residual_fn=residual_fn,
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        exact_fn=exact_fn,
        source_fn=source_fn,
        problem_params=params,
    )


# ============================================================================
# 4. Travelling Gaussian II: with cos(gamma*t) oscillation
# ============================================================================

def make_travelling_gaussian_ii(
    dim: int,
    delta: float = 0.01,
    alpha_g: float = 1.0,
    beta_g: float = 0.5,
    gamma: float = 3.0,
    T: float = 1.0,
    a: Optional[jax.Array] = None,
    b: Optional[jax.Array] = None,
    c: Optional[jax.Array] = None,
) -> PDEProblem:
    """Travelling + oscillating Gaussian packet."""
    if a is None:
        a = jnp.ones(dim)
    if b is None:
        b = jnp.zeros(dim)
    if c is None:
        c = jnp.ones(dim) * 0.5

    v = -c / a
    w = -2.0 * delta * alpha_g * jnp.sum(a ** 2)

    params = {
        "delta": delta, "alpha_g": alpha_g, "beta_g": beta_g, "gamma": gamma,
        "a": a, "b": b, "c": c, "v": v, "w": w,
    }

    def _gaussian(t, x, p):
        z = p["a"] * x - p["b"] + p["c"] * t
        return jnp.exp(-p["alpha_g"] * jnp.sum(z ** 2) - p["beta_g"] * t)

    def exact_fn(t, x, p):
        return _gaussian(t, x, p) * jnp.cos(p["gamma"] * t)

    def source_fn(t, x, p):
        z = p["a"] * x - p["b"] + p["c"] * t
        a2 = p["a"] ** 2
        ag = p["alpha_g"]
        gauss = _gaussian(t, x, p)
        term1 = 4.0 * ag**2 * p["delta"] * jnp.sum(a2 * z**2)
        term2 = p["beta_g"]
        cos_g = jnp.cos(p["gamma"] * t)
        sin_g = jnp.sin(p["gamma"] * t)
        return -((term1 + term2) * cos_g + p["gamma"] * sin_g) * gauss

    def residual_fn(u, du_dt, grad_u, laplacian_u, t, x, p):
        src = source_fn(t, x, p)
        return du_dt - p["delta"] * laplacian_u + jnp.dot(p["v"], grad_u) + p["w"] * u - src

    def ic_fn(x, p):
        return exact_fn(0.0, x, p)

    def bc_fn(t, x, p):
        return exact_fn(t, x, p)

    return PDEProblem(
        name="travelling_gaussian_ii",
        dim=dim,
        T=T,
        domain_type="box",
        domain_bounds=(-2.0, 2.0),
        residual_fn=residual_fn,
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        exact_fn=exact_fn,
        source_fn=source_fn,
        problem_params=params,
    )


# ============================================================================
# 5. Radially symmetric on unit ball
# ============================================================================

def make_radial_ball(dim: int, delta: float = 0.1, beta: float = 0.5,
                     T: float = 1.0) -> PDEProblem:
    """Radially symmetric diffusion on the unit ball.

    u(t,x) = sin(|x|^2 + a*t) * exp(-beta*t),  a = 2*d*delta
    """
    a_param = 2.0 * dim * delta
    params = {"delta": delta, "beta": beta, "a": a_param}

    def exact_fn(t, x, p):
        r2 = jnp.sum(x ** 2)
        return jnp.sin(r2 + p["a"] * t) * jnp.exp(-p["beta"] * t)

    def source_fn(t, x, p):
        r2 = jnp.sum(x ** 2)
        return (-p["beta"] + 4.0 * p["delta"] * r2) * jnp.sin(r2 + p["a"] * t) * jnp.exp(-p["beta"] * t)

    def residual_fn(u, du_dt, grad_u, laplacian_u, t, x, p):
        src = source_fn(t, x, p)
        return du_dt - p["delta"] * laplacian_u - src

    def ic_fn(x, p):
        return exact_fn(0.0, x, p)

    def bc_fn(t, x, p):
        return exact_fn(t, x, p)

    return PDEProblem(
        name="radial_ball",
        dim=dim,
        T=T,
        domain_type="ball",
        domain_bounds=(1.0,),  # radius
        residual_fn=residual_fn,
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        exact_fn=exact_fn,
        source_fn=source_fn,
        problem_params=params,
    )
