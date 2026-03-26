"""Collocation point samplers for PINNs.

Provides samplers for interior, initial condition, and boundary points
on box and ball domains.
"""

import jax
import jax.numpy as jnp


def sample_interior(key, n: int, dim: int, t_range: tuple,
                    domain_type: str, domain_bounds: tuple):
    """Sample (t, x) interior collocation points.

    Returns:
        t: (n,) in (0, T]
        x: (n, dim) in Omega interior
    """
    k1, k2 = jax.random.split(key)
    t_lo, t_hi = t_range
    # Sample t in (0, T] — use uniform and shift slightly away from 0
    t = jax.random.uniform(k1, (n,), minval=t_lo + 1e-7, maxval=t_hi)

    if domain_type == "box":
        lo, hi = domain_bounds
        x = jax.random.uniform(k2, (n, dim), minval=lo, maxval=hi)
    elif domain_type == "ball":
        radius = domain_bounds[0]
        x = _sample_ball_interior(k2, n, dim, radius)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")

    return t, x


def sample_initial(key, n: int, dim: int,
                   domain_type: str, domain_bounds: tuple):
    """Sample x points in Omega at t=0.

    Returns:
        x: (n, dim) in Omega
    """
    if domain_type == "box":
        lo, hi = domain_bounds
        x = jax.random.uniform(key, (n, dim), minval=lo, maxval=hi)
    elif domain_type == "ball":
        radius = domain_bounds[0]
        x = _sample_ball_interior(key, n, dim, radius)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")
    return x


def sample_boundary_box(key, n: int, dim: int, t_range: tuple,
                        bounds: tuple):
    """Sample (t, x) on box boundary faces.

    For each point: pick a random face (2*dim faces), fix that coordinate
    to the boundary value, sample remaining coords uniformly.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    lo, hi = bounds
    t_lo, t_hi = t_range

    t = jax.random.uniform(k1, (n,), minval=t_lo, maxval=t_hi)

    # Sample all coords uniformly first
    x = jax.random.uniform(k2, (n, dim), minval=lo, maxval=hi)

    # Choose which face: index in [0, 2*dim)
    face_idx = jax.random.randint(k3, (n,), minval=0, maxval=2 * dim)
    coord_idx = face_idx // 2  # which dimension
    side = face_idx % 2        # 0 -> lo, 1 -> hi

    # Set the chosen coordinate to the boundary value
    boundary_val = jnp.where(side == 0, lo, hi)
    # Create one-hot mask and replace
    one_hot = jax.nn.one_hot(coord_idx, dim)
    x = x * (1.0 - one_hot) + boundary_val[:, None] * one_hot

    return t, x


def sample_boundary_ball(key, n: int, dim: int, t_range: tuple,
                         radius: float):
    """Sample (t, x) on sphere surface |x| = radius."""
    k1, k2 = jax.random.split(key)
    t_lo, t_hi = t_range
    t = jax.random.uniform(k1, (n,), minval=t_lo, maxval=t_hi)

    # Sample uniformly on unit sphere, then scale
    x = jax.random.normal(k2, (n, dim))
    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms * radius

    return t, x


def _sample_ball_interior(key, n: int, dim: int, radius: float):
    """Sample uniformly inside d-ball using the radial method."""
    k1, k2 = jax.random.split(key)
    # Direction: uniform on sphere
    direction = jax.random.normal(k1, (n, dim))
    norms = jnp.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / norms
    # Radius: r ~ U(0,1)^(1/d) * radius for uniform density in ball
    r = jax.random.uniform(k2, (n, 1)) ** (1.0 / dim) * radius
    return direction * r
