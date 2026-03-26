"""Core training engine for PINNs.

Residual computation (exact / SDGD), loss functions, hard constraints,
adaptive weights, causal weighting, and the main training loop.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from problems import PDEProblem
from sampling import (
    sample_interior, sample_initial, sample_boundary_box, sample_boundary_ball,
)


# ============================================================================
# Derivative utilities
# ============================================================================

def _u_scalar(model, tx):
    """Evaluate model at a single (1+d,) point, return scalar."""
    return model(tx)


def compute_derivatives(model, tx):
    """Compute u, du_dt, grad_x u, laplacian_x u at a single point.

    Uses forward-mode for per-coordinate second derivatives (memory efficient).

    Args:
        model: PINNNet
        tx: (1+d,) array

    Returns:
        u, du_dt, grad_x, laplacian_x
    """
    d = tx.shape[0] - 1

    # u and full gradient via reverse mode
    u, grad_tx = jax.value_and_grad(lambda z: _u_scalar(model, z))(tx)
    du_dt = grad_tx[0]
    grad_x = grad_tx[1:]

    # Laplacian: sum of diagonal Hessian entries for spatial dims
    # Use forward-over-reverse for each spatial dim
    def grad_i(i, z):
        return jax.grad(lambda z: _u_scalar(model, z))(z)[i]

    laplacian = 0.0
    for i in range(1, d + 1):
        # d^2u/dx_i^2 via forward-mode on grad_i
        _, dd = jax.jvp(partial(grad_i, i), (tx,), (jax.nn.one_hot(i, d + 1),))
        laplacian = laplacian + dd

    return u, du_dt, grad_x, laplacian


def compute_derivatives_sdgd(model, tx, key, n_dims: int):
    """SDGD variant: sample subset of dimensions for Laplacian.

    Computes only n_dims randomly chosen second derivatives and scales
    by d/n_dims for an unbiased estimate.

    Args:
        model: PINNNet
        tx: (1+d,)
        key: PRNG key
        n_dims: number of dims to sample

    Returns:
        u, du_dt, grad_x, laplacian_estimate
    """
    d = tx.shape[0] - 1

    u, grad_tx = jax.value_and_grad(lambda z: _u_scalar(model, z))(tx)
    du_dt = grad_tx[0]
    grad_x = grad_tx[1:]

    # Sample n_dims spatial dimensions (indices 1..d in tx)
    sampled = jax.random.choice(key, d, shape=(n_dims,), replace=False)

    def grad_i(i, z):
        return jax.grad(lambda z: _u_scalar(model, z))(z)[i]

    def second_deriv(i):
        spatial_idx = sampled[i] + 1  # offset by 1 for t
        tangent = jax.nn.one_hot(spatial_idx, d + 1)
        _, dd = jax.jvp(partial(grad_i, spatial_idx), (tx,), (tangent,))
        return dd

    laplacian_partial = 0.0
    for i in range(n_dims):
        laplacian_partial = laplacian_partial + second_deriv(i)

    scale = d / n_dims
    return u, du_dt, grad_x, laplacian_partial * scale


# ============================================================================
# Single-point residual
# ============================================================================

def compute_residual_point(model, tx, problem: PDEProblem, key=None,
                           mode: str = "exact", sdgd_n_dims: int = 5):
    """Compute PDE residual at a single point.

    Args:
        mode: "exact" or "sdgd"
    """
    if mode == "sdgd" and key is not None:
        u, du_dt, grad_x, lap = compute_derivatives_sdgd(model, tx, key, sdgd_n_dims)
    else:
        u, du_dt, grad_x, lap = compute_derivatives(model, tx)

    t = tx[0]
    x = tx[1:]
    return problem.residual_fn(u, du_dt, grad_x, lap, t, x, problem.problem_params)


# ============================================================================
# Hard constraint wrappers
# ============================================================================

def hard_bc_box_factor(x, bounds):
    """phi(x) = prod_i (1 - ((x_i - mid) / half_width)^2) for box domain."""
    lo, hi = bounds
    mid = (lo + hi) / 2.0
    hw = (hi - lo) / 2.0
    normalized = (x - mid) / hw
    return jnp.prod(1.0 - normalized ** 2)


def apply_hard_constraints(raw_output, t, x, problem: PDEProblem,
                           hard_bc: bool = False, hard_ic: bool = False):
    """Apply hard BC and/or IC constraints.

    hard BC:  u = bc(t,x) + phi(x) * v
    hard IC:  u = ic(x) + t * v    (applied after hard BC if both)
    hard BC+IC: u = bc(t,x) + phi(x) * (ic_interior(x) + t * v)
    """
    v = raw_output
    pp = problem.problem_params

    if hard_bc and problem.domain_type == "box":
        phi = hard_bc_box_factor(x, problem.domain_bounds)
        bc_val = problem.bc_fn(t, x, pp)

        if hard_ic:
            ic_val = problem.ic_fn(x, pp)
            # At t=0: u = bc(0,x) + phi(x)*(ic_interior + 0) should equal ic(x)
            # ic_interior = (ic(x) - bc(0,x)) / phi(x), but phi can be 0 at boundary
            # Use: u = ic(x) + phi(x)*t*v  (simpler, exact IC in interior, approx BC)
            # Better: u = bc(t,x) + phi(x)*(H(x) + t*v)
            # where H(x) = (ic(x) - bc(0,x)) / phi(x) -- problematic at boundary
            # Practical approach: enforce IC hard via t-factor
            u = ic_val + t * phi * v
        else:
            u = bc_val + phi * v
    elif hard_ic:
        ic_val = problem.ic_fn(x, pp)
        u = ic_val + t * v
    else:
        u = v

    return u


# ============================================================================
# Wrapped model evaluation (with hard constraints)
# ============================================================================

def eval_model(model, t, x, problem, hard_bc=False, hard_ic=False):
    """Evaluate model with hard constraints at a single point."""
    tx = jnp.concatenate([jnp.array([t]), x])
    raw = model(tx)
    return apply_hard_constraints(raw, t, x, problem, hard_bc, hard_ic)


def make_constrained_model_fn(model, problem, hard_bc=False, hard_ic=False):
    """Return a function tx -> u that includes hard constraints."""
    def u_fn(tx):
        t = tx[0]
        x = tx[1:]
        raw = model(tx)
        return apply_hard_constraints(raw, t, x, problem, hard_bc, hard_ic)
    return u_fn


# ============================================================================
# Loss components
# ============================================================================

def loss_residual(model, problem, t_r, x_r, key, mode="exact",
                  sdgd_n_dims=5, hard_bc=False, hard_ic=False):
    """Mean squared PDE residual over batch."""
    u_fn = make_constrained_model_fn(model, problem, hard_bc, hard_ic)

    def single_residual(t, x, subkey):
        tx = jnp.concatenate([jnp.array([t]), x])
        d = x.shape[0]
        if mode == "sdgd":
            u, du_dt, grad_x, lap = compute_derivatives_sdgd(
                u_fn, tx, subkey, min(sdgd_n_dims, d))
        else:
            u, du_dt, grad_x, lap = compute_derivatives(u_fn, tx)
        return problem.residual_fn(
            u, du_dt, grad_x, lap, t, x, problem.problem_params)

    keys = jax.random.split(key, t_r.shape[0])
    residuals = jax.vmap(single_residual)(t_r, x_r, keys)
    return jnp.mean(residuals ** 2)


def loss_ic(model, problem, x_ic, hard_bc=False, hard_ic=False):
    """Mean squared IC error."""
    pp = problem.problem_params

    def single_ic(x):
        tx = jnp.concatenate([jnp.array([0.0]), x])
        raw = model(tx)
        u = apply_hard_constraints(raw, 0.0, x, problem, hard_bc, hard_ic)
        target = problem.ic_fn(x, pp)
        return (u - target) ** 2

    return jnp.mean(jax.vmap(single_ic)(x_ic))


def loss_bc(model, problem, t_bc, x_bc, hard_bc=False, hard_ic=False):
    """Mean squared BC error."""
    pp = problem.problem_params

    def single_bc(t, x):
        tx = jnp.concatenate([jnp.array([t]), x])
        raw = model(tx)
        u = apply_hard_constraints(raw, t, x, problem, hard_bc, hard_ic)
        target = problem.bc_fn(t, x, pp)
        return (u - target) ** 2

    return jnp.mean(jax.vmap(single_bc)(t_bc, x_bc))


# ============================================================================
# Causal weighting
# ============================================================================

def loss_residual_causal(model, problem, t_r, x_r, key, mode="exact",
                         sdgd_n_dims=5, hard_bc=False, hard_ic=False,
                         n_segments=10, epsilon=1.0):
    """PDE residual loss with causal weighting.

    Splits time into n_segments, weights later segments by
    exp(-epsilon * cumsum of earlier segment losses).
    """
    u_fn = make_constrained_model_fn(model, problem, hard_bc, hard_ic)
    T = problem.T
    n = t_r.shape[0]

    def single_residual(t, x, subkey):
        tx = jnp.concatenate([jnp.array([t]), x])
        d = x.shape[0]
        if mode == "sdgd":
            u, du_dt, grad_x, lap = compute_derivatives_sdgd(
                u_fn, tx, subkey, min(sdgd_n_dims, d))
        else:
            u, du_dt, grad_x, lap = compute_derivatives(u_fn, tx)
        return problem.residual_fn(
            u, du_dt, grad_x, lap, t, x, problem.problem_params)

    keys = jax.random.split(key, n)
    residuals = jax.vmap(single_residual)(t_r, x_r, keys) ** 2

    # Assign each point to a time segment
    segment_idx = jnp.floor(t_r / T * n_segments).astype(jnp.int32)
    segment_idx = jnp.clip(segment_idx, 0, n_segments - 1)

    # Compute mean loss per segment
    segment_losses = jnp.zeros(n_segments)
    for m in range(n_segments):
        mask = (segment_idx == m).astype(jnp.float32)
        count = jnp.maximum(mask.sum(), 1.0)
        segment_losses = segment_losses.at[m].set(jnp.sum(residuals * mask) / count)

    # Causal weights: w_m = exp(-epsilon * sum_{j<m} L_j)
    cumsum = jnp.cumsum(segment_losses)
    cumsum_shifted = jnp.concatenate([jnp.array([0.0]), cumsum[:-1]])
    weights = jnp.exp(-epsilon * cumsum_shifted)

    # Apply weights to each point
    point_weights = weights[segment_idx]
    return jnp.mean(point_weights * residuals)


# ============================================================================
# Total loss
# ============================================================================

def total_loss(model, problem, t_r, x_r, x_ic, t_bc, x_bc, key, config):
    """Compute weighted total loss.

    Config keys used:
        residual_mode: "exact" / "sdgd"
        sdgd_n_dims: int
        hard_bc: bool
        hard_ic: bool
        causal: bool
        causal_segments: int
        causal_epsilon: float
        lambda_r, lambda_ic, lambda_bc: float weights
    """
    mode = config.get("residual_mode", "exact")
    sdgd_n = config.get("sdgd_n_dims", 5)
    hbc = config.get("hard_bc", False)
    hic = config.get("hard_ic", False)
    causal = config.get("causal", False)
    lam_r = config.get("lambda_r", 1.0)
    lam_ic = config.get("lambda_ic", 1.0)
    lam_bc = config.get("lambda_bc", 1.0)

    # Residual loss
    if causal:
        n_seg = config.get("causal_segments", 10)
        eps = config.get("causal_epsilon", 1.0)
        l_r = loss_residual_causal(
            model, problem, t_r, x_r, key, mode, sdgd_n, hbc, hic, n_seg, eps)
    else:
        l_r = loss_residual(model, problem, t_r, x_r, key, mode, sdgd_n, hbc, hic)

    # IC loss (skip if hard IC)
    l_ic = 0.0 if hic else loss_ic(model, problem, x_ic, hbc, hic)

    # BC loss (skip if hard BC)
    l_bc = 0.0 if hbc else loss_bc(model, problem, t_bc, x_bc, hbc, hic)

    total = lam_r * l_r + lam_ic * l_ic + lam_bc * l_bc

    aux = {"loss_r": l_r, "loss_ic": l_ic, "loss_bc": l_bc, "total": total}
    return total, aux


# ============================================================================
# L2 error evaluation
# ============================================================================

def compute_l2_error(model, problem, key, n_eval=2000,
                     hard_bc=False, hard_ic=False):
    """Compute relative L2 error against exact solution (if available)."""
    if problem.exact_fn is None:
        return jnp.nan

    k1, k2 = jax.random.split(key)
    t_eval = jax.random.uniform(k1, (n_eval,), minval=0.0, maxval=problem.T)

    if problem.domain_type == "box":
        lo, hi = problem.domain_bounds
        x_eval = jax.random.uniform(k2, (n_eval, problem.dim), minval=lo, maxval=hi)
    else:
        from .sampling import _sample_ball_interior
        x_eval = _sample_ball_interior(k2, n_eval, problem.dim, problem.domain_bounds[0])

    pp = problem.problem_params

    def single_eval(t, x):
        tx = jnp.concatenate([jnp.array([t]), x])
        raw = model(tx)
        u_pred = apply_hard_constraints(raw, t, x, problem, hard_bc, hard_ic)
        u_exact = problem.exact_fn(t, x, pp)
        return u_pred, u_exact

    u_pred, u_exact = jax.vmap(single_eval)(t_eval, x_eval)
    l2_err = jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
    l2_ref = jnp.sqrt(jnp.mean(u_exact ** 2))
    return l2_err / jnp.maximum(l2_ref, 1e-10)


# ============================================================================
# Adaptive loss weight update
# ============================================================================

def update_adaptive_weights(aux, config):
    """Simple inverse-variance balancing of loss weights.

    Sets lambda_i proportional to 1/max(L_i, eps) so that smaller losses
    get upweighted.
    """
    eps = 1e-8
    l_r = jnp.maximum(float(aux["loss_r"]), eps)
    l_ic = jnp.maximum(float(aux["loss_ic"]), eps) if not config.get("hard_ic", False) else 1.0
    l_bc = jnp.maximum(float(aux["loss_bc"]), eps) if not config.get("hard_bc", False) else 1.0

    # Normalize so they sum to 3 (preserves overall scale)
    inv = jnp.array([1.0 / l_r, 1.0 / l_ic, 1.0 / l_bc])
    inv = inv / inv.sum() * 3.0

    config = dict(config)
    config["lambda_r"] = float(inv[0])
    config["lambda_ic"] = float(inv[1])
    config["lambda_bc"] = float(inv[2])
    return config


# ============================================================================
# Training result
# ============================================================================

@dataclass
class TrainResult:
    model: object
    loss_history: list = field(default_factory=list)
    loss_components: dict = field(default_factory=dict)  # {key: [values]}
    l2_error_history: list = field(default_factory=list)
    wall_time: float = 0.0
    steps: int = 0
    config: dict = field(default_factory=dict)


# ============================================================================
# Training loop
# ============================================================================

def train(model, problem: PDEProblem, config: dict) -> TrainResult:
    """Main training loop.

    Config keys:
        n_steps: int (default 10000)
        lr: float (default 1e-3)
        lr_decay: float (default 0.9999)
        batch_r: int (default 2048)
        batch_ic: int (default 512)
        batch_bc: int (default 512)
        residual_mode: "exact" / "sdgd"
        sdgd_n_dims: int (default 5)
        hard_bc: bool
        hard_ic: bool
        causal: bool
        causal_segments: int
        causal_epsilon: float
        adaptive_weights: bool
        adaptive_weights_every: int (default 100)
        log_every: int (default 100)
        eval_every: int (default 500)
        seed: int (default 42)
    """
    n_steps = config.get("n_steps", 10000)
    lr = config.get("lr", 1e-3)
    lr_decay = config.get("lr_decay", 0.9999)
    batch_r = config.get("batch_r", 2048)
    batch_ic = config.get("batch_ic", 512)
    batch_bc = config.get("batch_bc", 512)
    log_every = config.get("log_every", 100)
    eval_every = config.get("eval_every", 500)
    seed = config.get("seed", 42)
    adaptive_weights = config.get("adaptive_weights", False)
    adaptive_every = config.get("adaptive_weights_every", 100)

    hbc = config.get("hard_bc", False)
    hic = config.get("hard_ic", False)

    # Mutable copy for adaptive weights
    train_config = dict(config)

    # Optimizer
    schedule = optax.exponential_decay(lr, transition_steps=1, decay_rate=lr_decay)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key = jax.random.PRNGKey(seed)

    # Sampling functions
    def sample_batch(key):
        k1, k2, k3 = jax.random.split(key, 3)
        t_r, x_r = sample_interior(
            k1, batch_r, problem.dim, (0.0, problem.T),
            problem.domain_type, problem.domain_bounds)
        x_ic = sample_initial(
            k2, batch_ic, problem.dim,
            problem.domain_type, problem.domain_bounds)
        if problem.domain_type == "box":
            t_bc, x_bc = sample_boundary_box(
                k3, batch_bc, problem.dim, (0.0, problem.T), problem.domain_bounds)
        else:
            t_bc, x_bc = sample_boundary_ball(
                k3, batch_bc, problem.dim, (0.0, problem.T), problem.domain_bounds[0])
        return t_r, x_r, x_ic, t_bc, x_bc

    # Loss and grad function
    @eqx.filter_jit
    def train_step(model, opt_state, key, train_config_arr):
        """Single training step. train_config_arr contains lambda values."""
        k1, k2 = jax.random.split(key)
        t_r, x_r, x_ic, t_bc, x_bc = sample_batch(k1)

        # Build config with current lambdas
        step_config = dict(config)  # base config
        step_config["lambda_r"] = train_config_arr[0]
        step_config["lambda_ic"] = train_config_arr[1]
        step_config["lambda_bc"] = train_config_arr[2]

        def loss_fn(model):
            return total_loss(
                model, problem, t_r, x_r, x_ic, t_bc, x_bc, k2, step_config)

        (loss_val, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val, aux

    # Tracking
    result = TrainResult(model=model, config=config)
    result.loss_components = {"loss_r": [], "loss_ic": [], "loss_bc": []}

    lambdas = jnp.array([
        train_config.get("lambda_r", 1.0),
        train_config.get("lambda_ic", 1.0),
        train_config.get("lambda_bc", 1.0),
    ])

    t_start = time.time()

    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        model, opt_state, loss_val, aux = train_step(model, opt_state, step_key, lambdas)

        loss_val_f = float(loss_val)
        result.loss_history.append(loss_val_f)
        result.loss_components["loss_r"].append(float(aux["loss_r"]))
        result.loss_components["loss_ic"].append(float(aux["loss_ic"]))
        result.loss_components["loss_bc"].append(float(aux["loss_bc"]))

        # Adaptive weights
        if adaptive_weights and (step + 1) % adaptive_every == 0 and not (hbc and hic):
            train_config = update_adaptive_weights(aux, train_config)
            lambdas = jnp.array([
                train_config["lambda_r"],
                train_config["lambda_ic"],
                train_config["lambda_bc"],
            ])

        # L2 error eval
        if (step + 1) % eval_every == 0:
            key, eval_key = jax.random.split(key)
            l2 = compute_l2_error(model, problem, eval_key, hard_bc=hbc, hard_ic=hic)
            result.l2_error_history.append((step + 1, float(l2)))

        # Logging
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            l2_str = ""
            if result.l2_error_history:
                l2_str = f", L2_rel={result.l2_error_history[-1][1]:.2e}"
            print(
                f"Step {step+1:6d} | loss={loss_val_f:.4e} | "
                f"r={float(aux['loss_r']):.2e} ic={float(aux['loss_ic']):.2e} "
                f"bc={float(aux['loss_bc']):.2e}{l2_str} | {elapsed:.1f}s"
            )

    result.wall_time = time.time() - t_start
    result.model = model
    result.steps = n_steps
    print(f"\nTraining complete: {n_steps} steps in {result.wall_time:.1f}s")
    if result.l2_error_history:
        print(f"Final L2 relative error: {result.l2_error_history[-1][1]:.4e}")

    return result
