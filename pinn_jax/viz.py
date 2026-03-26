"""Visualization utilities for PINN training results."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from training import apply_hard_constraints


def plot_loss_curves(result, log_scale=True):
    """Plot all loss components + total vs step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(result.loss_history) + 1)
    ax.plot(steps, result.loss_history, label="total", linewidth=2)
    for key, vals in result.loss_components.items():
        if any(v > 0 for v in vals):
            ax.plot(steps, vals, label=key, alpha=0.7)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_l2_error(result):
    """Plot L2 relative error vs step."""
    if not result.l2_error_history:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    steps, errors = zip(*result.l2_error_history)
    ax.plot(steps, errors, "o-", linewidth=2, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("L2 Error vs Training Step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_solution_slice(model, problem, config, t_values=None,
                        fixed_dims=None, dim1=0, dim2=1, n_grid=50):
    """2D heatmap of predicted vs exact solution.

    Fixes all spatial dims except dim1, dim2 at fixed_dims values.
    One row per time value; columns: predicted, exact, |error|.
    Uses vmap for fast evaluation.
    """
    if t_values is None:
        t_values = [0.0, problem.T / 2, problem.T]
    dim = problem.dim
    if fixed_dims is None:
        fixed_dims = 0.2*jnp.ones(dim)

    hard_bc = config.get("hard_bc", False)
    hard_ic = config.get("hard_ic", False)

    if problem.domain_type == "box":
        lo, hi = problem.domain_bounds
    else:
        lo, hi = -problem.domain_bounds[0], problem.domain_bounds[0]

    xs = jnp.linspace(lo, hi, n_grid)
    X1, X2 = jnp.meshgrid(xs, xs, indexing="ij")
    x1_flat = X1.ravel()
    x2_flat = X2.ravel()

    # Build (n_grid^2, dim) array of spatial points: fixed_dims with dim1, dim2 varied
    base = jnp.broadcast_to(fixed_dims, (n_grid * n_grid, dim))
    x_all = base.at[:, dim1].set(x1_flat).at[:, dim2].set(x2_flat)

    def eval_pred(t_val, x):
        tx = jnp.concatenate([jnp.array([t_val]), x])
        raw = model(tx)
        return apply_hard_constraints(raw, t_val, x, problem, hard_bc, hard_ic)

    def eval_exact(t_val, x):
        return problem.exact_fn(t_val, x, problem.problem_params)

    has_exact = problem.exact_fn is not None
    n_t = len(t_values)
    ncols = 3 if has_exact else 1
    fig, axes = plt.subplots(n_t, ncols, figsize=(5 * ncols, 4 * n_t), squeeze=False)

    for row, t_val in enumerate(t_values):
        u_pred = np.array(jax.vmap(lambda x: eval_pred(t_val, x))(x_all)).reshape(n_grid, n_grid)
        if has_exact:
            u_exact = np.array(jax.vmap(lambda x: eval_exact(t_val, x))(x_all)).reshape(n_grid, n_grid)

        X1_np, X2_np = np.array(X1), np.array(X2)
        vmin = min(u_pred.min(), u_exact.min()) if has_exact else u_pred.min()
        vmax = max(u_pred.max(), u_exact.max()) if has_exact else u_pred.max()

        ax = axes[row, 0]
        im = ax.pcolormesh(X1_np, X2_np, u_pred, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"Predicted, t={t_val:.2f}")
        ax.set_xlabel(f"$x_{{{dim1}}}$")
        ax.set_ylabel(f"$x_{{{dim2}}}$")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

        if has_exact:
            ax2 = axes[row, 1]
            im2 = ax2.pcolormesh(X1_np, X2_np, u_exact, vmin=vmin, vmax=vmax, cmap="viridis")
            ax2.set_title(f"Exact, t={t_val:.2f}")
            ax2.set_xlabel(f"$x_{{{dim1}}}$")
            ax2.set_ylabel(f"$x_{{{dim2}}}$")
            ax2.set_aspect("equal")
            plt.colorbar(im2, ax=ax2)

            ax3 = axes[row, 2]
            err = np.abs(u_pred - u_exact)
            im3 = ax3.pcolormesh(X1_np, X2_np, err, cmap="Reds")
            ax3.set_title(f"|Error|, t={t_val:.2f}")
            ax3.set_xlabel(f"$x_{{{dim1}}}$")
            ax3.set_ylabel(f"$x_{{{dim2}}}$")
            ax3.set_aspect("equal")
            plt.colorbar(im3, ax=ax3)

    fig.suptitle(f"{problem.name}, d={problem.dim}", fontsize=14)
    fig.tight_layout()
    return fig


def plot_solution_1d(model, problem, config, t_values=None,
                     free_dim=0, fixed_dims=None, n_points=200):
    """1D slice along one coordinate. Fix all others."""
    if t_values is None:
        t_values = [0.0, problem.T / 2, problem.T]
    dim = problem.dim
    if fixed_dims is None:
        fixed_dims = jnp.zeros(dim)

    hard_bc = config.get("hard_bc", False)
    hard_ic = config.get("hard_ic", False)

    if problem.domain_type == "box":
        lo, hi = problem.domain_bounds
    else:
        lo, hi = -problem.domain_bounds[0], problem.domain_bounds[0]
    xs = np.linspace(lo, hi, n_points)

    has_exact = problem.exact_fn is not None
    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 4), squeeze=False)

    for col, t_val in enumerate(t_values):
        u_pred = []
        u_exact_vals = []
        for xi in xs:
            x_jax = jnp.array(fixed_dims).at[free_dim].set(xi)
            tx = jnp.concatenate([jnp.array([t_val]), x_jax])
            raw = model(tx)
            u_pred.append(float(apply_hard_constraints(
                raw, t_val, x_jax, problem, hard_bc, hard_ic)))
            if has_exact:
                u_exact_vals.append(float(problem.exact_fn(
                    t_val, x_jax, problem.problem_params)))

        ax = axes[0, col]
        ax.plot(xs, u_pred, label="predicted", linewidth=2)
        if has_exact:
            ax.plot(xs, u_exact_vals, "--", label="exact", linewidth=2)
        ax.set_title(f"t={t_val:.2f}")
        ax.set_xlabel(f"x_{free_dim}")
        ax.set_ylabel("u")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{problem.name}, d={problem.dim}, 1D slice", fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_summary(result):
    """Combined figure: loss curves + L2 error + timing info."""
    has_l2 = bool(result.l2_error_history)
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3 if has_l2 else 2, figure=fig)

    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    steps = np.arange(1, len(result.loss_history) + 1)
    ax1.plot(steps, result.loss_history, label="total", linewidth=2)
    for key, vals in result.loss_components.items():
        if any(v > 0 for v in vals):
            ax1.plot(steps, vals, label=key, alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # L2 error
    if has_l2:
        ax2 = fig.add_subplot(gs[0, 1])
        s, e = zip(*result.l2_error_history)
        ax2.plot(s, e, "o-", linewidth=2, markersize=4)
        ax2.set_yscale("log")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Relative L2 Error")
        ax2.set_title("L2 Error")
        ax2.grid(True, alpha=0.3)

    # Info text
    ax3 = fig.add_subplot(gs[0, -1])
    ax3.axis("off")
    info = [
        f"Steps: {result.steps}",
        f"Wall time: {result.wall_time:.1f}s",
        f"Final loss: {result.loss_history[-1]:.2e}" if result.loss_history else "",
    ]
    if result.l2_error_history:
        info.append(f"Final L2 rel: {result.l2_error_history[-1][1]:.2e}")
    cfg = result.config
    info.extend([
        "",
        f"Arch: {cfg.get('arch', 'N/A')}",
        f"Dim: {cfg.get('dim', 'N/A')}",
        f"Residual: {cfg.get('residual_mode', 'exact')}",
        f"Hard BC: {cfg.get('hard_bc', False)}",
        f"Hard IC: {cfg.get('hard_ic', False)}",
        f"Causal: {cfg.get('causal', False)}",
    ])
    ax3.text(0.1, 0.9, "\n".join(info), transform=ax3.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace")
    ax3.set_title("Config & Results")

    fig.tight_layout()
    return fig


def animate_solution(model, problem, config, fixed_dims=None,
                     dim1=0, dim2=1, n_grid=50, n_frames=60, fps=15):
    """Animate predicted vs exact vs |error| heatmaps over time.

    Returns a matplotlib FuncAnimation. Save with:
        anim.save("out.mp4", writer="ffmpeg", dpi=150)
    or  anim.save("out.gif", writer="pillow", dpi=100)
    """
    dim = problem.dim
    if fixed_dims is None:
        fixed_dims = 0.2*jnp.ones(dim)

    hard_bc = config.get("hard_bc", False)
    hard_ic = config.get("hard_ic", False)

    if problem.domain_type == "box":
        lo, hi = problem.domain_bounds
    else:
        lo, hi = -problem.domain_bounds[0], problem.domain_bounds[0]

    xs = jnp.linspace(lo, hi, n_grid)
    X1, X2 = jnp.meshgrid(xs, xs, indexing="ij")
    x1_flat, x2_flat = X1.ravel(), X2.ravel()

    base = jnp.broadcast_to(fixed_dims, (n_grid * n_grid, dim))
    x_all = base.at[:, dim1].set(x1_flat).at[:, dim2].set(x2_flat)
    X1_np, X2_np = np.array(X1), np.array(X2)

    t_values = np.linspace(0.0, float(problem.T), n_frames)
    has_exact = problem.exact_fn is not None

    def eval_pred(t_val, x):
        tx = jnp.concatenate([jnp.array([t_val]), x])
        raw = model(tx)
        return apply_hard_constraints(raw, t_val, x, problem, hard_bc, hard_ic)

    def eval_exact(t_val, x):
        return problem.exact_fn(t_val, x, problem.problem_params)

    # Precompute all frames
    all_pred = []
    all_exact = []
    for t_val in t_values:
        u_pred = np.array(jax.vmap(lambda x: eval_pred(t_val, x))(x_all)).reshape(n_grid, n_grid)
        all_pred.append(u_pred)
        if has_exact:
            u_exact = np.array(jax.vmap(lambda x: eval_exact(t_val, x))(x_all)).reshape(n_grid, n_grid)
            all_exact.append(u_exact)

    # Global color range for consistent coloring across frames
    all_pred_arr = np.stack(all_pred)
    vmin_sol = all_pred_arr.min()
    vmax_sol = all_pred_arr.max()
    if has_exact:
        all_exact_arr = np.stack(all_exact)
        vmin_sol = min(vmin_sol, all_exact_arr.min())
        vmax_sol = max(vmax_sol, all_exact_arr.max())
        all_err = np.abs(all_pred_arr - all_exact_arr)
        vmax_err = all_err.max()

    # Set up figure
    ncols = 3 if has_exact else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    # Initial meshes
    im_pred = axes[0].pcolormesh(X1_np, X2_np, all_pred[0],
                                  vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
    axes[0].set_title("Predicted")
    axes[0].set_xlabel(f"$x_{{{dim1}}}$")
    axes[0].set_ylabel(f"$x_{{{dim2}}}$")
    axes[0].set_aspect("equal")
    plt.colorbar(im_pred, ax=axes[0])

    if has_exact:
        im_exact = axes[1].pcolormesh(X1_np, X2_np, all_exact[0],
                                       vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        axes[1].set_title("Exact")
        axes[1].set_xlabel(f"$x_{{{dim1}}}$")
        axes[1].set_ylabel(f"$x_{{{dim2}}}$")
        axes[1].set_aspect("equal")
        plt.colorbar(im_exact, ax=axes[1])

        im_err = axes[2].pcolormesh(X1_np, X2_np, np.abs(all_pred[0] - all_exact[0]),
                                     vmin=0, vmax=vmax_err, cmap="Reds")
        axes[2].set_title("|Error|")
        axes[2].set_xlabel(f"$x_{{{dim1}}}$")
        axes[2].set_ylabel(f"$x_{{{dim2}}}$")
        axes[2].set_aspect("equal")
        plt.colorbar(im_err, ax=axes[2])

    suptitle = fig.suptitle(f"{problem.name}, d={problem.dim}, t={t_values[0]:.3f}", fontsize=14)
    fig.tight_layout()

    def update(frame):
        im_pred.set_array(all_pred[frame].ravel())
        if has_exact:
            im_exact.set_array(all_exact[frame].ravel())
            im_err.set_array(np.abs(all_pred[frame] - all_exact[frame]).ravel())
        suptitle.set_text(f"{problem.name}, d={problem.dim}, t={t_values[frame]:.3f}")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    return anim
