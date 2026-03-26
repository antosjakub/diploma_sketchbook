"""Postprocessing script: regenerate plots from a saved run folder.

Usage:
    python postprocess.py runs/20260326_143000_heat_d5
    python postprocess.py runs/20260326_143000_heat_d5 --no-show
    python postprocess.py runs/20260326_143000_heat_d5 --heatmap
    python postprocess.py runs/20260326_143000_heat_d5 --animate
"""

import argparse
import json
import os

import jax
import equinox as eqx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from networks import make_network
from problems import (
    make_heat, make_heat_ii, make_travelling_gaussian,
    make_travelling_gaussian_ii, make_radial_ball,
)
from viz import plot_solution_slice, animate_solution

# Map problem names (as stored in meta.json) to their constructor.
# Naming convention: problem_name may include a dim suffix like "heat_d5",
# or be a bare name like "heat", "heat_ii", etc.
_PROBLEM_MAKERS = {
    "heat": make_heat,
    "heat_ii": make_heat_ii,
    "trav_gauss": make_travelling_gaussian,
    "travelling_gaussian": make_travelling_gaussian,
    "trav_gauss_ii": make_travelling_gaussian_ii,
    "travelling_gaussian_ii": make_travelling_gaussian_ii,
    "radial_ball": make_radial_ball,
}


def _resolve_problem_name(problem_name):
    """Strip dimension suffix (e.g. 'heat_d5' -> 'heat') and return maker."""
    # Try exact match first
    if problem_name in _PROBLEM_MAKERS:
        return _PROBLEM_MAKERS[problem_name]
    # Try stripping _d<N> suffix
    for base_name, maker in _PROBLEM_MAKERS.items():
        if problem_name.startswith(base_name + "_d"):
            return maker
    raise ValueError(
        f"Unknown problem '{problem_name}'. Known: {list(_PROBLEM_MAKERS.keys())}"
    )


def load_run(run_dir):
    """Load losses, l2 errors, config, and metadata from a run folder."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    losses = np.load(os.path.join(run_dir, "losses.npz"))

    l2_path = os.path.join(run_dir, "l2_errors.npz")
    l2 = np.load(l2_path) if os.path.exists(l2_path) else None

    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    return config, losses, l2, meta


def load_model(run_dir, config):
    """Reconstruct the model skeleton from config and load saved weights."""
    key = jax.random.PRNGKey(0)
    model = make_network(config, key=key)
    model = eqx.tree_deserialise_leaves(os.path.join(run_dir, "model.eqx"), model)
    return model


def make_problem(meta, config):
    """Reconstruct the PDEProblem from metadata and config."""
    maker = _resolve_problem_name(meta["problem_name"])
    dim = config["dim"]
    return maker(dim=dim)


def plot_losses(losses, config, meta):
    """Plot loss curves from saved data."""
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(losses["total"]) + 1)
    ax.plot(steps, losses["total"], label="total", linewidth=2)
    for key in ["loss_r", "loss_ic", "loss_bc"]:
        vals = losses[key]
        if len(vals) > 0 and np.any(vals > 0):
            ax.plot(steps, vals, label=key, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    title = "Training Loss"
    if meta.get("problem_name"):
        title += f" - {meta['problem_name']}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_l2(l2, meta):
    """Plot L2 relative error from saved data."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(l2["steps"], l2["errors"], "o-", linewidth=2, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative L2 Error")
    title = "L2 Error vs Training Step"
    if meta.get("problem_name"):
        title += f" - {meta['problem_name']}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate plots from a saved PINN run.")
    parser.add_argument("run_dir", help="Path to the run folder")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots, only save")
    parser.add_argument("--heatmap", action="store_true", help="Generate solution heatmap (loads model)")
    parser.add_argument("--animate", action="store_true", help="Generate animated heatmap over time (loads model)")
    parser.add_argument("--n-frames", type=int, default=60, help="Number of animation frames (default: 60)")
    parser.add_argument("--fps", type=int, default=15, help="Animation FPS (default: 15)")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    run_dir = args.run_dir
    config, losses, l2, meta = load_run(run_dir)

    print(f"Run: {run_dir}")
    if meta:
        print(f"  Problem: {meta.get('problem_name', 'N/A')}")
        print(f"  Steps: {meta.get('n_steps', 'N/A')}")
        wt = meta.get('wall_time')
        if wt is not None:
            print(f"  Wall time: {wt:.1f}s")
        if meta.get("final_loss") is not None:
            print(f"  Final loss: {meta['final_loss']:.2e}")
        if meta.get("final_l2") is not None:
            print(f"  Final L2 rel: {meta['final_l2']:.2e}")

    # Loss plot
    fig_loss = plot_losses(losses, config, meta)
    loss_path = os.path.join(run_dir, "loss_curves.png")
    fig_loss.savefig(loss_path, dpi=150)
    plt.close(fig_loss)
    print(f"Saved {loss_path}")

    # L2 plot
    if l2 is not None:
        fig_l2 = plot_l2(l2, meta)
        l2_path = os.path.join(run_dir, "l2_error.png")
        fig_l2.savefig(l2_path, dpi=150)
        plt.close(fig_l2)
        print(f"Saved {l2_path}")

    # Load model + problem if needed for heatmap/animation
    if args.heatmap or args.animate:
        model = load_model(run_dir, config)
        problem = make_problem(meta, config)

    # Solution heatmap
    if args.heatmap:
        fig_heat = plot_solution_slice(model, problem, config)
        heat_path = os.path.join(run_dir, "solution_slice.png")
        fig_heat.savefig(heat_path, dpi=150)
        plt.close(fig_heat)
        print(f"Saved {heat_path}")

    # Animated heatmap
    if args.animate:
        anim = animate_solution(model, problem, config,
                                n_frames=args.n_frames, fps=args.fps)
        anim_path = os.path.join(run_dir, "solution_anim.gif")
        anim.save(anim_path, writer="pillow", dpi=100)
        print(f"Saved {anim_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
