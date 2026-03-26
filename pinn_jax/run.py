"""Example entry point: train a PINN on the heat equation."""

import json
import os
from datetime import datetime

import jax
import equinox as eqx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks import make_network
from problems import (
    make_heat, make_heat_ii, make_travelling_gaussian,
    make_travelling_gaussian_ii, make_radial_ball,
)
from training import train
from viz import plot_loss_curves, plot_l2_error, plot_solution_slice, plot_training_summary


def save_run(result, problem_name, run_dir=None):
    """Save training result to a timestamped folder under runs/.

    Creates runs/YYYYMMDD_HHMMSS_<problem_name>/ with:
      - config.json, meta.json
      - model.eqx  (equinox model weights)
      - losses.npz  (total, loss_r, loss_ic, loss_bc)
      - l2_errors.npz  (steps, errors)
      - summary.png
    """
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"{timestamp}_{problem_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(result.config, f, indent=2)

    # Model weights
    eqx.tree_serialise_leaves(os.path.join(run_dir, "model.eqx"), result.model)

    # Losses
    np.savez(
        os.path.join(run_dir, "losses.npz"),
        total=np.array(result.loss_history),
        loss_r=np.array(result.loss_components.get("loss_r", [])),
        loss_ic=np.array(result.loss_components.get("loss_ic", [])),
        loss_bc=np.array(result.loss_components.get("loss_bc", [])),
    )

    # L2 errors
    if result.l2_error_history:
        steps, errors = zip(*result.l2_error_history)
        np.savez(
            os.path.join(run_dir, "l2_errors.npz"),
            steps=np.array(steps),
            errors=np.array(errors),
        )

    # Metadata
    meta = {
        "problem_name": problem_name,
        "wall_time": result.wall_time,
        "n_steps": result.steps,
        "final_loss": float(result.loss_history[-1]) if result.loss_history else None,
        "final_l2": float(result.l2_error_history[-1][1]) if result.l2_error_history else None,
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Summary plot
    fig = plot_training_summary(result)
    fig.savefig(os.path.join(run_dir, "summary.png"), dpi=150)
    plt.close(fig)

    print(f"Run saved to {run_dir}")
    return run_dir


def run_heat_example(dim=5, n_steps=5000):
    """Train on the heat equation and visualize."""
    print(f"=== Heat equation, d={dim} ===")
    problem = make_heat(dim=dim, alpha=0.001, T=1.0)

    config = {
        "dim": dim,
        "arch": "modified_mlp",
        "use_fourier": True,
        "fourier_sigma": 1.0,
        "fourier_features": 64,
        "use_rwf": True,
        "depth": 4,
        "width": 148,
        "activation": "tanh",
        # Training
        "n_steps": n_steps,
        "lr": 1e-3,
        "lr_decay": 0.9999,
        "batch_r": 2048,
        "batch_ic": 512,
        "batch_bc": 512,
        "residual_mode": "sdgd" if dim > 8 else "exact",
        "sdgd_n_dims": min(dim, 5),
        "hard_bc": False,
        "hard_ic": False,
        "causal": False,
        "adaptive_weights": True,
        "adaptive_weights_every": 200,
        "log_every": 100,
        "eval_every": 500,
        "seed": 42,
    }

    key = jax.random.PRNGKey(0)
    model = make_network(config, key=key)
    result = train(model, problem, config)

    # Save run
    run_dir = save_run(result, f"heat_d{dim}")

    # Solution slice plot
    fig = plot_solution_slice(result.model, problem, config)
    fig.savefig(os.path.join(run_dir, "solution_slice.png"), dpi=150)
    plt.close(fig)
    print(f"Saved solution_slice.png")

    return result


def run_all_problems():
    """Quick demo of all 5 problems at low dim."""
    problems = {
        "heat": make_heat(dim=3),
        "heat_ii": make_heat_ii(dim=3),
        "trav_gauss": make_travelling_gaussian(dim=3),
        "trav_gauss_ii": make_travelling_gaussian_ii(dim=3),
        "radial_ball": make_radial_ball(dim=3),
    }

    base_config = {
        "arch": "modified_mlp",
        "use_fourier": True,
        "fourier_sigma": 1.0,
        "fourier_features": 64,
        "use_rwf": True,
        "depth": 4,
        "width": 128,
        "activation": "tanh",
        "n_steps": 3000,
        "lr": 1e-3,
        "lr_decay": 0.9999,
        "batch_r": 1024,
        "batch_ic": 256,
        "batch_bc": 256,
        "residual_mode": "exact",
        "hard_bc": False,
        "hard_ic": False,
        "causal": False,
        "adaptive_weights": True,
        "adaptive_weights_every": 200,
        "log_every": 500,
        "eval_every": 1000,
        "seed": 42,
    }

    for name, prob in problems.items():
        print(f"\n{'='*60}")
        print(f"Problem: {name}, dim={prob.dim}")
        print(f"{'='*60}")
        config = dict(base_config)
        config["dim"] = prob.dim
        key = jax.random.PRNGKey(0)
        model = make_network(config, key=key)
        result = train(model, prob, config)

        save_run(result, name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all_problems()
    else:
        dim = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        run_heat_example(dim=dim, n_steps=n_steps)
