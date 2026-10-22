"""Shared plotting module for main_score_pinn_3losses.py / main_score_pinn_hardcoded.py.

Two uses:
  (1) Called at the end of a training run via `plot_run(dir_name, model, pde_model,
      score_sde_model, args, device, model_s=None, losses=None, l2_errs=None)`.
  (2) Run standalone against a saved run directory to regenerate all plots:

          python plot_results.py gauss/run_3losses_score_pde
          python plot_results.py laplace/run_hardcoded_ll_ode
"""
import argparse
import os
import sys

import torch
from matplotlib import pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, '../src/')
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)
#
#import architecture
#import utility
import viz
#import visualize_training_metrics
#import pde_models



# ---------------- plotting ----------------

def _plot_log_pde(dir_name, model, pde_model, args, options):
    T = args.T
    model_fn_q = viz.wrapp_model(model)
    model_fn_p = lambda X: torch.exp(model_fn_q(X))
    q_ic = lambda X: pde_model.q_0(X[:, :-1])
    q_inf = lambda X: pde_model.q_inf(X[:, :-1])
    p_ic = lambda X: pde_model.p_0(X[:, :-1])
    p_inf = lambda X: pde_model.p_inf(X[:, :-1])

    # IC: q and p vs q_0 and p_0
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,0)").heatmap(model_fn_q)
    plotter.add_panel('q_ic', title="q_0(x)").heatmap(q_ic)
    plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q0.png', t_val=0.0)
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,0) = exp(model_q(x,0))").heatmap(model_fn_p)
    plotter.add_panel('p_ic', title="p_0(x)").heatmap(p_ic)
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p0.png', t_val=0.0)
    # inf: q and p vs q_inf and p_inf
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,T)").heatmap(model_fn_q)
    plotter.add_panel('q_inf', title="q_inf(x)").heatmap(q_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q_inf.png', t_val=T)
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,T)").heatmap(model_fn_p)
    plotter.add_panel('p_inf', title="p_inf(x)").heatmap(p_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p_inf.png', t_val=T)
    # anim: q and p
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,t)").heatmap(model_fn_q)
    plotter.save_animation(f'{dir_name}/viz/anim_model_q.gif', num_frames=30, fps=5, t_end=T)
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,t) = exp(model_q(x,t))").heatmap(model_fn_p)
    plotter.save_animation(f'{dir_name}/viz/anim_model_p.gif', num_frames=30, fps=5, t_end=T)


def _plot_class_pde(dir_name, model, pde_model, args, options):
    T = args.T
    model_fn = viz.wrapp_model(model)
    p_ic = lambda X: pde_model.p_0(X[:, :-1])
    p_tc = lambda X: pde_model.p_inf(X[:, :-1])
    #u_inf = lambda X: pde_model.u_analytic(X)

    # IC
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model', title="model(x,0)").heatmap(model_fn)
    plotter.add_panel('ic', title="u_0(x)").heatmap(p_ic)
    plotter.save_plot(f'{dir_name}/viz/plot_model_vs_u0.png', t_val=0.0)
    # inf
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model', title="model(x,T)").heatmap(model_fn)
    plotter.add_panel('inf', title="u_inf(x)").heatmap(p_tc)
    plotter.save_plot(f'{dir_name}/viz/plot_model_vs_u_inf.png', t_val=T)
    # anim: 
    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model', title="model(x,t)").heatmap(model_fn)
    plotter.save_animation(f'{dir_name}/viz/anim_model.gif', num_frames=30, fps=5, t_end=T, cbar="fixed")



def plot_viz(dir_name, model, pde_model, score_sde_model, args, device, model_s=None):
    d = args.d
    mode = args.mode

    options = {
        "d": d,
        "plot_dims": [0, 1],
        "fixed_dims_vals": 0.5 * torch.ones(d, device=device),
        "device": device,
        "x_start": args.L_min,
        "x_end": args.L_max,
    }

    os.makedirs(f"{dir_name}/viz/", exist_ok=True)

    if mode == "log_pde":
        _plot_log_pde(dir_name, model, pde_model, args, options)
    elif mode == "class_pde":
        _plot_class_pde(dir_name, model, pde_model, args, options)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def plot_variational_step(dir_name, model, pde_model, previous_timestep_fn, args, device, titles):
    """Save a 2x3 comparison of the previous and current variational states.

    Columns are the prescribed state at time level ``n``, the trained state at
    ``n + 1``, and their signed difference. The top row shows ``tilde_rho``
    (the optimization variable); the bottom row shows the mapped physical
    density ``rho = tilde_rho * exp(-beta V)``.
    """
    if args.d < 2:
        print("Skipping variational plots: at least two spatial dimensions are required.")
        return

    options = {
        "d": args.d,
        "plot_dims": [0, 1],
        "fixed_dims_vals": torch.zeros(args.d, device=device),
        "device": device,
        "x_start": args.L_min,
        "x_end": args.L_max,
    }
    os.makedirs(f"{dir_name}/viz", exist_ok=True)

    def values_for_plot(value, label):
        """Replace non-finite values for rendering and report model divergence."""
        finite = torch.isfinite(value)
        n_invalid = value.numel() - finite.sum().item()
        if n_invalid:
            print(
                f"Warning: {label} contains {n_invalid}/{value.numel()} non-finite values; "
                "rendering them as 0."
            )
        return torch.where(finite, value, torch.zeros_like(value))

    was_training = model.training
    model.eval()

    try:
        grid = viz.FunctionPlotter(**options)
        x = grid.X[:, :-1]
        with torch.no_grad():
            tilde_n = previous_timestep_fn(x)
            tilde_model_np1 = model(x)
            rho_n = pde_model._map_from_tilde(tilde_n, x)
            rho_model_np1 = pde_model._map_from_tilde(tilde_model_np1, x)

        tilde_n = values_for_plot(tilde_n, "given tilde rho_n")
        tilde_model_np1 = values_for_plot(tilde_model_np1, "model tilde rho_(n+1)")
        rho_n = values_for_plot(rho_n, "given rho_n")
        rho_model_np1 = values_for_plot(rho_model_np1, "model rho_(n+1)")
        rows = (
            (
                tilde_n,
                tilde_model_np1,
            ) + titles[0],
            (
                rho_n,
                rho_model_np1,
            ) + titles[1]
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for row_index, (value_n, value_np1, title_n, title_np1, title_diff) in enumerate(rows):
            difference = value_np1 - value_n
            value_min = min(value_n.min().item(), value_np1.min().item())
            value_max = max(value_n.max().item(), value_np1.max().item())
            value_levels = grid._make_levels(value_n, value_min, value_max)
            max_difference = difference.abs().max().item()
            difference_levels = grid._make_levels(difference, -max_difference, max_difference)

            contour = None
            for axis, value, title in zip(
                axes[row_index, :2],
                (value_n, value_np1),
                (title_n, title_np1),
            ):
                contour = axis.contourf(
                    grid.X_grid.cpu(), grid.Y_grid.cpu(),
                    value.detach().reshape(grid.N, grid.N).cpu(),
                    levels=value_levels, cmap="jet",
                )
                axis.set_title(title)
                axis.set_xlabel("x0")
                axis.set_ylabel("x1")
            fig.colorbar(contour, ax=axes[row_index, :2])

            difference_contour = axes[row_index, 2].contourf(
                grid.X_grid.cpu(), grid.Y_grid.cpu(),
                difference.detach().reshape(grid.N, grid.N).cpu(),
                levels=difference_levels, cmap="RdBu_r",
            )
            axes[row_index, 2].set_title(title_diff)
            axes[row_index, 2].set_xlabel("x0")
            axes[row_index, 2].set_ylabel("x1")
            fig.colorbar(difference_contour, ax=axes[row_index, 2])

        path = f"{dir_name}/viz/plot_variational_step.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved to {path}")
    finally:
        model.train(was_training)
