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
