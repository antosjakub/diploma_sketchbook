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

import architecture
import utility
import viz
import visualize_training_metrics

import pde_model_sde


def _make_head_fn(tag, mode, score_sde_model, T):
    """Rebuild the head_fn from its saved tag. None => default PINN head.

    The training-time code writes head_fns against `pde_model.s0/q0`, but those
    always forward to `score_sde_model.s0/q0` (via __getattr__ or by definition),
    so we bind directly to `score_sde_model` here — one fewer name to juggle.
    """
    if tag is None:
        return None
    if tag == "hardcoded_ic":
        if mode == "score_pde":
            return lambda nn_out, X: nn_out * X[:, -1:] / T + score_sde_model.s0(X[:, :-1])
        if mode == "ll_ode":
            return lambda nn_out, X: nn_out * X[:, -1:] / T + score_sde_model.q0(X[:, :-1])
        raise ValueError(f"Unknown mode '{mode}' for head_fn tag '{tag}'")
    raise ValueError(f"Unknown head_fn tag: {tag!r}")




# ---------------- plotting ----------------

def plot_training_metrics(dir_name, losses, l2_errs, args):
    loss_name = f'{dir_name}/training_loss'
    l2_name = f'{dir_name}/training_l2_error'
    visualize_training_metrics.plot_loss(losses, loss_name)
    if getattr(args, "enable_testing", False) and len(l2_errs) > 0:
        n_steps_log = args.testing_frequency
        n_logged_pnts = len(l2_errs)
        steps = n_steps_log * torch.linspace(1, n_logged_pnts, n_logged_pnts, dtype=torch.int)
        visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)


def _plot_score_pde(dir_name, model, pde_model, score_sde_model, args, options):
    T = args.T
    model_fn_s = viz.wrapp_model(model)
    s_ic = lambda X: pde_model.s0(X[:, :-1])
    p_ic = lambda X: pde_model.p0(X[:, :-1])
    p_inf = lambda X: pde_model.p_inf(X[:, :-1])

    if args.enable_testing:
        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_s', title="model_s(x,t)").quiver(model_fn_s)
        plotter.add_panel('s_analytic', title="s_analytic(x,t)").quiver(score_sde_model.s_analytic)
        plotter.add_panel('err', title="err").quiver(lambda X: model_fn_s(X) - score_sde_model.s_analytic(X))
        plotter.save_animation(f'{dir_name}/viz/anim_model_s_vs_s_analytic.gif', num_frames=30, fps=5, t_end=T)

    plotter_ic = viz.FunctionPlotter(**options)
    plotter_ic.add_panel('nn', rf"$s_\theta(x,0)$").quiver(model_fn_s)
    plotter_ic.add_panel('ic', rf"$s_0(x)$").quiver(s_ic)
    plotter_ic.add_panel('err', "err(x)").quiver(lambda X: model_fn_s(X) - s_ic(X))
    plotter_ic.save_plot(
        f'{dir_name}/viz/plot_s_nn_vs_s0.png',
        t_val=0.0, cbar={"nn": "linked:ic", "err": "linked:ic"},
    )

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('nn', "s_nn(x,t)").quiver(model_fn_s)
    plotter.save_animation(f'{dir_name}/viz/anim_s_nn_fixed.gif', cbar='fixed', num_frames=30, fps=5, t_end=T)
    plotter.save_animation(f'{dir_name}/viz/anim_s_nn_dynamic.gif', cbar='dynamic', num_frames=30, fps=5, t_end=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('ic', title="p_0(x)").heatmap(p_ic)
    plotter.add_panel('final', title="p_inf(x)").heatmap(p_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_p0_vs_p_inf.png', t_val=0.0)


def _plot_ll_ode(dir_name, model, model_s, pde_model, score_sde_model, args, options):
    T = args.T
    model_fn_q = viz.wrapp_model(model)
    model_fn_p = lambda X: torch.exp(model_fn_q(X))
    model_fn_s = viz.wrapp_model(model_s)
    q_ic = lambda X: pde_model.q0(X[:, :-1])
    q_inf = lambda X: torch.log(pde_model.p_inf(X[:, :-1]))
    p_ic = lambda X: pde_model.p0(X[:, :-1])
    p_inf = lambda X: pde_model.p_inf(X[:, :-1])

    if args.enable_testing:
        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_q', title="model_q(x,t)").heatmap(model_fn_q)
        plotter.add_panel('q_analytic', title="q_analytic(x,t)").heatmap(score_sde_model.q_analytic)
        plotter.add_panel('err', title="err").heatmap(lambda X: model_fn_q(X) - score_sde_model.q_analytic(X))
        plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q_analytic.png', t_val=0.234)
        plotter.save_animation(f'{dir_name}/viz/anim_model_q_vs_q_analytic.gif', num_frames=30, fps=5, t_end=T)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_p', title="model_p(x,t)").heatmap(model_fn_p)
        plotter.add_panel('p_analytic', title="p_analytic(x,t)").heatmap(score_sde_model.p_analytic)
        plotter.add_panel('err', title="err").heatmap(lambda X: model_fn_p(X) - score_sde_model.p_analytic(X))
        plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p_analytic.png', t_val=0.234)
        plotter.save_animation(f'{dir_name}/viz/anim_model_p_vs_p_analytic.gif', num_frames=30, fps=5, t_end=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,0)").heatmap(model_fn_q)
    plotter.add_panel('q_ic', title="q_0(x)").heatmap(q_ic)
    plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q0.png', t_val=0.0)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,0) = exp(model_q(x,0))").heatmap(model_fn_p)
    plotter.add_panel('p_ic', title="p_0(x)").heatmap(p_ic)
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p0.png', t_val=0.0)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,T)").heatmap(model_fn_q)
    plotter.add_panel('q_inf', title="q_inf(x)").heatmap(q_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q_inf.png', t_val=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,T)").heatmap(model_fn_p)
    plotter.add_panel('p_inf', title="p_inf(x)").heatmap(p_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p_inf.png', t_val=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_q', title="model_q(x,t)").heatmap(model_fn_q)
    plotter.save_animation(f'{dir_name}/viz/anim_model_q.gif', num_frames=30, fps=5, t_end=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="model_p(x,t) = exp(model_q(x,t))").heatmap(model_fn_p)
    plotter.save_animation(f'{dir_name}/viz/anim_model_p.gif', num_frames=30, fps=5, t_end=T)

    plotter = viz.FunctionPlotter(**options)
    p = plotter.add_panel('sq', title="model_s & model_q")
    p.heatmap(model_fn_q)
    p.quiver(model_fn_s, color='k')
    p = plotter.add_panel('sp', title="model_s & model_p")
    p.heatmap(model_fn_p)
    p.quiver(model_fn_s, color='k')
    plotter.save_animation(f'{dir_name}/viz/anim_model_sq_sp.gif', num_frames=30, fps=5, t_end=T)


def plot_viz(dir_name, model, pde_model, score_sde_model, args, device, model_s=None):
    d = args.d
    type_sp = args.mode

    options = {
        "d": d,
        "plot_dims": [0, 1],
        "fixed_dims_vals": 0.5 * torch.ones(d),
        "device": device,
        "x_start": args.L_min,
        "x_end": args.L_max,
    }

    os.makedirs(f"{dir_name}/viz/", exist_ok=True)

    if type_sp == "score_pde":
        _plot_score_pde(dir_name, model, pde_model, score_sde_model, args, options)
    elif type_sp == "ll_ode":
        if model_s is None:
            raise ValueError("ll_ode mode requires model_s")
        _plot_ll_ode(dir_name, model, model_s, pde_model, score_sde_model, args, options)
    else:
        raise ValueError(f"Unknown mode: {type_sp}")


def plot_run(dir_name, model, pde_model, score_sde_model, args, device,
             model_s=None, losses=None, l2_errs=None):
    """Top-level: training metrics + viz plots. `losses`/`l2_errs` optional."""
    if losses is not None:
        plot_training_metrics(dir_name, losses, l2_errs if l2_errs is not None else [], args)
    plot_viz(dir_name, model, pde_model, score_sde_model, args, device, model_s=model_s)


# ---------------- loading saved runs ----------------

def _build_pinn(D, layers, out_dim, device, head_fn=None):
    if head_fn is not None:
        return architecture.PINN(D, layers, out_dim, head_fn=head_fn).to(device)
    return architecture.PINN(D, layers, out_dim).to(device)


def load_run(dir_name, device=None):
    """Reconstruct everything needed to replot from a saved run directory.

    Returns: (model, pde_model, score_sde_model, args, losses, l2_errs, device, model_s)
    `model_s` is None for score_pde mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_metadata = utility.json_load(f"{dir_name}/model_metadata.json")
    pde_metadata = utility.json_load(f"{dir_name}/pde_metadata.json")
    args = argparse.Namespace(**model_metadata["args"])

    d = args.d
    D = d + 1
    layers = utility.layers_from_string(args.layers)
    type_sp = args.mode
    head_fn_tag = model_metadata["head_fn"]

    score_sde_cls = utility.get_module_classes(pde_model_sde)[pde_metadata["pde_class"]]
    score_sde_model = score_sde_cls(d=d)

    model_s = None
    if type_sp == "score_pde":
        pde_model = score_sde_model.Score_PDE(score_sde_model)
        pde_model.load_pde_metadata(pde_metadata)
    elif type_sp == "ll_ode":
        # Parent score_pde run is identified by args.starting_model, not by naming convention.
        parent_dir = os.path.dirname(args.starting_model)
        parent_metadata = utility.json_load(f"{parent_dir}/model_metadata.json")
        parent_args = argparse.Namespace(**parent_metadata["args"])
        parent_head_fn_tag = parent_metadata["head_fn"]
        layers_s = utility.layers_from_string(parent_args.layers)

        head_fn_s = _make_head_fn(parent_head_fn_tag, "score_pde", score_sde_model, parent_args.T)
        model_s = _build_pinn(D, layers_s, d, device, head_fn=head_fn_s)
        model_s.load_state_dict(torch.load(f"{parent_dir}/model.pth", weights_only=True, map_location=device))
        model_s.eval()

        pde_model = score_sde_model.LL_ODE(score_sde_model, model_s)
        pde_model.load_pde_metadata(pde_metadata)
    else:
        raise ValueError(f"Unknown mode: {type_sp}")

    out_dim = d if type_sp == "score_pde" else 1
    head_fn = _make_head_fn(head_fn_tag, type_sp, score_sde_model, args.T)

    model = _build_pinn(D, layers, out_dim, device, head_fn=head_fn)
    model.load_state_dict(torch.load(f"{dir_name}/model.pth", weights_only=True, map_location=device))
    model.eval()

    losses = torch.load(f"{dir_name}/training_loss.pth", weights_only=True)
    l2_errs = torch.load(f"{dir_name}/training_l2_error.pth", weights_only=True)

    return model, pde_model, score_sde_model, args, losses, l2_errs, device, model_s


def replot(dir_name, device=None):
    """Load a saved run directory and regenerate all plots."""
    model, pde_model, score_sde_model, args, losses, l2_errs, device, model_s = load_run(dir_name, device)
    plot_run(
        dir_name, model, pde_model, score_sde_model, args, device,
        model_s=model_s, losses=losses, l2_errs=l2_errs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate plots from a saved run directory.")
    parser.add_argument("dir_name", type=str,
                        help="Run directory, e.g. 'gauss/run_3losses_score_pde' or 'laplace/run_hardcoded_ll_ode'")
    cli_args = parser.parse_args()
    dir_name = cli_args.dir_name.rstrip('/')
    replot(dir_name)

# run as:
# python plot_results.py gauss/run_3losses_score_pde