import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="148,148,148", type=str, help="")
parser.add_argument("--n_steps", default=1950, type=int, help="")
parser.add_argument("--n_steps_decay", default=5_000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--bs", default=1000, type=int, help="")

parser.add_argument("--n_res_points", default=10_000, type=int, help="")
parser.add_argument("--n_trajs", default=1_000, type=int, help="")
parser.add_argument("--nt_steps", default=100, type=int, help="")
parser.add_argument("--T", default=1.5, type=float, help="")

parser.add_argument("--L", default=4.0, type=float, help="")

parser.add_argument("--n_test_points", default=10_000, type=int, help="Number of test points for the testing suite.")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")

parser.add_argument("--resampling_frequency", default=5_000, type=int, help="")

parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# smart Defaults
parser.add_argument("--output_dir", default="run_score_pinn_latest/", type=str, help="")
parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")

parser.add_argument("--mode", default="score_pde", type=str, help="score_pde, ll_ode")
parser.add_argument("--ic_type", default="gauss", type=str, help="gauss, cauchy, laplace")
parser.add_argument("--sampling_type", default="trajectories", type=str, help="trajectories, domain")
#
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")


"""
1. sampling to obtain X
    - sample x_0 ~ p_0
    - collect into X_ic
    - use x_0 to evolve the trajectories via the SDE
    - select random points from the trajectories
    - collect into X_pde
    - sample once, split into batches, resample every once in a while
2. loss
    - just normal loss with two lambda weights
3. residual
    -
"""

import torch

import os, sys
src_dir = os.path.join(os.path.dirname(__file__), '../src/')
sys.path.append(src_dir)


import sampling, loss, architecture, utility
import run_utils


# Main execution

args = parser.parse_args([] if "__file__" not in globals() else None)

d = args.d  # space dims
D = d + 1   # space + time dims
layers = utility.layers_from_string(args.layers)
print(f"\n{'='*60}")
print(f"Training PINN for {d}D PDE")
print(f"Domain: [0,1]^{d} x [0,1]")
print(f"{'='*60}\n")

type_sp = args.mode
print(f"Training Score-PINN, type: '{type_sp}'\n")

label = args.ic_type
if type_sp == "score_pde":
    args.output_dir = f"run_{label}_{type_sp}"
    args.clear_dir = True
    args.enable_testing = False
elif type_sp == "ll_ode":
    args.output_dir = f"run_{label}_{type_sp}"
    args.clear_dir = True
    args.enable_testing = False
    score_pde_dir_name = f"run_{label}_score_pde"
    args.starting_model = f"{score_pde_dir_name}/model.pth"
else:
    raise NameError("Incorrect mode specified.")

dir_name, device = run_utils.setup_run(args)


### PREP PDE MODEL
import pde_model_sde
if label == "gauss":
    score_sde_model = pde_model_sde.Gaussian_OU(d=d)
elif label == "cauchy":
    score_sde_model = pde_model_sde.Cauchy_OU(d=d)
elif label == "laplace":
    score_sde_model = pde_model_sde.Laplace_OU(d=d)

# Score PDE
if type_sp == "score_pde":
    pde_model = score_sde_model.Score_PDE(score_sde_model)
## LL ODE
elif type_sp == "ll_ode":
    model_metadata = utility.json_load(f'{score_pde_dir_name}/model_metadata.json')
    head_fn = lambda nn_out, X: nn_out * X[:,-1:] + pde_model.s0(X[:,:-1])
    layers_s = utility.layers_from_string(model_metadata["args"]["layers"])
    model_s = architecture.PINN(D, layers_s, d, head_fn=head_fn).to(device)
    print(f"Loading in trained score pde model: '{args.starting_model}'")
    model_s.load_state_dict(torch.load(args.starting_model, weights_only=True))
    model_s.eval()
    pde_model = score_sde_model.LL_ODE(score_sde_model, model_s)
    print(f"Loading in score pde model parameters: '{score_pde_dir_name}/pde_metadata.json'")
    pde_model.load_pde_metadata(utility.json_load(f'{score_pde_dir_name}/pde_metadata.json'))

print(type(pde_model))
print(pde_model.gaussian_obj.gamma)
print(pde_model.Sigma)
pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')
print()


# Select the model architecture
if type_sp == "score_pde":
    head_fn = lambda nn_out, X: nn_out * X[:,-1:] + pde_model.s0(X[:,:-1])
    model = architecture.PINN(D, layers, d, head_fn=head_fn).to(device)
elif type_sp == "ll_ode":
    head_fn = lambda nn_out, X: nn_out * X[:,-1:] + pde_model.q0(X[:,:-1])
    model = architecture.PINN(D, layers, 1, head_fn=head_fn).to(device)
#model = torch.compile(model, mode="reduce-overhead")
#model = torch.compile(model)


active_losses = ("pde",)
print(f"Active losses: {active_losses}")

# Preparation time
losses = run_utils.init_losses(("total",) + active_losses)
l2_errs = []

optimizer, scheduler = run_utils.make_optim(model, args)
loss_weighting = loss.ConstantWeights(weights=[1.0])
profiler = run_utils.make_profiler(dir_name, args)

sdgd_num_dims = args.sdgd_num_dims if args.sdgd_num_dims is not None else d
if args.use_sdgd:
    print(f"Using SDGD with {sdgd_num_dims} dimensions (d={d})")
else:
    print(f"Using regular Adam training.")

import time
t1 = time.time()
if args.enable_testing:
    if type_sp == "score_pde":
        analytic_fn = score_sde_model.s_analytic
    elif type_sp == "ll_ode":
        analytic_fn = score_sde_model.q_analytic
    testing_suite = utility.ScorePINNTestingSuite(d, analytic_fn)
    testing_suite.make_test_data(score_sde_model, args.n_test_points)
    print(f"Testing suite ready ({args.n_test_points} points, mode='{type_sp}').")
else:
    testing_suite = None

L = args.L
T = args.T
sampling_type = args.sampling_type
if sampling_type == "trajectories":
    sampling_settings = {
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "spatial_domain": torch.stack([torch.full((d,), -L), torch.full((d,), L)], dim=1),
        "T": args.T,
    }
elif sampling_type == "domain":
    sampling_settings = {
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "spatial_domain": torch.stack([torch.full((d,), -L), torch.full((d,), L)], dim=1),
        "T": T,
        "use_rbas": args.use_rbas,
    }

from trainers import PINN_Trainer_1k
trainer = PINN_Trainer_1k(
    model, optimizer, scheduler, pde_model,
    sampling_type=sampling_type, sampling_settings=sampling_settings,
    testing_suite=testing_suite, profiler=profiler, device=device,
)
losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
    n_steps=args.n_steps,
    n_steps_decay=args.n_steps_decay,
    resampling_frequency=args.resampling_frequency,
    testing_frequency=args.testing_frequency,
    use_sdgd=args.use_sdgd,
    sdgd_num_dims=sdgd_num_dims,
)
run_utils.merge_losses(losses, losses_adam)
l2_errs += l2_errs_adam
print("\nAdam training complete!")
run_utils.print_train_duration(t1, time.time())

print("\nTraining complete!")

loss_name, l2_name = run_utils.save_run(dir_name, model, losses, l2_errs, args)


# Plot results
import visualize_training_metrics
visualize_training_metrics.plot_loss(losses, loss_name)
if args.enable_testing:
    n_steps_log = args.testing_frequency
    n_logged_pnts = len(l2_errs)
    steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)
    visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)


import viz
p_ic = lambda X: pde_model.p0(X[:,:-1])
p_inf = lambda X: pde_model.p_inf(X[:,:-1])

options = {
    "d": d,
    "plot_dims": [0,1],
    "fixed_dims_vals": 0.5*torch.ones(d),
    "device": device,
    "x_start": -L,
    "x_end": L,
}


import os
os.makedirs(f"{dir_name}/viz/", exist_ok=True)
if type_sp == "score_pde":
    model_fn_s = viz.wrapp_model(model)
    s_ic = lambda X: pde_model.s0(X[:,:-1])

    if args.enable_testing:
        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_s', title="model_s(x,t)").quiver(model_fn_s)
        plotter.add_panel('s_analytic', title="s_analytic(x,t)").quiver(score_sde_model.s_analytic)
        plotter.add_panel('err', title="err").quiver(lambda X: model_fn_s(X) - score_sde_model.s_analytic(X))
        plotter.save_animation(f'{dir_name}/viz/anim_model_s_vs_s_analytic.gif', num_frames=30, fps=5, t_end=T)

    plotter_ic = viz.FunctionPlotter(**options)
    plotter_ic.add_panel('nn', rf"s_\theta(x,0)").quiver(model_fn_s)
    plotter_ic.add_panel('ic', "s_0(x)").quiver(s_ic)
    plotter_ic.add_panel('err', "err(x)").quiver(lambda X: model_fn_s(X) - s_ic(X))
    plotter_ic.save_plot(f'{dir_name}/viz/plot_s_nn_vs_s0.png', t_val=0.0, cbar={"nn": "linked:ic", "err": "linked:ic"})

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('nn', "s_nn(x,t)").quiver(model_fn_s)
    plotter.save_animation(f'{dir_name}/viz/anim_s_nn_fixed.gif', cbar='fixed', num_frames=30, fps=5, t_end=T)
    plotter.save_animation(f'{dir_name}/viz/anim_s_nn_dynamic.gif', cbar='dynamic', num_frames=30, fps=5, t_end=T)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('ic', title="p_0(x)").heatmap(p_ic)
    plotter.add_panel('final', title="p_inf(x)").heatmap(p_inf)
    plotter.save_plot(f'{dir_name}/viz/plot_p0_vs_p_inf.png', t_val=0.0)

elif type_sp == "ll_ode":
    model_fn_q = viz.wrapp_model(model)
    model_fn_p = lambda X: torch.exp(model_fn_q(X))
    model_fn_s = viz.wrapp_model(model_s)
    q_ic = lambda X: pde_model.q0(X[:,:-1])
    q_inf = lambda X: torch.log(pde_model.p_inf(X[:,:-1]))

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
