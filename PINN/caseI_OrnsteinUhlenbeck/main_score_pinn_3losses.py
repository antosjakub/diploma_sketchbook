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

parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--lambda_norm", default=0.1, type=float, help="Weight of the ∫p dx = 1 normalization loss.")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--active_losses", default="pde,bc,ic", type=str, help="Comma-separated subset of {pde,bc,ic,norm}. 'pde' is required.")

parser.add_argument("--n_res_points", default=10_000, type=int, help="")
parser.add_argument("--n_trajs", default=1_000, type=int, help="")
parser.add_argument("--nt_steps", default=100, type=int, help="")
parser.add_argument("--T", default=1.5, type=float, help="")

parser.add_argument("--L_min", default=-4.0, type=float, help="")
parser.add_argument("--L_max", default=4.0, type=float, help="")

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
    args.output_dir = f"{label}/run_3losses_{type_sp}"
    args.clear_dir = True
    args.enable_testing = False
elif type_sp == "ll_ode":
    args.output_dir = f"{label}/run_3losses_{type_sp}"
    args.clear_dir = True
    args.enable_testing = False
    score_pde_dir_name = f"{label}/run_3losses_score_pde"
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
    layers_s = utility.layers_from_string(model_metadata["args"]["layers"])
    model_s = architecture.PINN(D, layers_s, d).to(device)
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
    model = architecture.PINN(D, layers, d).to(device)
elif type_sp == "ll_ode":
    model = architecture.PINN(D, layers, 1).to(device)
#model = torch.compile(model, mode="reduce-overhead")
#model = torch.compile(model)


active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
print(f"Active losses: {active_losses}")

# Preparation time
losses = run_utils.init_losses(("total",) + active_losses)
l2_errs = []

optimizer, scheduler = run_utils.make_optim(model, args)
loss_weighting = run_utils.make_loss_weighting(args, active_losses)
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

T = args.T
sampling_type = args.sampling_type
if sampling_type == "trajectories":
    sampling_settings = {
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "spatial_domain": torch.stack([torch.full((d,), args.L_min), torch.full((d,), args.L_max)], dim=1),
        "T": args.T,
    }
elif sampling_type == "domain":
    sampling_settings = {
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "spatial_domain": torch.stack([torch.full((d,), args.L_min), torch.full((d,), args.L_max)], dim=1),
        "T": T,
        "use_rbas": args.use_rbas,
    }

from trainers import PINN_Trainer
trainer = PINN_Trainer(
    model, optimizer, scheduler, pde_model,
    sampling_type=sampling_type, sampling_settings=sampling_settings,
    loss_weighting=loss_weighting, testing_suite=testing_suite,
    active_losses=active_losses, profiler=profiler, device=device,
)
losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
    n_steps=args.n_steps,
    n_steps_decay=args.n_steps_decay,
    resampling_frequency=args.resampling_frequency,
    testing_frequency=args.testing_frequency,
    use_sdgd=args.use_sdgd,
    sdgd_num_dims=sdgd_num_dims,
    one_batch_per_epoch = True,
)
run_utils.merge_losses(losses, losses_adam)
l2_errs += l2_errs_adam
print("\nAdam training complete!")
run_utils.print_train_duration(t1, time.time())

print("\nTraining complete!")

loss_name, l2_name = run_utils.save_run(dir_name, model, losses, l2_errs, args, head_fn=None)


import plot_results
plot_results.plot_run(
    dir_name, model, pde_model, score_sde_model, args, device,
    model_s=model_s if type_sp == "ll_ode" else None,
    losses=losses, l2_errs=l2_errs,
)
