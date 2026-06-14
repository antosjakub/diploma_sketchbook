import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, type=str, help="Path to a JSON file with parameter values.")
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=4, type=int, help="Number of spatial dimensions.")
#parser.add_argument("--layers", default="256,256,256,256", type=str, help="")
#parser.add_argument("--layers", default="64,64,64,64", type=str, help="")
parser.add_argument("--layers", default="128,128,128,128", type=str, help="")
parser.add_argument("--n_steps", default=9_999, type=int, help="")
parser.add_argument("--n_steps_decay", default=1_600, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--bs", default=1000, type=int, help="")

parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--lambda_norm", default=0.1, type=float, help="Weight of the integral p dx = 1 normalization loss.")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--active_losses", default="pde,ic", type=str, help="Comma-separated subset of {pde,bc,ic,norm}. 'pde' is required.")
parser.add_argument("--grad_clip_norm", default=None, type=float, help="Max-norm gradient clipping for the train step. None disables it.")

parser.add_argument("--n_res_points", default=10_000, type=int, help="")
parser.add_argument("--n_trajs", default=1_000, type=int, help="")
parser.add_argument("--nt_steps", default=100, type=int, help="")
parser.add_argument("--T", default=3.5, type=float, help="")

parser.add_argument("--L_min", default=-5.0, type=float, help="")
parser.add_argument("--L_max", default=5.0, type=float, help="")

parser.add_argument("--n_test_points", default=10_000, type=int, help="Number of test points for the testing suite.")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")

parser.add_argument("--resampling_frequency", default=100, type=int, help="")

parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# smart Defaults
parser.add_argument("--output_dir", default="run_latest_vanilla", type=str, help="")

parser.add_argument("--mode", default="q_pde", type=str, help="score_pde, ll_ode")
parser.add_argument("--ic_type", default="laplace", type=str, help="gauss, cauchy, laplace")
parser.add_argument("--sampling_type", default="domain_and_trajectories", type=str, help="trajectories, domain")
parser.add_argument("--f_pde_full_domain", default=1, type=int, help="")
parser.add_argument("--f_pde_trajs", default=1, type=int, help="")
parser.add_argument("--f_ic_full_domain", default=1, type=int, help="")
parser.add_argument("--f_ic_trajs", default=1, type=int, help="")

#
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")

parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")
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


import architecture, utility
import run_utils


# Main execution

args = run_utils.parse_args_with_config(
    parser, [] if "__file__" not in globals() else None
)

d = args.d  # space dims
D = d + 1   # space + time dims
layers = utility.layers_from_string(args.layers)
print(f"\n{'='*60}")
print(f"Training vanilla-PINN for {d}D PDE")
print(f"Domain: [0,1]^{d} x [0,1]")
print(f"{'='*60}\n")


dir_name, device = run_utils.setup_run(args)
run_utils.save_input_config(dir_name, args)


### PREP PDE MODEL
gamma = torch.tensor([1.3, 2.5, 2.1, 1.6, 3.1, 1.8, 2.7, 1.9, 2.3, 2.9])[:d]
import pde_model_sde
if args.ic_type == "gauss":
    score_sde_model = pde_model_sde.Gaussian_OU(d=d, gamma=gamma)
elif args.ic_type == "cauchy":
    score_sde_model = pde_model_sde.Cauchy_OU(d=d, gamma=gamma)
elif args.ic_type == "laplace":
    score_sde_model = pde_model_sde.Laplace_OU(d=d, gamma=gamma)

pde_model = score_sde_model.q_PDE(score_sde_model)

print(type(pde_model))
print(pde_model.gaussian_obj.gamma)
print(pde_model.Sigma)
pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')
print()


# Select the model architecture
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
    analytic_fn = score_sde_model.q_analytic
    testing_suite = utility.ScorePINNTestingSuite(d, analytic_fn)
    testing_suite.make_test_data(score_sde_model, args.n_test_points)
    print(f"Testing suite ready ({args.n_test_points} points.")
else:
    testing_suite = None

T = args.T
sampling_type = args.sampling_type
if sampling_type == "trajectories":
    sampling_settings = {
        "T": args.T,
        "spatial_domain": torch.stack([torch.full((d,), args.L_min), torch.full((d,), args.L_max)], dim=1),
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
    }
elif sampling_type == "domain":
    sampling_settings = {
        "T": T,
        "spatial_domain": torch.stack([torch.full((d,), args.L_min), torch.full((d,), args.L_max)], dim=1),
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "use_rbas": args.use_rbas,
    }
elif sampling_type == "domain_and_trajectories":
    sampling_settings = {
        "T": args.T,
        "spatial_domain": torch.stack([torch.full((d,), args.L_min), torch.full((d,), args.L_max)], dim=1),
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
        "use_rbas": args.use_rbas,
        "f_pde_full_domain": args.f_pde_full_domain,
        "f_pde_trajs": args.f_pde_trajs,
        "f_ic_full_domain": args.f_ic_full_domain,
        "f_ic_trajs": args.f_ic_trajs,
    }

from trainers import PINN_Trainer
trainer = PINN_Trainer(
    model, optimizer, scheduler, pde_model,
    sampling_type=sampling_type, sampling_settings=sampling_settings,
    loss_weighting=loss_weighting, testing_suite=testing_suite,
    active_losses=active_losses, profiler=profiler, device=device,
    dir_name=dir_name,
    grad_clip_norm=args.grad_clip_norm,
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
utility.print_duration_h_m_s(t1, time.time(), "Adam training")

print("\nTraining complete!")

loss_name, l2_name = run_utils.save_run(dir_name, model, losses, l2_errs, args, head_fn=None, loss_weighting=loss_weighting if args.n_steps > 0 else None) 


import plot_results
plot_results.plot_run(
    dir_name, model, pde_model, score_sde_model, args, device,
    model_s=None,
    losses=losses, l2_errs=l2_errs,
)
