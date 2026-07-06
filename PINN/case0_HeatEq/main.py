

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

import utility
import run_utils
import main_runner


# ARGS:
import argparse
parser = argparse.ArgumentParser()
main_runner.add_common_args(parser)

parser.set_defaults(
    # FIXED (heat eq specific)
    mode="class_pde",
    L_min=0,
    L_max=1,
    T=5.0,
    f_pde_trajs=0,
    f_pde_full_domain=1,
    f_ic_trajs=0,
    f_ic_full_domain=1,
    glorot_init=True,
    # nice defaults
    d=2,
    layers="64,64,64,64",
    active_losses="pde,ic",
    lambda_pde=1.0,
    lambda_ic=1.0,
    use_hard_constrains=True,
    #
    bs=512,
    n_res_points=1024,
    one_batch_per_epoch=False,
    prevent_resampling=True,
    resampling_frequency=1,
    #
    n_steps=299,
    n_steps_decay=1000,
    lambda_strategy="gradnorm_adapt",
    #lambda_strategy="fixed",
    time_strategy="causal_loss",
    #time_strategy="time_adapt_sampl",
    #use_lbfgs=True,
    #enable_profiler = True,
    enable_memory_tracking = True,
    enable_testing = True,
    clear_dir = True,
    output_dir = 'run_latest',
    #initial_model="run_latest/model.pth"
    #use_rbas=True
)


args = run_utils.parse_args_with_config(
    parser, [] if "__file__" not in globals() else None
)

if args.use_hard_constrains:
    #head_fun = lambda out, X: out * torch.prod(X[:,:-1]*(1-X[:,:-1]), dim=1, keepdim=True)
    head_fun = lambda out, X: out * torch.prod(torch.sin(torch.pi*X[:,:-1]), dim=1, keepdim=True)
else:
    head_fun = lambda out, X: out



d = args.d  # space dims
D = d + 1   # space + time dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_name = run_utils.setup_dir(args)

### PREP PDE MODEL
import pde_models as pde_mod
pde_model = pde_mod.HeatEquation(d)
print(type(pde_model))

active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
print(f"Active losses: {active_losses}")

## testing obj
spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)
test_sampling_type = "domain_and_trajectories"
if args.enable_testing:
    # pass in a distribtion for sampling the center more heavily ? - for L_inf error est.
    # do not do the MSE loss reporting
    testing_suite = utility.TestingSuiteHeatEq(d, device, args.bs)
    test_sampling_type = "domain_and_trajectories"
    sampling_settings = {
        "T": args.T,
        "spatial_domain": spatial_domain,
        "n_res_points": args.n_test_points,
        "bs": args.n_test_points,
        #
        "f_pde_full_domain": 1,
        "f_pde_trajs": 0,
        "f_ic_full_domain": 1,
        "f_ic_trajs": 0,
    }
    sampling_settings["res_points"] = args.n_test_points
    testing_suite.make_test_data(lambda X: None, pde_model, test_sampling_type, sampling_settings, pde_model.u_analytic, f"{dir_name}/testing_data.pth", device)
    print(f"Testing suite ready ({args.n_test_points} points.")
else:
    testing_suite = None


sampling_type = args.sampling_type
spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)

sampling_settings_base = {
    "T": args.T,
    "spatial_domain": spatial_domain,
    "n_res_points": args.n_res_points,
    "bs": args.bs,
}
print(sampling_settings_base)

if sampling_type == "trajectories":
    sampling_settings = sampling_settings_base | {
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
    }
elif sampling_type == "domain":
    sampling_settings = sampling_settings_base | {
        "use_rbas": args.use_rbas,
    }
elif sampling_type == "domain_and_trajectories":
    sampling_settings = sampling_settings_base | {
        "n_trajs": args.n_trajs,
        "nt_steps": args.nt_steps,
        "use_rbas": args.use_rbas,
        "rbas_chunk_size": args.rbas_chunk_size,
        "f_pde_full_domain": args.f_pde_full_domain,
        "f_pde_trajs": args.f_pde_trajs,
        "f_ic_full_domain": args.f_ic_full_domain,
        "f_ic_trajs": args.f_ic_trajs,
    }


trainer, model, dir_name = main_runner.runner(args, dir_name, pde_model, sampling_settings, sampling_type, testing_suite, head_fun)


import plot_results
plot_results.plot_viz(dir_name, model, pde_model, None, args, device, model_s=None)