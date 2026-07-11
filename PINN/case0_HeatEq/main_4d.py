

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
    d=4,
    active_losses="pde,ic",
    lambda_pde=1.0,
    lambda_ic=0.1,
    lambda_strategy="fixed",
    use_hard_constrains=True,
    one_batch_per_epoch=True,
    #

    # TD:
    #layers="64,64,64,64",
    layers="128,128,128,128",


    bs=2048,
    resampling_frequency=100,

    n_res_points=204800,
    prevent_resampling=False,

    #n_res_points=1024,
    #prevent_resampling=True,


    #n_steps=10_000,
    #n_steps_decay=400, #25x
    #logging_frequency=40,

    n_steps=20_000,
    n_steps_decay=666, #30x
    logging_frequency=80,


    #lambda_strategy="gradnorm_adapt",
    #use_rbas=True
    use_rbas = False,
    rbas_k = 1.0,
    rbas_c = 0.0,

    #crit_loss_val=1e-6,
    #time_strategy="causal_loss",
    #time_strategy="time_adapt_sampl",
    time_strategy = "none",
    enable_profiler = False,
    enable_memory_tracking = True,
    enable_testing = True,
    n_test_points=100_000,
    n_test_chunk_size=100_000,

    clear_dir = True,
    use_lbfgs = False,
    #starting_model=f"run_adam/model.pth",
    output_dir = 'run_adam',
    #lr=0.9**6 * 1e-3
)


args = run_utils.parse_args_with_config(
    parser, [] if "__file__" not in globals() else None
)

import time
t1 = time.time()

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
    testing_suite = utility.TestingSuiteHeatEq(d, device, args.n_test_chunk_size)
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
    print(f"Testing suite ready: n_points={args.n_test_points}, n_chunk_size={args.n_test_chunk_size}.")
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
        "rbas_k": args.rbas_k,
        "rbas_c": args.rbas_c,
        "f_pde_full_domain": args.f_pde_full_domain,
        "f_pde_trajs": args.f_pde_trajs,
        "f_ic_full_domain": args.f_ic_full_domain,
        "f_ic_trajs": args.f_ic_trajs,
    }

t2 = time.time()
print("---- Prep works inside main.py", t2-t1)

trainer, model, dir_name = main_runner.runner(args, dir_name, pde_model, sampling_settings, sampling_type, testing_suite, head_fun)

t1 = time.time()
import plot_results
plot_results.plot_viz(dir_name, model, pde_model, None, args, device, model_s=None)
t2 = time.time()
print("---- Time for saving viz", t2-t1)