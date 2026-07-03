

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
args = run_utils.parse_args_with_config(
    parser, [] if "__file__" not in globals() else None
)
# FIXED (heat eq specific)
args.mode = "class_pde"
args.L_min = 0
args.L_max = 1
#args.T = 4.5
args.T = 5.0
args.f_pde_trajs = 0
args.f_pde_full_domain = 1
args.f_ic_trajs = 0
args.f_ic_full_domain = 1

args.d = 2
args.layers="64,64,64,64"
# bs and sampling
args.bs=1000
args.n_res_points=100_000
args.resampling_frequency=100
#args.prevent_resampling=True
args.prevent_resampling=False
args.one_batch_per_epoch=True
# other
#args.n_steps=10_000
args.n_steps=1000
args.n_steps_decay=args.n_steps/10
args.active_losses="pde,bc,ic"

#args.use_lbfgs = False
args.use_gradnorm = True

#args.enable_profiler = True
#args.enable_memory_tracking = True
args.enable_testing = True
args.clear_dir = True
args.output_dir = 'run_latest_test'



d = args.d  # space dims
D = d + 1   # space + time dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PREP PDE MODEL
import pde_models as pde_mod
pde_model = pde_mod.HeatEquation(d)
print(type(pde_model))


head_fun = lambda out, X: out

active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
print(f"Active losses: {active_losses}")

## testing obj
spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)
test_sampling_type = "domain_and_trajectories"
if args.enable_testing:
    testing_suite = utility.TestingSuite(d, device)
    test_sampling_type = "domain_and_trajectories"
    #termina_cond_fun = 
    #analytic_sol_fun = 
    sampling_settings = {
        "T": args.T,
        "spatial_domain": spatial_domain,
        "n_res_points": args.n_test_points,
        "bs": args.bs,
        #
        "f_pde_full_domain": 1,
        "f_pde_trajs": 0,
        "f_ic_full_domain": 1,
        "f_ic_trajs": 0,
    }
    sampling_settings["res_points"] = args.n_test_points
    testing_suite.make_test_data("testing_data.pth", test_sampling_type, lambda X: None, pde_model, sampling_settings, active_losses, device, analytic_sol_fn=pde_model.u_analytic, terminal_condition_fn=None)
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
        "f_pde_full_domain": args.f_pde_full_domain,
        "f_pde_trajs": args.f_pde_trajs,
        "f_ic_full_domain": args.f_ic_full_domain,
        "f_ic_trajs": args.f_ic_trajs,
    }


trainer, model, dir_name = main_runner.runner(args, pde_model, sampling_settings, sampling_type, testing_suite, head_fun)


import plot_results
plot_results.plot_viz(dir_name, model, pde_model, None, args, device, model_s=None)