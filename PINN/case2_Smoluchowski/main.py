

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
    # FIXED (smol eq specific)
    L_min=-2.5,
    L_max=2.5,
    T=2.0,
    f_pde_trajs=0,
    f_pde_full_domain=1,
    f_ic_trajs=0,
    f_ic_full_domain=1,
    # defaults
    glorot_init=True,
    one_batch_per_epoch=True,

    use_hard_constrains=True,
    mode="class_pde", #TD
    #mode="log_pde",

    #act_fn='',
    act_fn='tanh',
    #act_fn='silu',
    lr=1e-3,

    #active_losses="pde,ic,bc,norm", #TD
    active_losses="pde,ic", #TD
    #lambda_pde=1.0,
    #lambda_ic=1000.0,
    #lambda_norm=0.1,
    #lambda_bc=10.0,

    lambda_pde=10.0,
    lambda_ic=200.0,
    ##lambda_norm=0.1,
    lambda_norm=0.05,
    lambda_bc=100.0,

    #lambda_pde=100.0,
    #lambda_ic=1000.0,
    #lambda_norm=1.0,
    #lambda_bc=1000.0,
    bc_type='neu',
    lambda_strategy="fixed",
    #lambda_strategy="gradnorm_adapt",
    #gradnorm_update_freq=50,

    d=2,
    #layers="128,128,128,128",
    layers="96,96,96,96",
    #layers="64,64,64,64",

    bs=1024,
    resampling_frequency=100,
    n_res_points=102400,
    prevent_resampling=False,

    bs_norm=2048,
    #bs_norm=1024,
    n_loss_norm_slices=6,
    test_norm_slices="0.0, 0.5, 1.0, 2.0",
    #test_norm_slices="0.0, 0.15, 0.3",

    # 10x more?
    #n_trajs=1024,
    #nt_steps=100,

    #n_steps=20_000,
    #n_steps_decay=800, # 25x decay
    #logging_frequency=100,

    #n_steps=100,
    #n_steps_decay=50,
    #logging_frequency=5,
    #use_lbfgs=True,
    #output_dir = 'run_adam_lbfgs',
    #time_strategy = "none",
    #starting_model=f"run_adam_time_adapt_resample_best/model.pth",


    n_steps=10_000,
    n_steps_decay=400, # 25x decay
    logging_frequency=50,
    use_lbfgs=False,
    output_dir = 'run_adam_test',
    #time_strategy="time_adapt_sampl",
    time_strategy="none",
    starting_model=None,
    #starting_model=f"run_adam_time_adapt_resample_best/model.pth",

    #time_strategy="causal_loss",
    t_discr = "0.0, 0.25, 0.5, 0.75, 2.0",

    enable_testing = True, #TD
    n_test_points=100_000, # for ic, tc, and each time slice
    n_test_chunk_size=100_000,

    ##########################
    use_sdgd=False,
    sdgd_num_dims=2,

    use_rbas = False,
    #use_rbas = True,
    rbas_k = 2.0,
    rbas_c = 0.0,
    rbas_chunk_size = 2048,

    #crit_loss_val=1e-6,
    enable_profiler = False,
    enable_memory_tracking = True,

    clear_dir = True,
)


args = run_utils.parse_args_with_config(
    parser, [] if "__file__" not in globals() else None
)

import time
t1 = time.time()


d = args.d  # space dims
D = d + 1   # space + time dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_name = run_utils.setup_dir(args)

### PREP PDE MODEL
#a = 0.7 + 0.5*torch.rand(d)
a = torch.tensor([0.8067, 1.0391, 1.0101, 0.7722, 1.0593, 1.0706, 0.9805, 0.9893, 0.9514, 1.1419])[:d]
import pde_models_sde as pde_mod
pde_model_base = pde_mod.SmoluchowskiDoubleWell(d=d, beta=1.0, a=a, Z=9.037)
if args.mode == "log_pde":
    pde_model = pde_model_base.LogPDE(pde_model_base, args.bc_type)
elif args.mode == 'class_pde':
    pde_model = pde_model_base.ClassPDE(pde_model_base, args.bc_type)
else:
    raise NameError
print(type(pde_model))


if args.use_hard_constrains:
    #head_fun = lambda out, X: out * torch.prod(X[:,:-1]*(1-X[:,:-1]), dim=1, keepdim=True)
    #head_fun = lambda out, X: out * torch.exp(-pde_model.beta * pde_model.V(X[:,:-1]))
    #head_fun = lambda out, X: out * torch.exp((X[:,:-1]**2).sum(dim=1,keepdim=True))
    head_fun = lambda out, X: out*pde_model.p_0(X[:,:-1])
else:
    #head_fun = lambda out, X: out
    head_fun = lambda out, X: out
    #head_out = torch.nn.Softplus()
    #head_fun = lambda out, X: head_out(out)


active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
print(f"Active losses: {active_losses}")

## testing obj
spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)
test_sampling_type = "domain_and_trajectories"
if args.enable_testing:
    # pass in a distribtion for sampling the center more heavily ? - for L_inf error est.
    # do not do the MSE loss reporting
    #testing_suite.testing_terms
    import test
    testing_suite = test.TestingSuiteFP(d, device, args.n_test_chunk_size, utility.floats_from_string_list(args.test_norm_slices))
    test_sampling_type = "domain_and_trajectories"
    sampling_settings = {
        "T": args.T,
        "spatial_domain": spatial_domain,
        "n_res_points": args.n_test_points,
        #
        #"f_pde_full_domain": 1,
        #"f_pde_trajs": 1,
        "f_ic_full_domain": 1,
        "f_ic_trajs": 1,
    }
    testing_suite.make_test_data(lambda X: None, pde_model, test_sampling_type, sampling_settings, pde_model.p_0, pde_model.p_inf, f"{dir_name}/testing_data.pth", device)
    print(f"Testing suite ready: n_points={args.n_test_points}, n_chunk_size={args.n_test_chunk_size}.")
    #sfsdf
else:
    testing_suite = None


sampling_type = args.sampling_type
spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)

sampling_settings_base = {
    #"T": 1.1*args.T,
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
        "n_loss_norm_slices": args.n_loss_norm_slices,
        "bs_norm": args.bs_norm,
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