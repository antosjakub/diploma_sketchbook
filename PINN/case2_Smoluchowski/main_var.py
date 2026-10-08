"""
training based on ../../var_form/fp_1a.md
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
    T=2.0,
    glorot_init=True,

    act_fn='tanh',
    #act_fn='silu',
    lr=1e-3,

    d=2,
    #layers="128,128,128,128",
    #layers="96,96,96,96",
    layers="64,64,64,64",

    #bs=1024,
    bs=8192,
    resampling_frequency=100,
    n_res_points=819200,
    prevent_resampling=False,

    n_steps=1_000,
    n_steps_decay=40, # 25x decay
    logging_frequency=10,

    output_dir = 'run_adam_test',
    starting_model=None,
    #starting_model=f"run_adam_time_adapt_resample_best/model.pth",

    enable_testing = True, #TD
    n_test_points=100_000, # for ic, tc, and each time slice
    n_test_chunk_size=100_000,

    ##########################
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_name = run_utils.setup_dir(args)

### PREP PDE MODEL
a = torch.tensor([0.8067, 1.0391, 1.0101, 0.7722, 1.0593, 1.0706, 0.9805, 0.9893, 0.9514, 1.1419])[:d]
import pde_models_sde
smol_mod = pde_models_sde.SmoluchowskiDoubleWell(d=d, a=a, beta=0.0, Z=0.0) # beta and Z not needed as params...
pde_model = pde_models_sde.SmoluchowskiVariational(smol_mod, d, alpha=1.0, beta=1.0, dt=0.01)
print(type(pde_model))


args.enable_testing = False
if args.enable_testing:
    # pass in a distribtion for sampling the center more heavily ? - for L_inf error est.
    # do not do the MSE loss reporting
    #testing_suite.testing_terms
    import test
    testing_suite = test.TestingSuiteFP(d, device, args.n_test_chunk_size, utility.floats_from_string_list(args.test_norm_slices))
    test_sampling_type = "domain_and_trajectories"
    spatial_domain = run_utils.make_spatial_domain(d, args.L_min, args.L_max, device=device)
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
else:
    testing_suite = None


sampling_settings = {
    "d": d,
    "T": args.T,
    "bs": args.bs,
    "n_samples": args.bs * args.resampling_frequency,
    "n_steps": 1000,
}
t2 = time.time()
print("---- Prep works inside main.py", t2-t1)



t1 = time.time()

layers = utility.layers_from_string(args.layers)
print(f"\n{'='*60}")
print(f"Training variational-PINN for {d}D PDE")
print(f"Domain: R^{d} x [0,T]")
print(f"{'='*60}\n")
print(args.layers)
output_dim = 1

torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

run_utils.save_input_config(dir_name, args)

pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')
print()

if args.act_fn == 'tanh':
    act_fn = torch.nn.Tanh
elif args.act_fn == 'silu':
    act_fn = torch.nn.SiLU
elif args.act_fn == 'softplus':
    act_fn = torch.nn.Softplus
elif args.act_fn == 'relu':
    act_fn = torch.nn.ReLU
else:
    raise NameError

import architecture
model = architecture.PINN_spatial(d, output_dim, layers, activation_fn=act_fn).to(device)
if args.glorot_init:
    model.apply(architecture.init_glorot_weights)

if False:
    if args.starting_model:
        print("Starting from an existin model!")
        model.load_state_dict(torch.load(args.starting_model, weights_only=True))
        model.train()
    if args.custom_ic_model is not None:
        print(f"Using a custom IC function: {args.custom_ic_model}")
        #model_ic = architecture.load_model_from_json(args.custom_ic_model, device)
        import sys, os
        parent_dir = os.path.dirname(args.custom_ic_model)
        model_metadata = utility.json_load(f'{parent_dir}/model_metadata.json')
        layers = utility.layers_from_string(model_metadata["args"]["layers"])
        model_ic = architecture.PINN(D, layers, output_dim, head_fn=head_fun, activation_fn=act_fn).to(device)
        model_ic.load_state_dict(torch.load(args.custom_ic_model, weights_only=True))
        model_ic.eval()
        custom_ic_fn = model_ic.forward
    else:
        custom_ic_fn = None


optimizer, scheduler = run_utils.make_optim(model, args)
profiler = run_utils.make_profiler(dir_name, args, device)

t2 = time.time()
print("---- Prep works inside runner()", t2-t1)

v_fun = pde_model.tilde_rho_0

t1 = time.time()
from trainers import PINN_Trainer_VarFun
trainer = PINN_Trainer_VarFun(
    model, optimizer, scheduler, pde_model,
    sampling_settings=sampling_settings,
    testing_suite=testing_suite,
    v_fun=v_fun,
    profiler=profiler,
    device=device,
    dir_name=dir_name,
    grad_clip_norm=args.grad_clip_norm,
    memory_tracker=utility.MemoryTracker(device) if args.enable_memory_tracking else None
)

losses, rel_l2_data, linf_data, norm_data = trainer.train_adam_minibatch(
    n_steps=args.n_steps,
    n_steps_decay=args.n_steps_decay,
    resampling_frequency=args.resampling_frequency,
    logging_frequency=args.logging_frequency,
    prevent_resampling=args.prevent_resampling,
    gradnorm_update_freq=args.gradnorm_update_freq,
    crit_loss_val=args.crit_loss_val,
)
print("\nAdam training complete!")
t2 = time.time()
print(utility.get_duration_h_m_s(t1, t2, "Adam training"))

report = {}
report['device'] = device.type
report['runtime'] = t2-t1
report['runtime_fmt'] = utility.get_duration_h_m_s(t1, t2, "Run")

# create a simulation results file?
# - add in time
# - weights
sdfs

print("\nTraining complete!")
t1 = time.time()
run_utils.save_run(
    dir_name, model, losses, args, device,
    head_fn=None,
    loss_weighting=None,
    output_dim=output_dim,
)
t2 = time.time()
print("---- Time for save_run()", t2-t1)

t1 = time.time()




import visualize_training_metrics
file_name = f'{dir_name}/training_loss'
# change the order
losses = {k: losses[k] for k in active_losses} | {'total': losses['total']}
visualize_training_metrics.plot_loss(losses, file_name)
report["loss"] = {}
for term in list(active_losses) + ['total']:
    last_100 = torch.tensor(losses[term])[-100:]
    report["loss"][f"{term}_last"] = last_100[-1].item()
    report["loss"][f"{term}_mean"] = torch.mean(last_100).item()
    report["loss"][f"{term}_std"] = torch.std(last_100).item()
    report["loss"][f"{term}_min"] = torch.min(last_100).item()
    report["loss"][f"{term}_max"] = torch.max(last_100).item()

if args.enable_testing:
    #rel_l2_data = {k: [d[k] for d in test_log_rel_l2] for k in test_log_rel_l2[0]}
    #linf_data = {k: [d[k] for d in test_log_linf] for k in test_log_linf[0]}
    file_name = f'{dir_name}/training_test'
    # tot
    rel_l2_data['total'] = torch.mean(torch.tensor([ rel_l2_data[term] for term in testing_suite.analytic_terms ]), dim=0)
    linf_data['total'] = torch.tensor(
        [ linf_data[term] for term in testing_suite.analytic_terms ]
    ).max(dim=0).values
    visualize_training_metrics.plot_test_rel_l2(rel_l2_data, file_name+'_rel_l2', x=trainer.log_steps)
    visualize_training_metrics.plot_test_linf(linf_data, file_name+'_linf', x=trainer.log_steps)
    visualize_training_metrics.plot_test_norm(norm_data, file_name+'_norm', x=trainer.log_steps)
    torch.save(rel_l2_data, f'{file_name}_rel_l2.pth')
    torch.save(linf_data, f'{file_name}_linf.pth')
    torch.save(norm_data, f'{file_name}_norm.pth')
    report["test_rel_l2"] = {}
    report["test_linf"] = {}
    for term in list(testing_suite.analytic_terms)+['total']:
        for test_metric, test_data in zip(('test_rel_l2', 'test_linf'), (rel_l2_data, linf_data)):
            last_100 = test_data[term][-100:] if (term == 'total') else torch.tensor(test_data[term])[-100:] 
            report[test_metric][f"{term}_last"] = last_100[-1].item()
            report[test_metric][f"{term}_mean"] = torch.mean(last_100).item()
            report[test_metric][f"{term}_std"] = torch.std(last_100).item()
            report[test_metric][f"{term}_min"] = torch.min(last_100).item()
            report[test_metric][f"{term}_max"] = torch.max(last_100).item()


# individual loss term weighting - pde,bc,ic

if args.enable_memory_tracking:
    file_name = f'{dir_name}/training_memory_metrics'
    memory_history = {}
    report["memory_metrics"] = {}
    for k, v in trainer.memory_history.items():
        if (len(v) > 0) and (v[0] is not None):
            report["memory_metrics"][k] = max(v)
            memory_history[k] = v
    visualize_training_metrics.plot_mem(memory_history, file_name)
    utility.json_dump(f"{file_name}.json", memory_history)

# save the most imporant stufff
utility.json_dump(f"{dir_name}/report.json", report)

t2 = time.time()
print("---- Time for all training metric saves", t2-t1)


trainer
