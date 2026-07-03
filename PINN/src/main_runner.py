


def add_common_args(parser):
    # basic
    parser.add_argument("--config", default=None, type=str, help="Path to a JSON file with parameter values.")
    parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--grad_clip_norm", default=None, type=float, help="Max-norm gradient clipping for the train step. None disables it.")
    parser.add_argument("--mode", default="class_pde", type=str, help="score_pde, ll_ode, q_pde, class_pde")
    # don't touch this
    parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
    parser.add_argument("--lr", default=1e-3, type=float, help="")
    parser.add_argument("--term_loss_val", default=None, type=float, help="")
    # dir
    parser.add_argument("--output_dir", default="run_latest_vanilla", type=str, help="")
    parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")
    # problem specific
    parser.add_argument("--d", default=4, type=int, help="Number of spatial dimensions.")
    parser.add_argument("--T", default=3.5, type=float, help="")
    parser.add_argument("--L_min", default=-5.0, type=float, help="")
    parser.add_argument("--L_max", default=5.0, type=float, help="")
    # common arch choices
    parser.add_argument("--glorot_init", action="store_true", help="")
    parser.add_argument("--use_lbfgs", action="store_true", help="")
    # reporting / profiling
    parser.add_argument("--enable_profiler", action="store_true", help="")
    parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
    parser.add_argument("--enable_memory_tracking", action="store_true", help="Log process RAM and GPU memory at each logging interval.")
    # commonly edited
    #parser.add_argument("--layers", default="256,256,256,256", type=str, help="")
    #parser.add_argument("--layers", default="32,32,32,32", type=str, help="")
    parser.add_argument("--layers", default="64,64,64,64", type=str, help="")
    #parser.add_argument("--layers", default="128,128,128,128", type=str, help="")
    parser.add_argument("--bs", default=100, type=int, help="")
    parser.add_argument("--one_batch_per_epoch", action="store_true", help="")
    parser.add_argument("--n_steps", default=4999, type=int, help="")
    #parser.add_argument("--n_steps", default=499, type=int, help="")
    parser.add_argument("--n_steps_decay", default=1_000, type=int, help="Decay by 0.9 every 2000 steps.")
    # loss weights
    parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
    parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
    parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
    parser.add_argument("--lambda_norm", default=0.1, type=float, help="Weight of the integral p dx = 1 normalization loss.")
    parser.add_argument("--use_gradnorm", action="store_true", help="grad adaptive loss term weighting.")
    parser.add_argument("--gradnorm_update_freq", default=50, help="how many steps until we update the weights")
    parser.add_argument("--active_losses", default="pde,ic", type=str, help="Comma-separated subset of {pde,bc,ic,norm}. 'pde' is required.")
    # sampling
    parser.add_argument("--n_res_points", default=10_000, type=int, help="")
    parser.add_argument("--n_trajs", default=1_000, type=int, help="")
    parser.add_argument("--nt_steps", default=100, type=int, help="")
    parser.add_argument("--resampling_frequency", default=10_000, type=int, help="")
    parser.add_argument("--prevent_resampling", action="store_true", help="")
    parser.add_argument("--sampling_type", default="domain_and_trajectories", type=str, help="trajectories, domain")
    parser.add_argument("--f_pde_full_domain", default=1, type=int, help="")
    parser.add_argument("--f_pde_trajs", default=1, type=int, help="")
    parser.add_argument("--f_ic_full_domain", default=1, type=int, help="")
    parser.add_argument("--f_ic_trajs", default=1, type=int, help="")
    parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
    # testing and logging
    parser.add_argument("--n_test_points", default=10_000, type=int, help="Number of test points for the testing suite.")
    parser.add_argument("--logging_frequency", default=100, type=int, help="")
    parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")
    # causal strategies
    parser.add_argument("--time_strategy", default=0, type=int, choices=[0,1,2], help="0=none, 1=time_adapt_sampling, 2=causal_loss_weighting")
    # for causal_loss_weighting
    parser.add_argument("--t_discr", default="0.0, 0.5, 1.5, 3.5", type=str, help="")
    parser.add_argument("--eps", default=0.1, type=float, help="")
    # sdgd
    parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
    parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
    # enable transfer learning / finetuning
    parser.add_argument("--starting_model", default=None, type=str, help="")
    parser.add_argument("--custom_ic_model", default=None, type=str, help="")



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
import architecture, utility
import run_utils
def runner(args, pde_model, sampling_settings, sampling_type, testing_suite, head_fun):


    d = args.d  # space dims
    D = d + 1   # space + time dims
    layers = utility.layers_from_string(args.layers)
    print(f"\n{'='*60}")
    print(f"Training vanilla-PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")
    print(args.layers)
    output_dim = 1


    dir_name, device = run_utils.setup_run(args)
    run_utils.save_input_config(dir_name, args)

    pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')
    print()

    model = architecture.PINN(D, layers, output_dim, head_fn=head_fun).to(device)
    if args.glorot_init:
        model.apply(architecture.init_glorot_weights)

    #model = architecture.PINN_base(D, layers, 1).to(device)
    if args.starting_model:
        model.load_state_dict(torch.load(args.starting_model, weights_only=True))


    if args.custom_ic_model is not None:
        print(f"Using a custom IC function: {args.custom_ic_model}")
        model_ic = architecture.load_model_from_json(args.custom_ic_model, device)
        model_ic.eval()
        custom_ic_fn = model_ic.forward
    else:
        custom_ic_fn = None
    sampling_settings["custom_ic_fn"] = custom_ic_fn


    active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
    print(f"Active losses: {active_losses}")

    optimizer, scheduler = run_utils.make_optim(model, args)
    loss_weighting = run_utils.make_loss_weighting(args, active_losses, device=device)
    profiler = run_utils.make_profiler(dir_name, args, device)

    sdgd_num_dims = args.sdgd_num_dims if args.sdgd_num_dims is not None else d
    if args.use_sdgd:
        print(f"Using SDGD with {sdgd_num_dims} dimensions (d={d})")
    else:
        print(f"Using regular Adam training.")

    import time
    t1 = time.time()



    from trainers import PINN_Trainer
    trainer = PINN_Trainer(
        model, optimizer, scheduler, pde_model,
        sampling_type=sampling_type, sampling_settings=sampling_settings,
        loss_weighting=loss_weighting, testing_suite=testing_suite,
        active_losses=active_losses, profiler=profiler, device=device,
        dir_name=dir_name,
        grad_clip_norm=args.grad_clip_norm,
        memory_tracker=utility.MemoryTracker(device) if args.enable_memory_tracking else None
    )
    #losses_adam, l2_errs_adam = trainer.train_adam_minibatch(


    use_causal_loss_weighting = False
    use_time_adapt_sampling = False
    t_discr = None
    if args.time_strategy == 0:
        print("time_strategy = none")
    elif args.time_strategy == 1:
        print("time_strategy = time_adapt_sampling")
        use_time_adapt_sampling = True
    elif args.time_strategy == 2:
        print("time_strategy = causal_loss_weighting")
        use_causal_loss_weighting = True
        t_discr = torch.tensor(utility.floats_from_string_list(args.t_discr), device=device)
    else:
        raise NameError("time_strategy is whaaat???")
    print()



    losses = run_utils.init_losses(("total",) + active_losses)
    if args.use_lbfgs:
        trainer.optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1e-3,
            max_iter=20, # inner CG iterations per step
            max_eval=25,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn='strong_wolfe'
        )
        trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=0.9)
        losses_adam, test_log_res_mse, test_log_rel_l2 = trainer.train_lbfgs(
            n_steps=args.n_steps,
            n_steps_decay=args.n_steps_decay,
            resampling_frequency=args.resampling_frequency,
            logging_frequency=args.logging_frequency,
            use_sdgd=args.use_sdgd,
            sdgd_num_dims=sdgd_num_dims,
            one_batch_per_epoch=args.one_batch_per_epoch,
            use_causal_loss_weighting=use_causal_loss_weighting, t_discr=t_discr, eps=args.eps,
            use_time_adapt_sampling=use_time_adapt_sampling,
            prevent_resampling=args.prevent_resampling,
            gradnorm_update_freq=args.gradnorm_update_freq,
            term_loss_val=args.term_loss_val,
        )
    else:
        losses_adam, test_log_res_mse, test_log_rel_l2 = trainer.train_adam_minibatch(
            n_steps=args.n_steps,
            n_steps_decay=args.n_steps_decay,
            resampling_frequency=args.resampling_frequency,
            logging_frequency=args.logging_frequency,
            use_sdgd=args.use_sdgd,
            sdgd_num_dims=sdgd_num_dims,
            one_batch_per_epoch=args.one_batch_per_epoch,
            use_causal_loss_weighting=use_causal_loss_weighting, t_discr=t_discr, eps=args.eps,
            use_time_adapt_sampling=use_time_adapt_sampling,
            prevent_resampling=args.prevent_resampling,
            gradnorm_update_freq=args.gradnorm_update_freq,
            term_loss_val=args.term_loss_val,
        )
    run_utils.merge_losses(losses, losses_adam)
    print("\nAdam training complete!")
    print(utility.get_duration_h_m_s(t1, time.time(), "Adam training"))

    # create a simulation results file?
    # - add in time
    # - weights

    print("\nTraining complete!")

    run_utils.save_run(
        dir_name, model, losses, args, device,
        head_fn=None,
        loss_weighting=loss_weighting if args.n_steps > 0 else None,
        output_dim=output_dim,
    ) 
    import visualize_training_metrics

    file_name = f'{dir_name}/training_loss'
    visualize_training_metrics.plot_loss(losses, file_name)
    if args.enable_testing:
        res_mse_data = {k: [d[k] for d in test_log_res_mse] for k in test_log_res_mse[0]}
        rel_l2_data = {k: [d[k] for d in test_log_rel_l2] for k in test_log_rel_l2[0]}
        file_name = f'{dir_name}/training_test'
        #n = len(res_mse_data[list(res_mse_data.keys())[0]])
        #x = args.logging_frequency * torch.linspace(1, n, n, dtype=torch.int)
        visualize_training_metrics.plot_test_res_mse(res_mse_data, file_name+'_res_mse')
        visualize_training_metrics.plot_test_rel_l2(rel_l2_data, file_name+'_rel_l2')
        torch.save(res_mse_data, f'{file_name}_res_mse.pth')
        torch.save(rel_l2_data, f'{file_name}_rel_l2.pth')


    # individual loss term weighting - pde,bc,ic
    if args.use_gradnorm:
        weights_hist_tensor = torch.stack(loss_weighting.weights_history, dim=1)
        weights_hist = {}
        for i, loss_name in enumerate(active_losses):
            weights_hist[loss_name] = weights_hist_tensor[i]
        file_name = f'{dir_name}/training_gradnorm'
        torch.save(weights_hist, f'{file_name}.pth')
        visualize_training_metrics.plot_GradNorm_weights(weights_hist, file_name)

    if use_time_adapt_sampling:
        file_name = f'{dir_name}/training_time_adapt_sampling'
        torch.save(torch.tensor(trainer.time_adapt_sampl_hist), f'{file_name}.pth')
        visualize_training_metrics.plot_time_adapt_sampling(trainer.time_adapt_sampl_hist, file_name)

    if use_causal_loss_weighting:
        def save_causal_weights_losses(causal_wl_hist, wl_type, term_type):
            file_name = f'{dir_name}/training_causal_{wl_type}_{term_type}'
            causal_wl_hist_tensor = torch.stack(causal_wl_hist, dim=1)
            causal_wl_hist = {}
            if wl_type == 'losses':
                causal_wl_hist[term_type] = losses[term_type]
            for i in range(len(t_discr)-1):
                causal_wl_hist[f"{i+1}"] = causal_wl_hist_tensor[i]
            torch.save(causal_wl_hist, f'{file_name}.pth')
            visualize_training_metrics.plot_causal_weights_losses(causal_wl_hist, file_name, wl_type, term_type, t_discr)
        save_causal_weights_losses(trainer.causal_weights_hist_pde, 'weights', 'pde')
        save_causal_weights_losses(trainer.causal_losses_hist_pde,  'losses', 'pde')
        if 'bc' in active_losses:
            save_causal_weights_losses(trainer.causal_weights_hist_bc,  'weights', 'bc')
            save_causal_weights_losses(trainer.causal_losses_hist_bc,   'losses', 'bc')

    if args.enable_memory_tracking:
        file_name = f'{dir_name}/training_memory_metrics'
        visualize_training_metrics.plot_mem(trainer.memory_history, file_name)
        utility.json_dump(f"{file_name}.json", trainer.memory_history)

    return trainer, model, dir_name