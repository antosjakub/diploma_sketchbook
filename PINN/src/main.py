import torch
import argparse
from torch.profiler import record_function
import sampling, loss, architecture, utility, pde_models

parser = argparse.ArgumentParser()
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="148,148,148", type=str, help="")
parser.add_argument("--n_steps", default=450, type=int, help="")
parser.add_argument("--n_steps_decay", default=2000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--n_calloc_points", default=10_000, type=int, help="")
parser.add_argument("--n_test_calloc_points", default=10_000, type=int, help="")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--resampling_frequency", default=500, type=int, help="")
parser.add_argument("--bs", default=512, type=int, help="")
parser.add_argument("--lambda_pde", default=10.0, type=float, help="")
parser.add_argument("--lambda_bc", default=0.01, type=float, help="")
parser.add_argument("--lambda_ic", default=1.0, type=float, help="")
parser.add_argument("--lambda_norm", default=0.1, type=float, help="Weight of the ∫p dx = 1 normalization loss.")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--active_losses", default="pde,bc,ic,norm", type=str, help="Comma-separated subset of {pde,bc,ic,norm}. 'pde' is required.")
parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
#parser.add_argument("--n_norm_buffer", default=10_000, type=int, help="Size of the p_inf sample buffer for the normalization loss.")
#parser.add_argument("--n_norm_batch", default=1_000, type=int, help="Per-step minibatch drawn from the p_inf buffer for the normalization loss.")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# L-BFGS
parser.add_argument("--n_steps_lbfgs", default=0, type=int, help="Number of L-BFGS steps after Adam. Set to 0 to skip.")
parser.add_argument("--n_steps_log_lbfgs", default=100, type=int, help="")
parser.add_argument("--lr_lbfgs", default=1.0, type=float, help="Learning rate for L-BFGS (typically 1.0).")
# smart Defaults
parser.add_argument("--l2_stop_crit", default=0.01, type=float, help="")
parser.add_argument("--l2_stop_crit_lbfgs", default=0.001, type=float, help="")

parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")
parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")
# 
parser.add_argument("--output_dir", default="run_latest/", type=str, help="")
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
parser.add_argument("--use_weak_form", action="store_true", help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")






class PINN_Trainer:
    VALID_LOSS_KEYS = ("pde", "bc", "ic", "norm")
    LOADER_KEYS = ("pde", "bc", "ic", "norm")  # all terms come from a DataLoader

    def __init__(
        self, model, optimizer, scheduler, pde_model,
        sampling_type, sampling_settings,
        loss_weighting, testing_suite, active_losses=("pde", "bc", "ic", "norm"),
        profiler=None, device='cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
        self.sampling_type = sampling_type
        self.sampling_settings = sampling_settings
        self.loss_weighting = loss_weighting
        self.profiler = profiler
        self.testing_suite = testing_suite
        self.device = device
        self.d = self.pde_model.d

        for k in active_losses:
            if k not in self.VALID_LOSS_KEYS:
                raise ValueError(f"Unknown loss term '{k}'. Valid: {self.VALID_LOSS_KEYS}")
        if "pde" not in active_losses:
            raise ValueError("'pde' must be in active_losses — it drives the step count.")
        self.active_losses = tuple(active_losses)
        self.bundle = None  # populated in train_adam_minibatch

    def normalization_loss(self, x, model, precomputed, n_time_slices=4):
        """Importance-sampled estimate of (∫p(x,t) dx - 1)^2, averaged over K random
        time slices. `batch` is (x, {"p_inf": p_inf_x}) from the norm DataLoader;
        Z is read from pde_model (cached at bundle-build time)."""
        p_inf_x = precomputed["p_inf"]
        Z = self.pde_model.Z
        T = self.sampling_settings.get("T", 1.0)
        n_batch = x.shape[0]

        t = T * torch.rand(n_time_slices, 1, device=self.device)
        X_rep = x.unsqueeze(0).expand(n_time_slices, n_batch, self.d).reshape(-1, self.d)
        t_rep = t.unsqueeze(1).expand(n_time_slices, n_batch, 1).reshape(-1, 1)
        X = torch.cat([X_rep, t_rep], dim=1)
        p = model(X).reshape(n_time_slices, n_batch)
        p_inf = p_inf_x.squeeze(-1).unsqueeze(0)
        integral_est = Z * (p / p_inf).mean(dim=1)
        return ((integral_est - 1.0) ** 2).mean()

    def _loss_term(self, k, batches_by_name, use_sdgd=False, sdgd_num_dims=None):
        if k == "pde":
            b = batches_by_name["pde"]
            if use_sdgd:
                return loss.sdgd_loss(b[0], self.model, self.pde_model, b[1], sdgd_num_dims)
            return self.pde_model.pde_loss(b[0], self.model, b[1])
        if k == "bc":
            b = batches_by_name["bc"]
            return self.pde_model.bc_loss(b[0], self.model, b[1])
        if k == "ic":
            b = batches_by_name["ic"]
            return self.pde_model.ic_loss(b[0], self.model, b[1])
        if k == "norm":
            b = batches_by_name["norm"]
            return self.normalization_loss(b[0], self.model, b[1])
        raise ValueError(f"Unknown loss term '{k}'")

    def train_adam_step(self, batches_by_name, use_sdgd=False, sdgd_num_dims=None):
        self.optimizer.zero_grad()
        with record_function("loss"):
            per_term = [
                self._loss_term(k, batches_by_name, use_sdgd, sdgd_num_dims)
                for k in self.active_losses
            ]
        loss_value = self.loss_weighting.weight_loss(per_term)
        with record_function("backward"):
            loss_value.backward()
        with record_function("optimizer_step"):
            self.optimizer.step()
        per_term_vals = {k: per_term[i].item() for i, k in enumerate(self.active_losses)}
        return loss_value, per_term_vals


    ## --- DEAD CODE (kept for reference) ---------------------------------
    #def train_adam_step_accumulated(self, batch_iterator):
    #    self.optimizer.zero_grad()
    #
    #    n_cycles = 0
    #    loss_pde, loss_bc, loss_ic = 0.0, 0.0, 0.0
    #    for batch_pde, batch_bc, batch_ic in batch_iterator:
    #        n_cycles += 1
    #        batch_pde[0].requires_grad = True
    #        # Compute individual losses
    #        with record_function("loss"):
    #            loss_pde += self.pde_model.pde_loss(batch_pde[0], self.model, batch_pde[1])
    #            loss_bc += self.pde_model.bc_loss(batch_bc[0], self.model, batch_bc[1])
    #            loss_ic += self.pde_model.ic_loss(batch_ic[0], self.model, batch_ic[1])
    #    # Weighted loss
    #    loss_pde /= n_cycles
    #    loss_bc /= n_cycles
    #    loss_ic /= n_cycles
    #    loss_value = self.loss_weighting.weight_loss(
    #        [loss_pde, loss_bc, loss_ic]
    #    )
    #
    #    # Backward pass
    #    with record_function("backward"):
    #        loss_value.backward()
    #    with record_function("optimizer_step"):
    #        self.optimizer.step()
    #
    #    return loss_value, (loss_pde.item(), loss_bc.item(), loss_ic.item())



    ## --- DEAD CODE (kept for reference) ---------------------------------
    #def train_adam_fullbatch(self, n_steps, n_steps_decay, n_calloc_points, n_test_calloc_points, resampling_frequency=2000, testing_frequency=100):
    #    """
    #    Train the model using Adam optimizer.
    #    """
    #    losses = []
    #    l2_errs = []
    #
    #    n_points_interior = 8*n_calloc_points//10
    #    n_points_boundary = n_calloc_points//10
    #    n_points_initial = n_calloc_points//10
    #
    #    if self.profiler: self.profiler.make()
    #
    #    # Generate training data
    #    X_interior, X_boundary, X_initial, _ = sampling.sample_collocation_points(
    #        self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
    #    )
    #    u_bc_target = self.pde_model.u_bc(X_boundary)
    #    u_ic_target = self.pde_model.u_ic(X_initial)
    #
    #    for si in range(n_steps):
    #
    #        if (si + 1) % resampling_frequency == 0:
    #            print("New training data arrived?")
    #            X_interior, X_boundary, X_initial, _ = sampling.sample_collocation_points(
    #                self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
    #            )
    #            u_bc_target = self.pde_model.u_bc(X_boundary)
    #            u_ic_target = self.pde_model.u_ic(X_initial)
    #
    #        if self.profiler: self.profiler.start(si)
    #
    #        loss_value, (loss_pde, loss_bc, loss_ic) = self.train_adam_step(X_interior, X_boundary, X_initial, u_bc_target, u_ic_target)
    #        losses.append(loss_value.item())
    #
    #        if (si + 1) % n_steps_decay == 0:
    #            self.scheduler.step()
    #
    #        if self.profiler: self.profiler.exit(si)
    #
    #        if (si + 1) % testing_frequency == 0:
    #            l2_err, l1_err, rel_err = self.test_model(n_test_calloc_points)
    #            l2_errs.append(l2_err)
    #            print(f'Step {si+1}/{n_steps}, Loss: {loss_value.item():.6f}, '
    #                  f'PDE: {loss_pde:.6f}, '
    #                  f'BC: {loss_bc:.6f}, '
    #                  f'IC: {loss_ic:.6f}, '
    #                  f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
    #                  f'L2: {l2_err:.6f}, '
    #                  f'L1: {l1_err:.6f}, '
    #                  f'rel_max: {rel_err:.6f}'
    #            )
    #
    #    return losses, l2_errs


    def _build_bundle(self):
        return sampling.create_dataloaders(
            self.sampling_type, self.model, self.pde_model,
            self.sampling_settings, self.active_losses, device=self.device,
        )

    def train_adam_minibatch(self, n_steps, n_steps_decay, resampling_frequency=2000, testing_frequency=100, use_sdgd=False, sdgd_num_dims=None):
        """Train the model using Adam optimizer."""
        losses = {"total": [], **{k: [] for k in self.active_losses}}
        l2_errs = []

        if self.profiler: self.profiler.make()

        self.bundle = self._build_bundle()
        loader_keys = [k for k in self.LOADER_KEYS if k in self.active_losses]

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                print("New training data arrived!")
                self.bundle = self._build_bundle()

            if self.profiler: self.profiler.start(si)

            loaders = [self.bundle[k] for k in loader_keys]
            for batches in zip(*loaders):
                batches_by_name = dict(zip(loader_keys, batches))
                loss_value, last_losses = self.train_adam_step(
                    batches_by_name, use_sdgd=use_sdgd, sdgd_num_dims=sdgd_num_dims
                )
            losses["total"].append(loss_value.item())
            for k in self.active_losses:
                losses[k].append(last_losses[k])

            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            if type(self.loss_weighting).__name__ == 'AdaptiveWeights':
                if (si + 1) % 50 == 0:
                    self.optimizer.zero_grad()
                    per_term_w = [self._loss_term(k, batches_by_name) for k in self.active_losses]
                    self.loss_weighting.update(per_term_w, self.model)

            if self.profiler: self.profiler.exit(si)

            if (si + 1) % testing_frequency == 0:
                parts = [f"Step {si+1}/{n_steps}", f"Loss: {loss_value.item():.6f}"]
                for k in self.active_losses:
                    parts.append(f"{k}: {last_losses[k]:.6f}")
                parts.append(f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                log = ", ".join(parts)
                if self.testing_suite is not None:
                    l2_err, l1_err, rel_err = self.testing_suite.test_model(self.model)
                    l2_errs.append(l2_err)
                    log += f", L2: {l2_err:.6f}, L1: {l1_err:.6f}, rel_max: {rel_err:.6f}"
                print(log)

        return losses, l2_errs




#def train_pinn_lbfgs(
#        model,
#        pde_residual, bc_residual, ic_residual,
#        u_analytic,
#        d,
#        n_steps=500,
#        n_steps_log=100,
#        n_points_pde=2000, n_points_bc=400, n_points_ic=400,
#        lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
#        lr=1.0,
#        l2_stop_crit=0.001,
#        compute_laplace=True,
#        device='cpu'
#    ):
#    """Fine-tune the PINN model with L-BFGS after Adam pre-training."""
#
#    # Sample fixed collocation points (L-BFGS works best with a fixed dataset)
#    X_interior, X_boundary, X_initial = sample_collocation_points(
#        d, n_points_pde, n_points_bc, n_points_ic, device
#    )
#    X_interior.requires_grad = True
#
#    optimizer_lbfgs = torch.optim.LBFGS(
#        model.parameters(),
#        lr=lr,
#        max_iter=20,           # inner CG iterations per step
#        max_eval=25,
#        history_size=50,
#        tolerance_grad=1e-7,
#        tolerance_change=1e-9,
#        line_search_fn='strong_wolfe'
#    )
#
#    losses = []
#    l2_errs = []
#    step_counter = [0]  # mutable counter accessible inside closure
#
#    def closure():
#        optimizer_lbfgs.zero_grad()
#        loss_pde = pde_loss(model, X_interior, pde_residual, compute_laplace=compute_laplace)
#        loss_bc  = boundary_condition_loss(model, X_boundary, bc_residual)
#        loss_ic  = initial_condition_loss(model, X_initial, ic_residual)
#        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
#        loss.backward()
#        return loss
#
#    print(f"\n{'='*60}")
#    print(f"Starting L-BFGS fine-tuning ({n_steps} steps)")
#    print(f"{'='*60}\n")
#
#    l2_err = 1.0 + l2_stop_crit # init with some val
#    for si in range(n_steps):
#        loss = optimizer_lbfgs.step(closure)
#        losses.append(loss.item())
#        step_counter[0] += 1
#
#        if (si + 1) % n_steps_log == 0:
#            X_interior_test, _, _ = sample_collocation_points(
#                d, n_points_pde, n_points_bc, n_points_ic, device=device
#            )
#            with torch.no_grad():
#                u_pred = model(X_interior_test)
#                u_true = u_analytic(X_interior_test)
#            l2_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
#            l2_errs.append(l2_err)
#
#            # Recompute individual losses for logging (no grad needed)
#            with torch.no_grad():
#                u_bc  = model(X_boundary)
#                u_ic  = model(X_initial)
#            X_log = X_interior.detach().requires_grad_(True)
#            loss_pde_log = pde_loss(model, X_log, pde_residual, compute_laplace=compute_laplace)
#            loss_bc_log  = boundary_condition_loss(model, X_boundary, bc_residual)
#            loss_ic_log  = initial_condition_loss(model, X_initial, ic_residual)
#
#            print(f'[L-BFGS] Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
#                  f'PDE: {loss_pde_log.item():.6f}, BC: {loss_bc_log.item():.6f}, '
#                  f'IC: {loss_ic_log.item():.6f}, L2: {l2_err:.6f}')
#
#        if l2_err < l2_stop_crit:
#            break
#
#    return losses, l2_errs


# Main execution
if __name__ == "__main__":
    import run_utils

    args = parser.parse_args([] if "__file__" not in globals() else None)
    dir_name, device = run_utils.setup_run(args)

    d = args.d  # space dims
    D = d + 1   # space + time dims
    layers = utility.layers_from_string(args.layers)
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")

    # PDE equation
    if False:
        import sys, os
        fp_dir = os.path.join(os.path.dirname(__file__), '../../Fokker-Planck')
        sys.path.append(fp_dir)
        import utility_fp
        mol_coords = torch.tensor(utility_fp.read_mol(os.path.join(fp_dir, 'molecules/mol.1')))

        n_atoms = 3
        dof_per_atom = 2
        d = n_atoms * dof_per_atom
        mol_coords_ss = mol_coords[:n_atoms,:dof_per_atom]

        L = 10.0
        mean = torch.mean(mol_coords, dim=0)
        mol_coords_ss -= mean.unsqueeze(dim=1)
        mol_coords_ss /= L
        mol_coords_ss += 0.5
        x0 = mol_coords_ss.reshape(-1)
        print(x0)

        pde_model = pde_models.FokkerPlanckLJ(n_atoms=n_atoms, dof_per_atom=dof_per_atom, x0=x0, L=L)

    else:
        #pde_model = pde_models.HeatEquationWithSource(d)
        #pde_model = pde_models.TravellingGaussPacket(d, gamma=1)
        #pde_model = pde_models.HeatEquation(d)
        a = 0.7 + 0.5*torch.rand(d)
        print(a)
        pde_model = pde_models.SmoluchowskiDoubleWell(d, beta=1.0, a=a)


    print(type(pde_model))
    print(pde_model.get_pde_metadata())
    if args.use_weak_form:
        if pde_model.has_weak_form:
            use_weak_form = True
            pde_residual = pde_model.pde_residual_weak_form
        else:
            raise Exception("!! ISSUE: '--use_weak_form' argument was passed, but the PDE model does not have a weak form defined.")
    else:
        use_weak_form = False


    # Select the model architecture
    if args.starting_model:
        model = torch.load(args.starting_model, weights_only=False)
    else:
        head_fn = utility.identity_fn
        #head_fn = lambda x: torch.exp(x)
        #head_fn = torch.nn.Softplus()
        model = architecture.PINN(D, layers, head_fn=head_fn).to(device)
        #model = PINN_SeparableTimes(D, layers).to(device)
        #model = PINN_SepTime(D, layers).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)


    active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
    print(f"Active losses: {active_losses}")

    # Preparation time
    losses = run_utils.init_losses(("total",) + active_losses)
    l2_errs = []

    # Train the model
    if args.n_steps > 0:
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
        if True:
            testing_suite = None
        else:
            testing_suite = utility.TestingSuite(d)
            testing_suite.make_test_data(pde_model, args.n_test_calloc_points, f"{dir_name}/testing_data.pt")
            #testing_suite.connect_test_data(f"{dir_name}/testing_data.pt")
        
        L = 3.5
        T = 2.0
        spatial_domain = torch.stack([torch.full((d,), -L), torch.full((d,), L)], dim=1)
        pde_model.Z = pde_model.estimate_Z(spatial_domain)
        print(f"Z ~ {pde_model.Z}")
        sampling_settings = {
            "n_res_points": args.n_calloc_points,
            "bs": args.bs,
            "spatial_domain": spatial_domain,
            "T": T,
            "use_rbas": args.use_rbas,
            #"n_norm_buffer": args.n_norm_buffer,
            #"n_norm_batch": args.n_norm_batch,
        }
        trainer = PINN_Trainer(
            model, optimizer, scheduler, pde_model,
            sampling_type="vanilla_pinn", sampling_settings=sampling_settings,
            loss_weighting=loss_weighting, testing_suite=testing_suite,
            active_losses=active_losses, profiler=profiler, device=device,
        )
        losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
        #losses_adam, l2_errs_adam = trainer.train_adam_fullbatch(
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
        #torch.save(model, f'{dir_name}/model_adam.pth')

    # --- Phase 2: L-BFGS fine-tuning ---
    if args.n_steps_lbfgs > 0:
        losses_lbfgs, l2_errs_lbfgs = train_pinn_lbfgs(
            model,
            d,
            n_steps=args.n_steps_lbfgs,
            n_steps_log=args.n_steps_log_lbfgs,
            lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ic=args.lambda_ic,
            n_points_pde=args.n_points_pde, n_points_bc=args.n_points_bc, n_points_ic=args.n_points_ic,
            lr=args.lr_lbfgs,
            l2_stop_crit=args.l2_stop_crit_lbfgs,
            compute_laplace=not use_weak_form,
            device=device
        )
        losses["total"] += losses_lbfgs
        l2_errs += l2_errs_lbfgs
        print("\nL-BFGS fine-tuning complete!")
        torch.save(model, f'{dir_name}/model_adam_lbfgs.pth')
    
    # Dan
    print("\nTraining complete!")

    loss_name, l2_name = run_utils.save_run(dir_name, model, losses, l2_errs, args, pde_model)

    print(l2_errs)

    # Plot results
    if args.enable_testing:
        n_steps_log = args.testing_frequency
        n_logged_pnts = len(l2_errs)
        steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)

    import visualize_training_metrics
    visualize_training_metrics.plot_loss(losses, loss_name)

    import viz
    model_fn = viz.wrapp_model(model)
    p_ic = lambda X: pde_model.p_0(X[:,:-1])

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
    if args.enable_testing:
        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_p', title="model_p(x,t)").heatmap(model_fn)
        plotter.add_panel('p_analytic', title="p_analytic(x,t)").heatmap(pde_model.p_analytic)
        plotter.add_panel('err', title="err").heatmap(lambda X: model_fn(X) - pde_model.p_analytic(X))
        plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p_analytic.png', t_val=0.234)
        plotter.save_animation(f'{dir_name}/viz/anim_model_p_vs_p_analytic.gif', num_frames=30, fps=5)

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model', title="p_theta(x)").heatmap(model_fn)
    plotter.add_panel('ic', title="p_0(x)").heatmap(p_ic)
    plotter.add_panel('err', title="err").heatmap(lambda X: model_fn(X) - p_ic(X))
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p0_lnk.png', t_val=0.0, cbar={"model": "linked:ic", "err": "linked:ic"})
    plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p0_dyn.png', t_val=0.0, cbar='dynamic')

    plotter = viz.FunctionPlotter(**options)
    plotter.add_panel('model_p', title="p_nn").heatmap(model_fn)
    plotter.save_animation(f'{dir_name}/viz/anim_model_p.gif', num_frames=100, fps=5, t_end = 1.5*T)

    #import viz
    #model_fn = viz.wrapp_model(model)
    #if False:
    #    p_ic = lambda X: pde_model.p_ic(X[:,:-1])

    #    plotter_ic = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
    #    plotter_ic.add_scalar_fn(model_fn, "PINN")
    #    plotter_ic.add_scalar_fn(p_ic, "Initial Condition")
    #    plotter_ic.add_scalar_fn(lambda X: torch.abs(model_fn(X) - p_ic(X)), "Error", cmap='hot')
    #    plotter_ic.save_plot(f'{dir_name}/pinn_plot_ic.png', t_val = 0.0)

    #    plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
    #    plotter.add_scalar_fn(model_fn, "PINN")
    #    plotter.save_animation(f'{dir_name}/pinn_anim.gif', num_frames=30, fps=5)

    #else:
    #    plotter = viz.FunctionPlotter(d=d, device=device)
    #    plotter.add_scalar_fn(model_fn, "PINN Solution")
    #    plotter.add_scalar_fn(pde_model.u_analytic, "Analytic Solution")
    #    plotter.add_scalar_fn(lambda X: torch.abs(model_fn(X) - pde_model.u_analytic(X)), "Error", cmap='hot')
    #    plotter.save_plot(f'{dir_name}/pinn_fig.png', t_val = 0.325)
    #    plotter.save_plot(f'{dir_name}/pinn_fig_ic.png', t_val = 0.0)
    #    plotter.save_animation(f'{dir_name}/pinn_anim.gif', num_frames=30, fps=5)

    #    #visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)
    #    #import visualize_solution_3plots
    #    #visualize_solution_3plots.plot_3(model, pde_model.u_analytic, d, dir_name)
    #    #import visualize_solution_3anims
    #    #visualize_solution_3anims.anim_3(model, pde_model.u_analytic, d, dir_name)