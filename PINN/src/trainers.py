import torch
from torch.profiler import record_function
import sampling, loss, architecture, utility



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


    def _build_bundle(self):
        self.bundle = sampling.create_dataloaders(
            self.sampling_type, self.model, self.pde_model,
            self.sampling_settings, self.active_losses, device=self.device,
        )
        self._bundle_iters = {k: iter(self.bundle[k]) for k in self.bundle}
        return self.bundle

    def _next_batches(self, loader_keys):
        """Pull one batch per active loader. Reshuffle (re-iter) on exhaustion."""
        batches = {}
        for k in loader_keys:
            try:
                batches[k] = next(self._bundle_iters[k])
            except StopIteration:
                self._bundle_iters[k] = iter(self.bundle[k])
                batches[k] = next(self._bundle_iters[k])
        return batches

    def train_adam_minibatch(self, n_steps, n_steps_decay, resampling_frequency=2000, testing_frequency=100, use_sdgd=False, sdgd_num_dims=None, one_batch_per_epoch=False):
        """Train the model using Adam optimizer.

        If one_batch_per_epoch=True, each `si` performs a single gradient step
        (one batch from each active loader). Otherwise iterates over all batches
        of the bundle per `si`.
        """
        losses = {"total": [], **{k: [] for k in self.active_losses}}
        l2_errs = []

        if self.profiler: self.profiler.make()

        self._build_bundle()
        loader_keys = [k for k in self.LOADER_KEYS if k in self.active_losses]

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                print("New training data arrived!")
                self._build_bundle()

            if self.profiler: self.profiler.start(si)

            if one_batch_per_epoch:
                batches_by_name = self._next_batches(loader_keys)
                loss_value, last_losses = self.train_adam_step(
                    batches_by_name, use_sdgd=use_sdgd, sdgd_num_dims=sdgd_num_dims
                )
            else:
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






class PINN_Trainer_1k:
    """Minimal trainer: only the PDE residual loss, single DataLoader.

    Default shape: dataloader holds ~100_000 points, each gradient step
    consumes bs points (~1000). Sampling mode is chosen per-call:
      - sampling_type="domain"       → vanilla-PINN-style uniform/LHS
      - sampling_type="trajectories" → Euler-Maruyama SDE trajectory bank
    """

    def __init__(
        self, model, optimizer, scheduler, pde_model,
        sampling_type, sampling_settings,
        testing_suite=None, profiler=None, device='cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
        self.sampling_type = sampling_type
        self.sampling_settings = sampling_settings
        self.testing_suite = testing_suite
        self.profiler = profiler
        self.device = device
        self.d = pde_model.d
        self.loader = None
        self._loader_iter = None

    def _build_loader(self):
        self.loader = sampling.create_pde_loader(
            self.sampling_type, self.pde_model, self.sampling_settings, device=self.device,
        )
        self._loader_iter = iter(self.loader)

    def _next_batch(self):
        """One step = one batch. Reshuffle (re-iter) when the buffer is exhausted."""
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self.loader)
            return next(self._loader_iter)

    def train_adam_step(self, batch, use_sdgd=False, sdgd_num_dims=None):
        self.optimizer.zero_grad()
        with record_function("loss"):
            if use_sdgd:
                loss_pde = loss.sdgd_loss(batch[0], self.model, self.pde_model, batch[1], sdgd_num_dims)
            else:
                loss_pde = self.pde_model.pde_loss(batch[0], self.model, batch[1])
        with record_function("backward"):
            loss_pde.backward()
        with record_function("optimizer_step"):
            self.optimizer.step()
        return loss_pde

    def train_adam_minibatch(self, n_steps, n_steps_decay, resampling_frequency=2000, testing_frequency=100, use_sdgd=False, sdgd_num_dims=None):
        losses = {"total": [], "pde": []}
        l2_errs = []

        if self.profiler: self.profiler.make()
        self._build_loader()

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                print("New training data arrived!")
                self._build_loader()

            if self.profiler: self.profiler.start(si)

            batch = self._next_batch()
            loss_pde = self.train_adam_step(batch, use_sdgd=use_sdgd, sdgd_num_dims=sdgd_num_dims)
            loss_val = loss_pde.item()
            losses["total"].append(loss_val)
            losses["pde"].append(loss_val)

            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            if self.profiler: self.profiler.exit(si)

            if (si + 1) % testing_frequency == 0:
                log = f"Step {si+1}/{n_steps}, Loss: {loss_val:.6f}, pde: {loss_val:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}"
                if self.testing_suite is not None:
                    l2_err, l1_err, rel_err = self.testing_suite.test_model(self.model)
                    l2_errs.append(l2_err)
                    log += f", L2: {l2_err:.6f}, L1: {l1_err:.6f}, rel_max: {rel_err:.6f}"
                print(log)

        return losses, l2_errs



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

