import torch
from torch.profiler import record_function
import sampling, loss, utility



class PINN_Trainer:
    VALID_LOSS_KEYS = ("pde", "bc", "ic", "norm")

    def __init__(
        self, model, optimizer, scheduler, pde_model,
        sampling_type, sampling_settings,
        loss_weighting, testing_suite, active_losses=("pde", "bc", "ic", "norm"),
        profiler=None, device='cpu', dir_name=None, grad_clip_norm=None,
        memory_tracker=None,
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
        self.dir_name = dir_name
        self.grad_clip_norm = grad_clip_norm
        self.memory_tracker = memory_tracker
        self.memory_history = self.memory_tracker.history if self.memory_tracker is not None else None

        for k in active_losses:
            if k not in self.VALID_LOSS_KEYS:
                raise ValueError(f"Unknown loss term '{k}'. Valid: {self.VALID_LOSS_KEYS}")
        if "pde" not in active_losses:
            raise ValueError("'pde' must be in active_losses — it drives the step count.")
        self.active_losses = active_losses
        self.bundle = None  # populated in train_adam_minibatch

    def _init_fast_batch_sources(self):
        """Cache tensor-backed dataset views to avoid DataLoader collation overhead."""
        self._fast_batch_sources = {}
        for k in self.active_losses:
            loader = self.bundle.get(k)
            dataset = loader.dataset
            batch_size = loader.batch_size
            assert hasattr(dataset, "X")
            assert hasattr(dataset, "precomputed")
            n_samples = len(dataset.X)
            self._fast_batch_sources[k] = {
                "X": dataset.X,
                "precomputed": dataset.precomputed,
                "batch_size": batch_size,
                "n_samples": n_samples,
                "perm": torch.randperm(n_samples, device=dataset.X.device),
                "pos": 0,
            }

    def _next_batch_fast(self, k):
        src = self._fast_batch_sources[k]
        if src["pos"] >= src["n_samples"]:
            src["perm"] = torch.randperm(src["n_samples"], device=src["X"].device)
            src["pos"] = 0
        end = min(src["pos"] + src["batch_size"], src["n_samples"])
        idx = src["perm"][src["pos"]:end]
        src["pos"] = end
        return src["X"][idx], {name: values[idx] for name, values in src["precomputed"].items()}


    def _iter_epoch_batches_fast(self):
        """
        Yield one shuffled epoch, matching zip(*loaders) stop-at-shortest semantics.
        Preparation for minibatch loop.
        """
        # create a random permutation of indices
        for src in self._fast_batch_sources.values():
            src["perm"] = torch.randperm(src["n_samples"], device=src["X"].device)
            src["pos"] = 0
        n_batches = min(
            (src["n_samples"] + src["batch_size"] - 1) // src["batch_size"]
            for src in self._fast_batch_sources.values()
        )
        # lets build iterator
        for _ in range(n_batches):
            yield {k: self._next_batch_fast(k) for k in self.active_losses}

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

    def _loss_term(self, k, batch_term_objs, use_sdgd=False, sdgd_num_dims=None, use_causal_loss_weighting=False, t_discr=None, eps=1.0):
        if k == "pde":
            b = batch_term_objs["pde"]
            if use_sdgd:
                # actually: res = self.pde_model.pde_sdgd()
                res = loss.sdgd_res(b[0], self.model, self.pde_model, b[1], sdgd_num_dims)
            else:
                res = self.pde_model.pde_residual(b[0], self.model, b[1])
            if use_causal_loss_weighting:
                loss_pde, causal_weights, causal_losses = loss.causal_loss(b[0], res, t_discr, eps)
                self.causal_weights_hist_pde.append(causal_weights)
                self.causal_losses_hist_pde.append(causal_losses)
                return loss_pde
            else:
                return torch.mean(res**2)
        elif k == "bc":
            b = batch_term_objs["bc"]
            res = self.pde_model.bc_residual(b[0], self.model, b[1])
            if use_causal_loss_weighting:
                loss_bc, causal_weights, causal_losses = loss.causal_loss(b[0], res, t_discr, eps)
                self.causal_weights_hist_bc.append(causal_weights)
                self.causal_losses_hist_bc.append(causal_losses)
                return loss_bc
            else:
                return torch.mean(res**2)
        elif k == "ic":
            b = batch_term_objs["ic"]
            custom_ic_fn = self.sampling_settings.get("custom_ic_fn", None)
            if custom_ic_fn is not None:
                return torch.mean((
                    self.model(b[0]) - custom_ic_fn(b[0])
                )**2)
            else:
                return self.pde_model.ic_loss(b[0], self.model, b[1])
        elif k == "norm":
            b = batch_term_objs["norm"]
            return self.normalization_loss(b[0], self.model, b[1])
        else:
            raise ValueError(f"Unknown loss term '{k}'")

    def train_adam_step(self, batch_term_objs,
        use_sdgd=False, sdgd_num_dims=None,
        use_causal_loss_weighting=False, t_discr=None, eps=1.0
        ):
        """
        batch_term_objs:
            batch_term_objs['pde'] = [ batch (tensor), precomputed (dict_ ]
            batch_term_objs['bc'] = ...
            batch_term_objs['ic'] = ...
        """
        self.optimizer.zero_grad()
        with record_function("loss"):
            loss_terms = [
                self._loss_term(k, batch_term_objs, use_sdgd, sdgd_num_dims, use_causal_loss_weighting, t_discr, eps)
                for k in self.active_losses
            ]
        loss_terms_dict = {k: loss_terms[i].item() for i, k in enumerate(self.active_losses)}
        loss_value = self.loss_weighting.weight_loss(loss_terms)
        with record_function("backward"):
            loss_value.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        with record_function("optimizer_step"):
            self.optimizer.step()
        return loss_value, loss_terms_dict


    def _build_bundle(self, si, n_steps, resampling_frequency, use_time_adapt_sampling):
        if use_time_adapt_sampling:
            T = min( (si+resampling_frequency)/n_steps, 1.0 ) * self.T_max
            print(f"si={si}, T={T:.4f}")
            self.sampling_settings["T"] = T
            self.time_adapt_sampl_hist.append(T)
        self.bundle = sampling.create_dataloaders(
            self.sampling_type, self.model, self.pde_model,
            self.sampling_settings, self.active_losses, device=self.device,
        )
        self._bundle_iters = {k: iter(self.bundle[k]) for k in self.bundle}
        self._init_fast_batch_sources()
        return self.bundle

    def _next_batches(self):
        """Pull one batch per active loader. Reshuffle (re-iter) on exhaustion."""
        return {k: self._next_batch_fast(k) for k in self.active_losses}
        #batch_term_objs = {}
        #for k in self.active_losses:
        #    try:
        #        batch_term_objs[k] = next(self._bundle_iters[k])
        #    except StopIteration:
        #        self._bundle_iters[k] = iter(self.bundle[k])
        #        batch_term_objs[k] = next(self._bundle_iters[k])
        #return batch_term_objs

    def train_adam_minibatch(self,
        n_steps, n_steps_decay, resampling_frequency=2000, logging_frequency=100,
        one_batch_per_epoch=False,
        use_sdgd=False, sdgd_num_dims=None,
        use_causal_loss_weighting=False, t_discr=None, eps=1.0,
        use_time_adapt_sampling=False,
        prevent_resampling=False,
        gradnorm_update_freq=50,
        crit_loss_val=None,
    ):
        """Train the model using Adam optimizer.

        If one_batch_per_epoch=True, each `si` performs a single gradient step
        (one batch from each active loader). Otherwise iterates over all batches
        of the bundle per `si`.
        """
        if use_causal_loss_weighting:
            assert type(t_discr) == torch.Tensor, "t_discr not provided"
            self.causal_weights_hist_pde = []
            self.causal_losses_hist_pde = []
            if 'bc' in self.active_losses:
                self.causal_weights_hist_bc = []
                self.causal_losses_hist_bc = []
        if use_time_adapt_sampling:
            self.time_adapt_sampl_hist = []
            self.T_max = self.sampling_settings["T"]

        losses_hist_dict = {"total": [], **{k: [] for k in self.active_losses}}
        test_rel_l2 = {"pde": [], "ic": []}
        test_linf = {"pde": [], "ic": []}
        self.log_steps = []

        if self.profiler: self.profiler.make()

        self._build_bundle(0, n_steps, resampling_frequency, use_time_adapt_sampling)

        #allow_first_save = True
        #allow_first_save_RAR = True
        for si in range(n_steps):
            memory_sample = None
            if self.memory_tracker is not None:
                memory_sample = self.memory_tracker.sample(si + 1)

            if self.profiler: self.profiler.start(si)

            if (not prevent_resampling) and ( (si + 1) % resampling_frequency == 0 ):
            #if (si + 1) % resampling_frequency == 0:
                print("New training data arrived!")
                with record_function("resample"):
                    self._build_bundle(si, n_steps, resampling_frequency, use_time_adapt_sampling)

            if one_batch_per_epoch: 
                ## create a new iterator and get the entire dataset
                #batch_term_objs = {
                #    k: next(iter(self.bundle[k]))
                #    for k in self.active_losses
                #}
                # iterate to the next batch
                batch_term_objs = self._next_batches()
                loss_value, last_losses_dict = self.train_adam_step(
                    batch_term_objs, use_sdgd, sdgd_num_dims,
                    use_causal_loss_weighting, t_discr, eps
                )
            else:
                batch_iter = self._iter_epoch_batches_fast()
                #loaders = [self.bundle[k] for k in self.active_losses]
                #batch_iter = (
                #    dict(zip(self.active_losses, batches))
                #    for batches in zip(*loaders)
                #)
                for batch_term_objs in batch_iter:
                    loss_value, last_losses_dict = self.train_adam_step(
                        batch_term_objs, use_sdgd, sdgd_num_dims,
                        use_causal_loss_weighting, t_discr, eps
                    )
            losses_hist_dict["total"].append(loss_value.item())
            for k in self.active_losses:
                losses_hist_dict[k].append(last_losses_dict[k])

            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            if isinstance(self.loss_weighting, loss.GradnormAdaptLambdas):
                if (si + 1) % gradnorm_update_freq == 0:
                    self.optimizer.zero_grad()
                    loss_terms = [
                        self._loss_term(k, batch_term_objs, use_sdgd, sdgd_num_dims, use_causal_loss_weighting, t_discr, eps)
                        for k in self.active_losses
                    ]
                    self.loss_weighting.update(loss_terms, self.model)
            elif isinstance(self.loss_weighting, loss.MaxAdaptLambdas):
                    loss_terms = [
                        self._loss_term(k, batch_term_objs, use_sdgd, sdgd_num_dims, use_causal_loss_weighting, t_discr, eps)
                        for k in self.active_losses
                    ]
                    self.loss_weighting.update(loss_terms)


            if self.profiler: self.profiler.exit(si)

            if (si + 1) % logging_frequency == 0:
                self.log_steps.append(si+1)
                parts = [f"Step {si+1}/{n_steps}", f"Loss: {loss_value.item()}"]
                for k in self.active_losses:
                    parts.append(f"{k}: {last_losses_dict[k]}")
                parts.append(f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                log = ", ".join(parts)
                print(log)
                if self.testing_suite is not None:
                    test_dict_rel_l2, test_dict_linf = self.testing_suite.test_model(self.model, self.pde_model, device=self.device) #,test_bs=self.sampling_settings["bs"])
                    test_log_rel_l2 = " - Testing: rel L2  | "
                    for k,v in test_dict_rel_l2.items():
                        test_rel_l2[k].append(v)
                        test_log_rel_l2 += f"{k}: {v:.6f}, "
                    print(test_log_rel_l2[:-2]) # no ', '
                    test_log_linf = " - Testing: Linf  | "
                    for k,v in test_dict_linf.items():
                        test_linf[k].append(v)
                        test_log_linf += f"{k}: {v:.6f}, "
                    print(test_log_linf[:-2]) # no ', '
                if memory_sample is not None:
                    mem_log = f" - {self.memory_tracker.format_sample(memory_sample)}"
                    print(mem_log)
                if use_causal_loss_weighting:
                    pass
                    #print(len(self.causal_weights_hist_pde), self.causal_weights_hist_pde[-1])
                    #print(len(self.causal_losses_hist_pde), self.causal_losses_hist_pde[-1])
                print(f" - {next(self.model.parameters()).device}, {loss_value.device}, {batch_term_objs['pde'][0].device.type}")

            #frac = si / n_steps
            #frac_target = 0.5
            #if allow_first_save_RAR and (frac > frac_target):
            #    X = self._fast_batch_sources['pde']['X']
            #    torch.save({f"{self.dir_name}/X_{frac_target}": X}, f"RAR_X_{frac_target}.pth")
            #    allow_first_save_RAR = False

            #if allow_first_save and (loss_value < 1e-5):
            #    print("++++++++++++++++++++++++++++++++++++++")
            #    print("Saving the model!!")
            #    print("++++++++++++++++++++++++++++++++++++++")
            #    torch.save(self.model.state_dict(), f'{self.dir_name}/model_1e-5_{si}.pth')
            #    print("\nModel saved.")
            #    allow_first_save = False
            #if loss_value < 1e-6:
            #    print("++++++++++++++++++++++++++++++++++++++")
            #    print("Saving the model!!")
            #    print("++++++++++++++++++++++++++++++++++++++")
            #    torch.save(self.model.state_dict(), f'{self.dir_name}/model_1e-6_{si}.pth')
            #    print("\nModel saved.")
            #    return losses_hist_dict, test_rel_l2, test_linf


            if (crit_loss_val is not None) and loss_value < crit_loss_val:
                return losses_hist_dict, test_rel_l2, test_linf

        #X = self._fast_batch_sources['pde']['X']
        #torch.save({f"X_last": X}, f"RAR_X_last.pth")
        return losses_hist_dict, test_rel_l2, test_linf


    def train_lbfgs(self,
        n_steps, n_steps_decay, resampling_frequency=2000, logging_frequency=100,
        one_batch_per_epoch=False,
        use_sdgd=False, sdgd_num_dims=None,
        use_causal_loss_weighting=False, t_discr=None, eps=1.0,
        use_time_adapt_sampling=False,
        prevent_resampling=False,
        gradnorm_update_freq=50,
        crit_loss_val=None,
    ):
        """Train the model using L-BFGS with one minibatch per step."""
        print(f"\n{'='*60}")
        print(f"Starting L-BFGS (n_steps={n_steps})")
        print(f"{'='*60}\n")

        if use_causal_loss_weighting:
            assert type(t_discr) == torch.Tensor, "t_discr not provided"
            self.causal_weights_hist_pde = []
            self.causal_losses_hist_pde = []
            if 'bc' in self.active_losses:
                self.causal_weights_hist_bc = []
                self.causal_losses_hist_bc = []
        if use_time_adapt_sampling:
            self.time_adapt_sampl_hist = []
            self.T_max = self.sampling_settings["T"]

        if self.profiler: self.profiler.make()

        losses_hist_dict = {"total": [], **{k: [] for k in self.active_losses}}
        test_rel_l2 = []
        test_linf = []
        self.log_steps = []

        self.last_losses = None
        self.last_total_loss = None
        self.i = 0

        def build_closure(batch_term_objs):
            def closure():
                # optimizer.zero_grad()
                # calc loss
                # store losses
                # loss.backward()
                # return loss
                self.optimizer.zero_grad()
                with record_function("loss"):
                    per_term = [
                        self._loss_term(
                            k, batch_term_objs, use_sdgd, sdgd_num_dims,
                            use_causal_loss_weighting, t_discr, eps
                        )
                        for k in self.active_losses
                    ]
                loss_value = self.loss_weighting.weight_loss(per_term)
                #print(f"s={self.s}, i={self.i}: loss = {loss_value}")
                self.i += 1
                with record_function("backward"):
                    loss_value.backward()
                self.last_losses = {k: per_term[i].item() for i, k in enumerate(self.active_losses)}
                self.last_total_loss = loss_value.item()

                return loss_value
            return closure

        self._build_bundle(0, n_steps, resampling_frequency, use_time_adapt_sampling)

        for si in range(n_steps):
            memory_sample = None
            if self.memory_tracker is not None:
                memory_sample = self.memory_tracker.sample(si + 1)

            if self.profiler: self.profiler.start(si)

            if (not prevent_resampling) and ( (si + 1) % resampling_frequency == 0 ):
                print("New training data arrived!")
                with record_function("resample"):
                    self._build_bundle(si, n_steps, resampling_frequency, use_time_adapt_sampling)

            # prepare the batch
            if one_batch_per_epoch: 
                batch_term_objs = self._next_batches()
                closure_step_fn = build_closure(batch_term_objs)
                self.i = 0
                self.s = si
                if (si + 1) % logging_frequency == 0:
                    params_before = [
                        p.detach().clone()
                        for p in self.model.parameters()
                        if p.requires_grad
                    ]
                    self.optimizer.step(closure_step_fn)
                    max_delta = 0.0
                    sq_delta = 0.0
                    for p_before, p_after in zip(params_before, self.model.parameters()):
                        if not p_after.requires_grad:
                            continue
                        d = (p_after.detach() - p_before).reshape(-1)
                        if d.numel() > 0:
                            max_delta = max(max_delta, d.abs().max().item())
                            sq_delta += torch.dot(d, d).item()
                    param_delta_l2 = sq_delta ** 0.5
                    print(
                        f"closure_calls={self.i}, "
                        f"param_delta_l2={param_delta_l2:.3e}, "
                        f"param_delta_max={max_delta:.3e}"
                    )
                else:
                    self.optimizer.step(closure_step_fn)
                with torch.enable_grad():
                    per_term_post = [
                        self._loss_term(
                            k, batch_term_objs, use_sdgd, sdgd_num_dims,
                            use_causal_loss_weighting, t_discr, eps
                        )
                        for k in self.active_losses
                    ]
                    loss_value = self.loss_weighting.weight_loss(per_term_post)
            else:
                raise TypeError
            losses_hist_dict["total"].append(self.last_total_loss)
            for k in self.active_losses:
                losses_hist_dict[k].append(self.last_losses[k])

            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            if isinstance(self.loss_weighting, loss.GradnormAdaptLambdas):
                if (si + 1) % gradnorm_update_freq == 0:
                    self.optimizer.zero_grad()
                    loss_terms = [
                        self._loss_term(k, batch_term_objs, use_sdgd, sdgd_num_dims, use_causal_loss_weighting, t_discr, eps)
                        for k in self.active_losses
                    ]
                    self.loss_weighting.update(loss_terms, self.model)
            elif isinstance(self.loss_weighting, loss.MaxAdaptLambdas):
                    loss_terms = [
                        self._loss_term(k, batch_term_objs, use_sdgd, sdgd_num_dims, use_causal_loss_weighting, t_discr, eps)
                        for k in self.active_losses
                    ]
                    self.loss_weighting.update(loss_terms)


            if self.profiler: self.profiler.exit(si)

            if (si + 1) % logging_frequency == 0:
                self.log_steps.append(si+1)
                parts = [f"Step {si+1}/{n_steps}", f"Loss_ret: {loss_value.item()}", f"Loss_tot: {losses_hist_dict['total'][-1]}"]
                for k in self.active_losses:
                    parts.append(f"{k}: {self.last_losses[k]}")
                parts.append(f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                log = ", ".join(parts)
                print(log)
                if self.testing_suite is not None:
                    test_dict_rel_l2, test_dict_linf = self.testing_suite.test_model(self.model, self.pde_model, device=self.device) #,test_bs=self.sampling_settings["bs"])
                    test_log_rel_l2 = " - Testing: rel L2  | " + \
                        ", ".join([f"{k}: {v:.6f}" for k,v in test_dict_rel_l2.items()])
                    test_rel_l2.append(test_dict_rel_l2)
                    print(test_log_rel_l2)
                    test_log_linf = " - Testing: Linf  | " + \
                        ", ".join([f"{k}: {v:.6f}" for k,v in test_dict_linf.items()])
                    test_linf.append(test_dict_linf)
                    print(test_log_linf)
                if memory_sample is not None:
                    mem_log = f" - {self.memory_tracker.format_sample(memory_sample)}"
                    print(mem_log)
                if use_causal_loss_weighting:
                    pass
                    #print(len(self.causal_weights_hist_pde), self.causal_weights_hist_pde[-1])
                    #print(len(self.causal_losses_hist_pde), self.causal_losses_hist_pde[-1])
                print(f" - {next(self.model.parameters()).device}, {loss_value.device}, {batch_term_objs['pde'][0].device.type}")

            if (crit_loss_val is not None) and loss_value < crit_loss_val:
                return losses_hist_dict, test_rel_l2, test_linf

        return losses_hist_dict, test_rel_l2, test_linf




class PINN_Trainer_1k:
    """Minimal trainer: only the PDE residual loss, single DataLoader.
    TD: delete all metions of total loss - keep only the pde loss

    Default shape: dataloader holds ~100_000 points, each gradient step
    consumes bs points (~1000). Sampling mode is chosen per-call:
      - sampling_type="domain"       → vanilla-PINN-style uniform/LHS
      - sampling_type="trajectories" → Euler-Maruyama SDE trajectory bank
    """

    def __init__(
        self, model, optimizer, scheduler, pde_model,
        sampling_type, sampling_settings,
        testing_suite=None, profiler=None, device='cpu', dir_name=None,
        grad_clip_norm=None,
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
        self._loader_iter = None,
        self.dir_name = dir_name
        self.grad_clip_norm = grad_clip_norm

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
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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
                    l2_err, l1_err, rel_err = self.testing_suite.test_model(self.model, device=self.device)
                    l2_errs.append(l2_err)
                    log += f", L2: {l2_err:.6f}, L1: {l1_err:.6f}, rel_max: {rel_err:.6f}"
                print(log)

            #if si == 29_999 or si == 49_999 or si == 69_999 or si == 89_999:
            #    loss_name = f'{self.dir_name}/training_loss'
            #    l2_name = f'{self.dir_name}/training_l2_error'
            #    torch.save(self.model.state_dict(), f'{self.dir_name}/model_{si}.pth')
            #    torch.save({k: torch.tensor(v) for k, v in losses.items()}, f'{loss_name}_{si}.pth')
            #    torch.save(torch.tensor(l2_errs), f'{l2_name}_{si}.pth')
            #    print("\nResults saved.")

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
