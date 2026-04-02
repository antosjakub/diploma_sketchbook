import torch
import argparse
from torch.profiler import record_function
import sampling, loss, architecture, utility, pde_models

parser = argparse.ArgumentParser()
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="128,128,128", type=str, help="")
parser.add_argument("--n_steps", default=10_000, type=int, help="")
parser.add_argument("--n_steps_decay", default=2000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--n_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--n_test_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--resampling_frequency", default=2000, type=int, help="")
parser.add_argument("--bs", default=512, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# L-BFGS
parser.add_argument("--n_steps_lbfgs", default=0, type=int, help="Number of L-BFGS steps after Adam. Set to 0 to skip.")
parser.add_argument("--n_steps_log_lbfgs", default=100, type=int, help="")
parser.add_argument("--lr_lbfgs", default=1.0, type=float, help="Learning rate for L-BFGS (typically 1.0).")
# smart Defaults
parser.add_argument("--l2_stop_crit", default=0.01, type=float, help="")
parser.add_argument("--l2_stop_crit_lbfgs", default=0.001, type=float, help="")
# 
parser.add_argument("--output_dir_name", default="run_latest/", type=str, help="")
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
parser.add_argument("--use_weak_form", action="store_true", help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")



class TestingSuite:
    def __init__(self, d, keep_in_cache=True):
        self.d = d
        self.test_file_exists = True
        self.test_file_path = ""
        self.keep_in_cache = keep_in_cache
    
    def connect_test_data(self, file_path: str):
        import os
        if os.path.exists(file_path):
            payload = torch.load(file_path, map_location="cpu")
            metadata = payload["metadata"]
            if (metadata["d"] != self.d):
                raise ValueError(
                    f"Dimension mismatch. Testing suite has d={self.d}, but the loaded data have d={metadata['d']}."
                )
            assert payload["data"]["X"].shape[1] == self.d+1
            assert payload["data"]["u_true"].shape[1] == 1
            assert payload["data"]["X"].shape[0] == payload["data"]["u_true"].shape[0]
        if self.keep_in_cache: self.payload = payload
        self.test_file_exists = True
        self.test_file_path = file_path


    def make_test_data(self, pde_model, n_test_calloc_points, file_path, sampling_strategy="lhs", seed=4242):
        # Create once, deterministic.
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            X, _, _ = sampling.sample_collocation_points(
                self.d,
                n_test_calloc_points,
                0,
                0,
                sampling_strategy=sampling_strategy,
                device="cpu",
            )

        # Optional: pre-store analytic truth to avoid recomputing every test call.
        with torch.no_grad():
            u_true = pde_model.u_analytic(X)

        payload = {
            "metadata": {
                "d": self.d,
                "N": n_test_calloc_points,
                "sampling_strategy": sampling_strategy,
                "seed": seed,
            },
            "data": {
                "X": X,
                "u_true": u_true,
            }
        }
        if self.keep_in_cache:
            self.payload = payload
        else:
            torch.save(payload, file_path)
        self.test_file_exists = True
        self.test_file_path = file_path


    def test_model(self, model, test_bs=100_000, device="cpu"):

        import time
        a = time.time()

        if not self.test_file_exists:
            raise ValueError(
                "Make or Connect test data first before testing."
            )

        try:
            if self.keep_in_cache:
                payload = self.payload
            else:
                payload = torch.load(self.test_file_path)
            X = payload["data"]["X"]
            u_true = payload["data"]["u_true"]
        except:
            raise "Unable to load the testing data."

        N = X.shape[0]
        sum_l2 = 0.0
        sum_l1 = 0.0
        max_rel = 0.0
        eps = 1e-10

        model.eval()
        with torch.no_grad():
            for i in range(0, N, test_bs):
                j = min(i + test_bs, N)
                X_chunk = X[i:j]
                u_true_chunk = u_true[i:j]

                u_pred = model(X_chunk)
                err = u_pred - u_true_chunk

                sum_l2 += torch.sum(err**2).item()
                sum_l1 += torch.sum(err.abs()).item()

                rel_chunk = ( (err-eps) / (u_true_chunk-eps) ).abs().max().item()
                if rel_chunk > max_rel:
                    max_rel = rel_chunk
        model.train()

        l2_err = (sum_l2 / N)**(1/2)
        l1_err = sum_l1 / N
        rel_err = max_rel

        b = time.time()
        print(f"Testing took: {b-a}s")
        return l2_err, l1_err, rel_err



class PINN_Trainer:
    def __init__(self, model, optimizer, scheduler, pde_model, loss_weighting, testing_suite, profiler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
        self.loss_weighting = loss_weighting
        self.profiler = profiler
        self.testing_suite = testing_suite
        self.device = device
        self.d = self.pde_model.d

    def train_adam_step(self, batch_pde, batch_bc, batch_ic, use_sdgd=False, sdgd_num_dims=None):
        """
        Train a single step of the model.
        """
        #batch_pde, batch_bc, batch_ic, u_bc_target, u_ic_target
        self.optimizer.zero_grad()

        # Compute individual losses
        with record_function("loss"):
            batch_pde[0].requires_grad = True
            if use_sdgd:
                loss_pde = loss.sdgd_loss(batch_pde[0], self.model, self.pde_model, batch_pde[1], sdgd_num_dims)
            else:
                #loss_pde = loss.causal_pde_loss(batch_pde[0], self.model, self.pde_model, batch_pde[1])
                loss_pde = self.pde_model.pde_loss(batch_pde[0], self.model, batch_pde[1])
            loss_bc = self.pde_model.bc_loss(batch_bc[0], self.model, batch_bc[1])
            loss_ic = self.pde_model.ic_loss(batch_ic[0], self.model, batch_ic[1])

        # Weighted loss
        loss_value = self.loss_weighting.weight_loss([loss_pde, loss_bc, loss_ic])

        # Backward pass
        with record_function("backward"):
            loss_value.backward()
        with record_function("optimizer_step"):
            self.optimizer.step()

        return loss_value, (loss_pde.item(), loss_bc.item(), loss_ic.item())
    

    def train_adam_step_accumulated(self, batch_iterator):
        self.optimizer.zero_grad()

        n_cycles = 0
        loss_pde, loss_bc, loss_ic = 0.0, 0.0, 0.0
        for batch_pde, batch_bc, batch_ic in batch_iterator:
            n_cycles += 1
            batch_pde[0].requires_grad = True
            # Compute individual losses
            with record_function("loss"):
                loss_pde += self.pde_model.pde_loss(batch_pde[0], self.model, batch_pde[1])
                loss_bc += self.pde_model.bc_loss(batch_bc[0], self.model, batch_bc[1])
                loss_ic += self.pde_model.ic_loss(batch_ic[0], self.model, batch_ic[1])
        # Weighted loss
        loss_pde /= n_cycles
        loss_bc /= n_cycles
        loss_ic /= n_cycles
        loss_value = self.loss_weighting.weight_loss(
            [loss_pde, loss_bc, loss_ic]
        )

        # Backward pass
        with record_function("backward"):
            loss_value.backward()
        with record_function("optimizer_step"):
            self.optimizer.step()
        
        return loss_value, (loss_pde.item(), loss_bc.item(), loss_ic.item())



    def train_adam_fullbatch(self, n_steps, n_steps_decay, n_calloc_points, n_test_calloc_points, resampling_frequency=2000, testing_frequency=100):
        """
        Train the model using Adam optimizer.
        """
        losses = []
        l2_errs = []

        n_points_interior = 8*n_calloc_points//10
        n_points_boundary = n_calloc_points//10
        n_points_initial = n_calloc_points//10

        if self.profiler: self.profiler.make()

        # Generate training data
        X_interior, X_boundary, X_initial = sampling.sample_collocation_points(
            self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
        )
        # other:
        #pre_precompute = 
        #other = self.pde_model.precompute(X_interior, X_boundary, X_initial)
        u_bc_target = self.pde_model.u_bc(X_boundary)
        u_ic_target = self.pde_model.u_ic(X_initial)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                #X_interior, X_boundary, X_initial = self.resample_training_data()
                print("New training data arrived?")
                X_interior, X_boundary, X_initial = sampling.sample_collocation_points(
                    self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
                )
                u_bc_target = self.pde_model.u_bc(X_boundary)
                u_ic_target = self.pde_model.u_ic(X_initial)
    
            # Start profiler context
            if self.profiler: self.profiler.start(si)

            #X_interior = X_interior.detach().requires_grad_(True)
            loss_value, (loss_pde, loss_bc, loss_ic) = self.train_adam_step(X_interior, X_boundary, X_initial, u_bc_target, u_ic_target)
            losses.append(loss_value.item())

            # Step scheduler
            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            # Exit profiler context after the last profiled step
            if self.profiler: self.profiler.exit(si)

            # Print progress
            if (si + 1) % testing_frequency == 0:
                l2_err, l1_err, rel_err = self.test_model(n_test_calloc_points)
                l2_errs.append(l2_err)
                print(f'Step {si+1}/{n_steps}, Loss: {loss_value.item():.6f}, '
                      f'PDE: {loss_pde:.6f}, '
                      f'BC: {loss_bc:.6f}, '
                      f'IC: {loss_ic:.6f}, '
                      f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                      f'L2: {l2_err:.6f}, '
                      f'L1: {l1_err:.6f}, '
                      f'rel_max: {rel_err:.6f}'
                )

        return losses, l2_errs




    def train_adam_minibatch(self, bs, n_steps, n_steps_decay, n_calloc_points, resampling_frequency=2000, testing_frequency=100, use_rbas=False, use_sdgd=False, sdgd_num_dims=None):
        """
        Train the model using Adam optimizer.
        """
        losses = []
        l2_errs = []

        if self.profiler: self.profiler.make()

        # batches 
        loader_interior, loader_bc, loader_ic = sampling.create_dataloaders(self.d, n_calloc_points, bs, self.model, self.pde_model)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                print("New training data arrived!")
                loader_interior, loader_bc, loader_ic = sampling.create_dataloaders(self.d, n_calloc_points, bs, self.model, self.pde_model, use_rbas=use_rbas)
    
            # Start profiler context
            if self.profiler: self.profiler.start(si)
        
            # batches
            batch_iterator = zip(loader_interior, loader_bc, loader_ic)
            if True:
                for batch_pde, batch_bc, batch_ic in batch_iterator:
                    loss_value, (loss_pde, loss_bc, loss_ic) = self.train_adam_step(batch_pde, batch_bc, batch_ic, use_sdgd=use_sdgd, sdgd_num_dims=sdgd_num_dims)
            else:
                loss_value, (loss_pde, loss_bc, loss_ic) = self.train_adam_step_accumulated(batch_iterator)
            losses.append(loss_value.item())


            # Step scheduler
            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            if type(self.loss_weighting).__name__ == 'AdaptiveWeights':
                if (si + 1) % 50 == 0:
                    self.optimizer.zero_grad()
                    loss_pde = self.pde_model.pde_loss(batch_pde, self.model)
                    loss_bc = self.pde_model.bc_loss(batch_bc, self.model)
                    loss_ic = self.pde_model.ic_loss(batch_ic, self.model)
                    self.loss_weighting.update([loss_pde, loss_bc, loss_ic], self.model)

            # Exit profiler context after the last profiled step
            if self.profiler: self.profiler.exit(si)

            # Print progress
            if (si + 1) % testing_frequency == 0:
                # do not resample - sample at the beggining, then reuse
                l2_err, l1_err, rel_err = self.testing_suite.test_model(model)
                l2_errs.append(l2_err)
                print(f'Step {si+1}/{n_steps}, Loss: {loss_value.item():.6f}, '
                      f'PDE: {loss_pde:.6f}, '
                      f'BC: {loss_bc:.6f}, '
                      f'IC: {loss_ic:.6f}, '
                      f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                      f'L2: {l2_err:.6f}, '
                      f'L1: {l1_err:.6f}, '
                      f'rel_max: {rel_err:.6f}'
                )

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

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Arguments
    args = parser.parse_args([] if "__file__" not in globals() else None)
    torch.manual_seed(args.seed)
    d = args.d # space dims
    D = d + 1 # space + time dims
    layers = utility.layers_from_string(args.layers)
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")

    # Prepare storage
    import os
    dir_name = args.output_dir_name
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    os.makedirs(dir_name, exist_ok=True)    
    



    # Select the model architecture
    if args.starting_model:
        model = torch.load(args.starting_model, weights_only=False)
    else:
        #model = architecture.PINN(D, layers, head_fn=lambda x: torch.sinh(5*x)/100).to(device)
        model = architecture.PINN(D, layers).to(device)
        #model = PINN_SeparableTimes(D, layers).to(device)
        #model = PINN_SepTime(D, layers).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)
    

    # PDE equation
    #pde_model = pde_models.HeatEquationWithSource(d)
    pde_model = pde_models.TravellingGaussPacket(d, gamma=1)
    #pde_model = pde_models.HeatEquation(d)
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

    # Preparation time
    losses = [] 
    l2_errs = [] 


    # Train the model
    if args.n_steps > 0:
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Option 1: ExponentialLR
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        # Initialize loss weighting and profiler
        if args.use_adaptive_weights:
            loss_weighting = loss.AdaptiveWeights(weights=torch.tensor([args.lambda_pde, args.lambda_bc, args.lambda_ic]))
        else:
            loss_weighting = loss.ConstantWeights(weights=[args.lambda_pde, args.lambda_bc, args.lambda_ic])

        profiler = utility.Profiler(report_filename=f"{dir_name}/{args.profiler_report_filename}.txt", start_step=100, end_step=110) if args.enable_profiler else None

        sdgd_num_dims = args.sdgd_num_dims if args.sdgd_num_dims is not None else d
        if args.use_sdgd:
            print(f"Using SDGD with {sdgd_num_dims} dimensions (d={d})")
        else:
            print(f"Using regular Adam training.")

        import time
        t1 = time.time()
        testing_suite = TestingSuite(d)
        testing_suite.make_test_data(pde_model, args.n_test_calloc_points, f"{dir_name}/testing_data.pt")
        #testing_suite.connect_test_data(f"{dir_name}/testing_data.pt")
        trainer = PINN_Trainer(model, optimizer, scheduler, pde_model, loss_weighting, testing_suite, profiler, device)
        losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
        #losses_adam, l2_errs_adam = trainer.train_adam_fullbatch(
            bs=args.bs,
            n_steps=args.n_steps,
            n_steps_decay=args.n_steps_decay,
            n_calloc_points=args.n_calloc_points,
            resampling_frequency=args.resampling_frequency,
            testing_frequency=args.testing_frequency,
            use_rbas=args.use_rbas,
            use_sdgd=args.use_sdgd,
            sdgd_num_dims=sdgd_num_dims,
        )
        losses += losses_adam
        l2_errs += l2_errs_adam
        print("\nAdam training complete!")
        t2 = time.time()
        h,m,s = utility.get_duration(t2-t1)
        train_time_str = f"Adam training completed in: "
        train_time_str += f"{h} hours " if h > 0 else ""
        train_time_str += f"{m} minutes " if m > 0 else ""
        train_time_str += f"{s} seconds"
        print(train_time_str)
        torch.save(model, f'{dir_name}/model_adam.pth')

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
        losses += losses_lbfgs
        l2_errs += l2_errs_lbfgs
        print("\nL-BFGS fine-tuning complete!")
        torch.save(model, f'{dir_name}/model_adam_lbfgs.pth')
    
    # Dan
    print("\nTraining complete!")
    
    import json
    with open(f'{dir_name}/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({"model_class": type(model).__name__, "args": args.__dict__}, f, ensure_ascii=False, indent=4)

    pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')

    loss_name = f'{dir_name}/training_loss'
    l2_name = f'{dir_name}/training_l2_error'
    # Save the results
    torch.save(model.state_dict(), f'{dir_name}/model.pth')
    torch.save(torch.tensor(losses), f'{loss_name}.pth')
    torch.save(torch.tensor(l2_errs), f'{l2_name}.pth')
    print("\nResults saved.")


    # Plot results
    n_steps_log = args.testing_frequency
    n_logged_pnts = len(l2_errs)
    steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)
    import visualize_training_metrics
    visualize_training_metrics.plot_loss(losses, loss_name)
    visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)
    import visualize_solution_3plots
    visualize_solution_3plots.plot_3(model, pde_model.u_analytic, d, dir_name)
    import visualize_solution_3anims
    visualize_solution_3anims.anim_3(model, pde_model.u_analytic, d, dir_name)