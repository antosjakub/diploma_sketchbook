import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="64,64,64", type=str, help="")
parser.add_argument("--n_steps", default=10_000, type=int, help="")
parser.add_argument("--n_steps_decay", default=1000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--n_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--n_test_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--resampling_frequency", default=2000, type=int, help="")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--bs", default=512, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--lr", default=1e-3, type=float, help="")
# L-BFGS
parser.add_argument("--n_steps_lbfgs", default=0, type=int, help="Number of L-BFGS steps after Adam. Set to 0 to skip.")
parser.add_argument("--n_steps_log_lbfgs", default=100, type=int, help="")
parser.add_argument("--lr_lbfgs", default=1.0, type=float, help="Learning rate for L-BFGS (typically 1.0).")
# smart Defaults
parser.add_argument("--l2_stop_crit", default=0.01, type=float, help="")
parser.add_argument("--l2_stop_crit_lbfgs", default=0.001, type=float, help="")
# 
parser.add_argument("--output_dir_name", default="run_latest/", type=str, help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
parser.add_argument("--use_weak_form", action="store_true", help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")




class PINN_Trainer:
    def __init__(self, model, optimizer, scheduler, pde_model, loss_weighting, device='cpu', profiler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
        self.loss_weighting = loss_weighting
        self.profiler = profiler
        self.device = device
        self.d = self.pde_model.d

    def test_model(self, n_test_calloc_points):
        n_test_interior = 8*n_test_calloc_points//10
        n_test_boundary = n_test_calloc_points//10
        n_test_initial = n_test_calloc_points//10

        X_interior_test, X_boundary_test, X_initial_test = sample_collocation_points(
            self.d, n_test_interior, n_test_boundary, n_test_initial, device=self.device
        )

        u_pred = self.model(X_interior_test)
        u_true = self.pde_model.u_analytic(X_interior_test)
        l2_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
        l1_err = torch.mean((u_pred - u_true).abs()).item()
        rel_err = torch.max( ((u_pred - u_true-1e-7)/(u_true+1e-7)).abs() ).item()
        return l2_err, l1_err, rel_err


    def resample_training_data(self):
        X_int1 = residual_based_adaptive_sampling(d, self.pde_model.pde_residual, n_new=2*n_points_pde//3, n_candidates=5*n_points_pde, sampling_strategy="latin", picking_criterion="multinomial")
        X_int2 = residual_based_adaptive_sampling(d, pde_residual, n_new=n_points_pde//3, n_candidates=5*n_points_pde, sampling_strategy="latin", picking_criterion="top_k")
        X_interior = torch.cat([X_int1, X_int2], dim=0)
        _, X_boundary, X_initial = sample_collocation_points(
            d, 0, n_points_bc, n_points_ic, device
        )
        return X_interior, X_boundary, X_initial


    def train_adam_step(self, batch_pde, batch_bc, batch_ic, u_bc_target, u_ic_target):
        """
        Train a single step of the model.
        """
        self.optimizer.zero_grad()

        # Compute individual losses
        with record_function("loss"):
            #loss_pde = sdgd_loss(model, X_interior, pde_residual, None, 3)
            loss_pde = pde_loss(self.model, batch_pde, self.pde_model.pde_residual)
            loss_bc = boundary_condition_loss(self.model, batch_bc, u_bc_target)
            loss_ic = initial_condition_loss(self.model, batch_ic, u_ic_target)

        # Weighted loss
        loss = self.loss_weighting.weight_loss([loss_pde, loss_bc, loss_ic])

        # Backward pass
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer_step"):
            self.optimizer.step()

        return loss, (loss_pde.item(), loss_bc.item(), loss_ic.item())

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
        X_interior, X_boundary, X_initial = sample_collocation_points(
            self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
        )
        u_bc_target = self.pde_model.u_bc(X_boundary)
        u_ic_target = self.pde_model.u_ic(X_initial)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                #X_interior, X_boundary, X_initial = self.resample_training_data()
                print("New training data arrived?")
                X_interior, X_boundary, X_initial = sample_collocation_points(
                    self.d, n_points_interior, n_points_boundary, n_points_initial, device=self.device
                )
                u_bc_target = self.pde_model.u_bc(X_boundary)
                u_ic_target = self.pde_model.u_ic(X_initial)
    
            # Start profiler context
            if self.profiler: self.profiler.start(si)

            X_interior = X_interior.detach().requires_grad_(True)
            loss, (loss_pde, loss_bc, loss_ic) = self.train_adam_step(X_interior, X_boundary, X_initial, u_bc_target, u_ic_target)

            # Step scheduler
            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            losses.append(loss.item())

            # Exit profiler context after the last profiled step
            if self.profiler: self.profiler.exit(si)

            # Print progress
            if (si + 1) % testing_frequency == 0:
                l2_err, l1_err, rel_err = self.test_model(n_test_calloc_points)
                l2_errs.append(l2_err)
                print(f'Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
                      f'PDE: {loss_pde.item():.6f}, '
                      f'BC: {loss_bc.item():.6f}, '
                      f'IC: {loss_ic.item():.6f}, '
                      f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                      f'L2: {l2_err:.6f}, '
                      f'L1: {l1_err:.6f}, '
                      f'rel_max: {rel_err:.6f}'
                )

        return losses, l2_errs


    def train_adam_minibatch(self, bs, n_steps, n_steps_decay, n_points_calloc, resampling_frequency=2000, testing_frequency=100):
        """
        Train the model using Adam optimizer.
        """
        losses = []
        l2_errs = []


        if self.profiler: self.profiler.make()

        # batches 
        loader_interior, loader_bc, loader_ic = create_dataloaders(self.d, n_points_calloc, bs, self.pde_model.u_bc, self.pde_model.u_ic)

        # Generate training data
        #X_interior, X_boundary, X_initial = sample_collocation_points(
        #    self.d, n_points_calloc, 0, 0, device=self.device
        #)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                print("New training data arrived?")
                #X_interior, X_boundary, X_initial = self.resample_training_data()
                loader_interior, loader_bc, loader_ic = create_dataloaders(self.d, n_points_calloc, bs, self.pde_model.u_bc, self.pde_model.u_ic)
    
            # Start profiler context
            if self.profiler: self.profiler.start(si)
        
            # batches
            batch_iterator = zip(loader_interior, loader_bc, loader_ic)
            for (batch_pde,), (batch_bc, u_bc_target), (batch_ic, u_ic_target) in batch_iterator:
                batch_pde.requires_grad = True
                loss = self.train_adam_step(batch_pde, batch_bc, batch_ic)

            # Step scheduler
            if (si + 1) % n_steps_decay == 0:
                scheduler.step()

            losses.append(loss.item())

            # Exit profiler context after the last profiled step
            if self.profiler: self.profiler.exit(si)

            # Print progress
            if (si + 1) % testing_frequency == 0:
                test_model(model, X_interior_test, X_boundary_test, X_initial_test)

    
    return losses, l2_errs

    def train_lbfgs():







def train_pinn_lbfgs(
        model,
        pde_residual, bc_residual, ic_residual,
        u_analytic,
        d,
        n_steps=500,
        n_steps_log=100,
        n_points_pde=2000, n_points_bc=400, n_points_ic=400,
        lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
        lr=1.0,
        l2_stop_crit=0.001,
        compute_laplace=True,
        device='cpu'
    ):
    """Fine-tune the PINN model with L-BFGS after Adam pre-training."""

    # Sample fixed collocation points (L-BFGS works best with a fixed dataset)
    X_interior, X_boundary, X_initial = sample_collocation_points(
        d, n_points_pde, n_points_bc, n_points_ic, device
    )
    X_interior.requires_grad = True

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,           # inner CG iterations per step
        max_eval=25,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn='strong_wolfe'
    )

    losses = []
    l2_errs = []
    step_counter = [0]  # mutable counter accessible inside closure

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde = pde_loss(model, X_interior, pde_residual, compute_laplace=compute_laplace)
        loss_bc  = boundary_condition_loss(model, X_boundary, bc_residual)
        loss_ic  = initial_condition_loss(model, X_initial, ic_residual)
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        loss.backward()
        return loss

    print(f"\n{'='*60}")
    print(f"Starting L-BFGS fine-tuning ({n_steps} steps)")
    print(f"{'='*60}\n")

    l2_err = 1.0 + l2_stop_crit # init with some val
    for si in range(n_steps):
        loss = optimizer_lbfgs.step(closure)
        losses.append(loss.item())
        step_counter[0] += 1

        if (si + 1) % n_steps_log == 0:
            X_interior_test, _, _ = sample_collocation_points(
                d, n_points_pde, n_points_bc, n_points_ic, device=device
            )
            with torch.no_grad():
                u_pred = model(X_interior_test)
                u_true = u_analytic(X_interior_test)
            l2_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
            l2_errs.append(l2_err)

            # Recompute individual losses for logging (no grad needed)
            with torch.no_grad():
                u_bc  = model(X_boundary)
                u_ic  = model(X_initial)
            X_log = X_interior.detach().requires_grad_(True)
            loss_pde_log = pde_loss(model, X_log, pde_residual, compute_laplace=compute_laplace)
            loss_bc_log  = boundary_condition_loss(model, X_boundary, bc_residual)
            loss_ic_log  = initial_condition_loss(model, X_initial, ic_residual)

            print(f'[L-BFGS] Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde_log.item():.6f}, BC: {loss_bc_log.item():.6f}, '
                  f'IC: {loss_ic_log.item():.6f}, L2: {l2_err:.6f}')

        if l2_err < l2_stop_crit:
            break

    return losses, l2_errs


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
    layers = list(map(lambda x: int(x), args.layers.split(",")))
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
        model = PINN(D, layers).to(device)
        #model = PINN_SeparableTimes(D, layers).to(device)
        #model = PINN_SepTime(D, layers).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)
    

    # PDE equation
    import pde_models
    #pde_model = pde_models.HeatEquation(d, alpha=0.01, a=2*torch.pi*torch.ones(d))
    pde_model = pde_models.TravellingGaussPacket_v2(d)
    if args.use_weak_form:
        if pde_model.has_weak_form:
            use_weak_form = True
            pde_residual = pde_model.pde_residual_weak_form
        else:
            raise Exception("!! ISSUE: '--use_weak_form' argument was passed, but the PDE model does not have a weak form defined.")
    else:
        use_weak_form = False
        pde_residual = pde_model.pde_residual

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
        loss_weighting = ConstantWeights(weights=[args.lambda_pde, args.lambda_bc, args.lambda_ic])
        profiler = Profiler(report_filename=f"{dir_name}/{args.profiler_report_filename}.txt", start_step=100, end_step=110) if args.enable_profiler else None

        trainer = PINN_Trainer(model, optimizer, scheduler, pde_model, loss_weighting, profiler, device)
        losses_adam, l2_errs_adam = trainer.train_adam_fullbatch(
            n_steps=args.n_steps,
            n_steps_decay=args.n_steps_decay,
            n_calloc_points=args.n_calloc_points,
            n_test_calloc_points=args.n_test_calloc_points,
            resampling_frequency=args.resampling_frequency,
            testing_frequency=args.testing_frequency
        )
        losses += losses_adam
        l2_errs += l2_errs_adam
        print("\nAdam training complete!")
        torch.save(model, f'{dir_name}/model_adam.pth')

    # --- Phase 2: L-BFGS fine-tuning ---
    if args.n_steps_lbfgs > 0:
        losses_lbfgs, l2_errs_lbfgs = train_pinn_lbfgs(
            model,
            pde_residual, pde_model.bc_residual, pde_model.ic_residual,
            pde_model.u_analytic,
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

    # Save the results
    torch.save(model.state_dict(), f'{dir_name}/model.pth')
    torch.save(torch.tensor(losses), f'{dir_name}/training_loss.pth')
    torch.save(torch.tensor(l2_errs), f'{dir_name}/training_l2_error.pth')
    print("\nResults saved.")