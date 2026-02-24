import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext

#python main.py --n_steps=500 --n_steps_log=100 --n_steps_lbfgs=0 --d=2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="64,64,64", type=str, help="")
parser.add_argument("--n_steps", default=10_000, type=int, help="")
parser.add_argument("--n_steps_decay", default=2000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--n_points_pde", default=2000, type=int, help="")
parser.add_argument("--n_points_bc", default=400, type=int, help="")
parser.add_argument("--n_points_ic", default=400, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--lr", default=0.001, type=float, help="")
parser.add_argument("--n_steps_log", default=500, type=int, help="")
# L-BFGS
parser.add_argument("--n_steps_log_lbfgs", default=100, type=int, help="")
parser.add_argument("--n_steps_lbfgs", default=500, type=int, help="Number of L-BFGS steps after Adam. Set to 0 to skip.")
parser.add_argument("--lr_lbfgs", default=1.0, type=float, help="Learning rate for L-BFGS (typically 1.0).")
# smart Defaults
parser.add_argument("--l2_stop_crit", default=0.01, type=float, help="")
parser.add_argument("--l2_stop_crit_lbfgs", default=0.001, type=float, help="")
# 
parser.add_argument("--output_dir_name", default="run_latest", type=str, help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
parser.add_argument("--use_weak_form", action="store_true", help="")


class PINN(nn.Module):
    def __init__(self, D, layers=[64]):
        """
        D: input dimension (d spatial dims + 1 time dim)
        """
        super().__init__()

        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(nn.Tanh())

        self.net = nn.Sequential(
            nn.Linear(D, layers[0]), nn.Tanh(),
            *net_layers,
            nn.Linear(layers[-1], 1)
        )

    def forward(self, x):
        with record_function("forward"):
            return self.net(x)

    
def compute_derivatives(model, X, compute_laplace=True):
    """
    Compute u, grad u, and laplace u
    X: (batch_size, D) where D = d + 1 (spatial dims + time)
    """
    u = model(X)
    bs, D = X.shape


    with record_function("grad_u"):
        # Gradient - spatial & temporatal
        grad_u = torch.autograd.grad(
            inputs=X,
            outputs=u,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

    if compute_laplace: 
        with record_function("laplace_u"):
            # Laplacian - spatial only
            spatial_laplace_u = torch.zeros_like(u)
            for i in range(D-1):
                hess_row = torch.autograd.grad(
                    inputs=X,
                    outputs=grad_u[:,i].sum(),
                    grad_outputs=torch.tensor(1.0),
                    create_graph=True,
                    retain_graph=True
                )[0]
                spatial_laplace_u += hess_row[:,i:i+1]
    else:
        spatial_laplace_u = None

    # shapes: bs x 1, bs x D, bs x 1
    return u, grad_u, spatial_laplace_u


def sample_hypercube_boundary(num_samples, d, device='cpu'):
    """
    Boundary sampling for d-dimensional hypercube [0,1]^d
    Parameters:
    - num_samples: number of points to sample
    - d: num of spatial dimensions
    - device: 'cuda' or 'cpu'
    Returns:
    - tensor of shape (num_samples, d)
    """
    # Sample all coordinates uniformly from [0,1]
    samples = torch.rand(num_samples, d, device=device, requires_grad=False)
    
    # Choose which dimension to fix for each sample
    fixed_dims = torch.randint(0, d, (num_samples,), device=device)
    
    # Choose whether to fix to 0 or 1 for each sample
    fixed_values = torch.randint(0, 2, (num_samples,), device=device).float()
    
    # Set the fixed dimension to 0 or 1
    samples[torch.arange(num_samples, device=device), fixed_dims] = fixed_values
    
    return samples


def pde_loss(model, X_in, pde_residual, compute_laplace=True):
    """
    X_in: (batch_size, d+1) tensor
    """
    X_in.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X_in, compute_laplace=compute_laplace)
    residual = pde_residual(X_in, u, grad_u, spatial_laplace_u)
    return torch.mean(residual**2)

def initial_condition_loss(model, X_ic, ic_residual):
    """
    X_ic: (batch_size, d+1) tensor with t = 0
    IC: u(x,0) = u_IC(x)
    """
    u = model(X_ic)
    residual = ic_residual(X_ic, u)
    return torch.mean(residual**2)

def boundary_condition_loss(model, X_bc, bc_residual):
    """
    X_bc: (batch_size, d+1) tensor with points on boundary
    BC: u(x,t) = u_BC(x,t)
    """
    u = model(X_bc)
    residual = bc_residual(X_bc, u)
    return torch.mean(residual**2)


def sample_collocation_points(
        d,
        n_interior, n_boundary, n_initial, 
        device='cpu'
    ):
    """
    Generate collocation points for training
    Parameters:
    - d: spatial dimensions
    - device: 'cuda' or 'cpu'
    """
    # Interior points (for PDE): [x1, ..., xd, t]
    X_interior = torch.rand(n_interior, d+1, device=device)
    
    # Boundary points: spatial coords on boundary, t random in [0,1]
    x_boundary = sample_hypercube_boundary(n_boundary, d, device=device)
    t_boundary = torch.rand(n_boundary, 1, device=device)
    X_boundary = torch.cat([x_boundary, t_boundary], dim=1)
    
    # Initial condition points: spatial coords random in [0,1]^d, t=0
    x_initial = torch.rand(n_initial, d, device=device)
    t_initial = torch.zeros(n_initial, 1, device=device)
    X_initial = torch.cat([x_initial, t_initial], dim=1)
    
    return X_interior, X_boundary, X_initial


def train_pinn(
        model,
        optimizer, scheduler,
        pde_residual, bc_residual, ic_residual,
        u_analytic,
        d,
        n_steps=10_000, 
        n_steps_decay=2000,
        n_steps_log=500,
        n_points_pde=2000, n_points_bc=400, n_points_ic=400,
        lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
        l2_stop_crit=0.01,
        profiler_report_filename='profiler_report',
        output_dir_name='run_latest',
        compute_laplace=True,
        device='cpu'
    ):
    """Train the PINN model"""
    
    losses = []
    l2_errs = []
    
    #profile_start = 1067
    #profile_end = profile_start + 100

    # Generate training data
    X_interior, X_boundary, X_initial = sample_collocation_points(
        d, n_points_pde, n_points_bc, n_points_ic, device
    )


    enable_profiler=True
    profile_step_start=100
    profile_n_steps=10
    profile_step_end = profile_step_start + profile_n_steps

    def make_profiler():
        if enable_profiler:
            return profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
        else:
            return nullcontext()

    prof_ctx = make_profiler()

    
    l2_err = 1.0 + l2_stop_crit # init with some val
    for si in range(n_steps):

        #if (si + 1) % 2000 == 0:
        ### Generate training data
        #    print("New training data arrived!")
        #    X_interior, X_boundary, X_initial = sample_collocation_points(
        #        d, n_points_pde, n_points_bc, n_points_ic, device
        #    )
    

        ## Turn on profiling
        #if si == profile_start:
        #    prof = profile(
        #        activities=[ProfilerActivity.CPU],
        #        profile_memory=True,
        #        record_shapes=True
        #    )
        #    prof.__enter__()

        if enable_profiler and si == profile_step_start:
            prof_ctx.__enter__()
            print(f"\n[Profiler] Started at step {si+1}")

        optimizer.zero_grad()

        # Sample the training points

        # Compute individual losses
        with record_function("loss"):
            loss_pde = pde_loss(model, X_interior, pde_residual, compute_laplace=compute_laplace)
            loss_bc = boundary_condition_loss(model, X_boundary, bc_residual)
            loss_ic = initial_condition_loss(model, X_initial, ic_residual)

        # Total loss with weights
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic

        # Backward pass
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()

        # Step scheduler
        if (si + 1) % n_steps_decay == 0:
            scheduler.step()

        losses.append(loss.item())

        #if si == profile_end:
        #    prof.__exit__(None, None, None)
        #    print("\n=== Profiling Results (100 steps averaged) ===")
        #    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        # Exit profiler context after the last profiled step
        if enable_profiler and si == profile_step_end - 1:
            prof_ctx.__exit__(None, None, None)
            print(f"\n[Profiler] Stopped at step {si+1}. Results ({profile_n_steps} steps):")
            prof_report = prof_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            print(prof_report)
            # save profiler report
            #prof_ctx.export_chrome_trace(f"run_latest/{profiler_report_filename}.json")
            with open(f"{output_dir_name}/{profiler_report_filename}.txt", "w") as f:
                f.write(prof_report)

        # Print progress
        if (si + 1) % n_steps_log == 0:
            X_interior_test, X_boundary_test, X_initial_test = sample_collocation_points(
                d, n_points_pde, n_points_bc, n_points_ic, device=device
            )
            u_pred = model(X_interior_test)
            u_true = u_analytic(X_interior_test)
            l2_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
            l2_errs.append(l2_err)
            print(f'Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde.item():.6f}, '
                  f'BC: {loss_bc.item():.6f}, '
                  f'IC: {loss_ic.item():.6f}, '
                  f'lr: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'L2: {l2_err:.6f}'
            )
        if l2_err < l2_stop_crit:
            break
    
    return losses, l2_errs


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
    if args.output_dir_name == 'run_latest':
        dir_name = args.output_dir_name
    else:
        dir_name = f'run_history/{args.output_dir_name}'
    os.makedirs(dir_name, exist_ok=True)    
    
    # Initialize model
    model = PINN(D, layers).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Option 1: ExponentialLR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # PDE equation
    #a = 4*torch.pi * (1.67 * torch.ones(d))
    #a = torch.tensor([11.0997, 7.5390, 9.535, 11.432, 8.1, 9.6, 7.4])[:d]
    a = 2 * torch.pi * torch.ones(d)
    print(a)
    import pde_models
    pde_model = pde_models.HeatEquation(d, alpha=0.01, a=a)
    if args.use_weak_form:
        if pde_model.has_weak_form:
            use_weak_form = True
            pde_residual = pde_model.pde_residual_weak_form
        else:
            raise Exception("!! ISSUE: '--use_weak_form' argument was passed, but the PDE model does not have a weak form defined.")
    else:
        use_weak_form = False
        pde_residual = pde_model.pde_residual

    
    # Train the model
    losses, l2_errs = train_pinn(
        model, optimizer, scheduler,
        pde_residual, pde_model.bc_residual, pde_model.ic_residual,
        pde_model.u_analytic,
        d,
        n_steps=args.n_steps,
        n_steps_decay=args.n_steps_decay,
        n_steps_log=args.n_steps_log,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ic=args.lambda_ic,
        n_points_pde=args.n_points_pde, n_points_bc=args.n_points_bc, n_points_ic=args.n_points_ic,
        l2_stop_crit=args.l2_stop_crit,
        profiler_report_filename=args.profiler_report_filename,
        output_dir_name=dir_name,
        compute_laplace=not use_weak_form,
        device=device
    )
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
        losses    += losses_lbfgs
        l2_errs   += l2_errs_lbfgs
        print("\nL-BFGS fine-tuning complete!")
        torch.save(model, f'{dir_name}/model_adam_lbfgs.pth')
    print("\nTraining complete!")
    
    import json
    with open(f'{dir_name}/args.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    pde_model.dump_pde_params(f'{dir_name}/pde_params.json')

    # Save the results
    torch.save(model, f'{dir_name}/model.pth')
    torch.save(torch.tensor(losses), f'{dir_name}/training_loss.pth')
    torch.save(torch.tensor(l2_errs), f'{dir_name}/training_l2_error.pth')
    print("\nResults saved.")