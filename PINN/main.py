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
parser.add_argument("--n_points_pde", default=5_000, type=int, help="")
parser.add_argument("--n_points_bc", default=500, type=int, help="")
parser.add_argument("--n_points_ic", default=500, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--n_steps_log", default=500, type=int, help="")
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


class PINN(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()

        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(activation_fn())

        self.net = nn.Sequential(
            nn.Linear(D, layers[0]), activation_fn(),
            *net_layers,
            nn.Linear(layers[-1], 1)
        )

    def forward(self, x):
        with record_function("forward"):
            return self.net(x)

class PINN_SepTime(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()

        # spatial termporal
        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(activation_fn())
        self.net_sptemp = nn.Sequential(
            nn.Linear(D, layers[0]), activation_fn(),
            *net_layers,
            nn.Linear(layers[-1], 1)
        )
        # temporal
        self.net_temp = nn.Sequential(
            nn.Linear(1, 128), activation_fn(),
            nn.Linear(128, 64), activation_fn(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        with record_function("forward"):
            return self.net_sptemp(X) * self.net_temp(X[:,-1:])

class PINN_SeparableTimes(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()
        self.d = D-1

        self.nets_1dspatial_temporal = []
        for di in range(self.d):
            net_layers = []
            for l1,l2 in zip(layers[:-1], layers[1:]):
                net_layers.append(nn.Linear(l1,l2))
                net_layers.append(activation_fn())
            self.nets_1dspatial_temporal.append( nn.Sequential(
                nn.Linear(2, layers[0]), activation_fn(),
                *net_layers,
                nn.Linear(layers[-1], 1)
            ) )
        self.net_temporal = nn.Sequential(
            nn.Linear(1, 128), activation_fn(),
            nn.Linear(128, 64), activation_fn(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        with record_function("forward"):
            out = torch.ones((X.shape[0],1))
            for di in range(self.d):
                out *= self.nets_1dspatial_temporal[di](torch.cat([X[:,di:di+1],X[:,-1:]],dim=1))
            out *= self.net_temporal(X[:,-1:])
            return out


# Usage example (d=15 case)
# input_dim = 15 + 1
# ff = FourierFeatures(input_dim, num_freqs=320, sigma=30.0)
# or multi-scale:
# ff = FourierFeatures(input_dim, num_freqs=300, scale_multiples=[1, 10, 50])
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_freqs=256, sigma=20.0, scale_multiples=None):
        super().__init__()
        self.num_freqs = num_freqs
        self.sigma = sigma

        # Optional: multi-scale
        self.multi_scale = scale_multiples is not None
        if self.multi_scale:
            self.scales = scale_multiples
            self.Bs = nn.ParameterList()
            for s in scale_multiples:
                Bi = torch.randn(num_freqs//len(scale_multiples), input_dim) * s
                self.Bs.append(nn.Parameter(Bi, requires_grad=False))  # still fixed
        else:
            # === Sample B once and freeze it ===
            B = torch.randn(num_freqs, input_dim) * sigma   # Normal(0, sigma²)
            self.register_buffer('B', B)                    # fixed, not in optimizer

    def forward(self, z):  # z: (batch, d+1)
        if self.multi_scale:
            feats = []
            for i, Bi in enumerate(self.Bs):
                x = 2 * torch.pi * (z @ Bi.T)
                feats.append(torch.cat([torch.cos(x), torch.sin(x)], dim=-1))
            return torch.cat(feats, dim=-1)
        else:
            x = 2 * torch.pi * (z @ self.B.T)          # (batch, M)
            return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class ResNetBlock(nn.Module):
    def __init__(self, width, activation=nn.Mish):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, width)
        )
        self.act = activation()

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.act(out + residual)   # residual connection + activation


class ResPINN(nn.Module):
    def __init__(self, d, num_freqs=256, sigma=30.0, hidden_width=256, num_blocks=5):
        super().__init__()
        input_dim = d + 1
        
        self.ff = FourierFeatures(input_dim, num_freqs=num_freqs, sigma=sigma)
        ff_out_dim = 2 * num_freqs
        
        # First layer: map FF output → hidden_width
        self.first = nn.Linear(ff_out_dim, hidden_width)
        
        # Stack residual blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_width) for _ in range(num_blocks)
        ])
        
        # Final layer
        self.out = nn.Linear(hidden_width, 1)
        
    def forward(self, x, t):
        z = torch.cat([x, t], dim=-1) # (batch, d+1)
        gamma = self.ff(z)            # (batch, 2M)
        h = self.first(gamma)
        h = nn.Mish()(h)              # initial activation
        
        for block in self.blocks:
            h = block(h)
        
        return self.out(h)
#d=5:  num_freqs=128, sigma=8, hidden_width=128, num_blocks=4
#d=10: num_freqs=256, sigma=15, hidden_width=256, num_blocks=5
#d=15: num_freqs=320, sigma=30, hidden_width=256, num_blocks=6
#d=20: num_freqs=512, sigma=40, hidden_width=256, num_blocks=8


def compute_grad_norm(loss, model):
    """L2 norm of gradients of `loss` w.r.t. model parameters."""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, allow_unused=True).detach()
    total = sum(g.norm() ** 2 for g in grads if g is not None)
    return total.sqrt()

class AdaptiveWeights:
    def __init__(self, n_terms, momentum=0.9, device='cpu'):
        self.w = torch.ones(n_terms, device=device)
        self.momentum = momentum

    def update(self, grad_norms):
        """grad_norms: list of grad_\theta(loss) terms"""
        # w_i = w_i * mean(losses) / val_loss_i
        norms = torch.stack([g.detach() for g in grad_norms])
        mean_norm = norms.mean()
        target = mean_norm / (norms + 1e-8)
        # exponential moving average
        self.w = self.momentum * self.w + (1 - self.momentum) * target
        return self.w


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
            spatial_laplace_u = []
            for i in range(D-1):
                hess_row = torch.autograd.grad(
                    inputs=X,
                    outputs=grad_u[:,i].sum(),
                    grad_outputs=torch.tensor(1.0),
                    create_graph=True,
                    retain_graph=True
                )[0]
                spatial_laplace_u.append(hess_row[:,i:i+1])
        spatial_laplace_u = torch.cat(spatial_laplace_u, dim=1)
    else:
        spatial_laplace_u = None

    # shapes: bs x 1, bs x D, bs x D-1
    return u, grad_u, spatial_laplace_u


#@torch.no_grad()
#with torch.enable_grad():
def residual_based_adaptive_sampling(d, residual_fn, n_new=1000, n_candidates=50_000, sampling_strategy="latin", picking_criterion="multinomial"):
    """
    sampling_strategy: "latin" or "uniform" 
    picking_criterion: "multinomial" or "top_k" 
    """
    if sampling_strategy == "uniform":
        X_cand = torch.rand(n_candidates, d+1, device=device)
    else:
        X_cand = sample_lhs(n_candidates, d+1)

    X_cand.requires_grad_(True) # needed for grad and laplace computatation
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X_cand)
    res = residual_fn(X_cand, u, grad_u, spatial_laplace_u).detach()
    abs_res = res.abs().squeeze()
    
    if picking_criterion == "top_k":
        # Pick top-k high-residual points
        _, idx = torch.topk(abs_res, n_new)
        return X_cand[idx].detach()
    else:
        probs = abs_res / abs_res.sum()
        idx = torch.multinomial(probs, n_new, replacement=False)
        return X_cand[idx].detach()

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


def sample_uniform(n_samples: int, n_dims: int, device="cpu") -> torch.Tensor:
    return torch.rand(n_samples, n_dims, device=device)

def sample_lhs(n_samples: int, n_dims: int, device="cpu") -> torch.Tensor:
    """Returns LHS in [0, 1]^n_dims."""
    # Create stratified intervals, then permute each dimension independently
    perms = torch.stack([torch.randperm(n_samples) for _ in range(n_dims)], dim=1)
    # Sample uniformly within each stratum
    uni = torch.rand(n_samples, n_dims, device=device)
    # shape: (n_samples, n_dims)
    return (perms.float() + uni) / n_samples


def compute_u_grad_u(model, X):
    u, vjp_fn = torch.func.vjp(model, X)
    grad_u = vjp_fn(torch.ones_like(u))
    return u, grad_u[0]

# entries contain either -1 or 1
sample_Rademacherlambda = lambda n1,n2: 2.0*torch.randint(0,2,(n1,n2),dtype=torch.float)-1.0,

def hutchinson_trace_estimation(model, X):

    def value_point(point):
        return model(point.unsqueeze(dim=0)).squeeze()

    from torch.func import jacrev, jvp
    def laplace_hutchinson_point(point):
        num_vectors = 100
        vectors = sample_Rademacherlambda(num_vectors, len(point))
        grad = lambda point: jacrev(value_point)(point)
        jvp_f = lambda v: torch.dot(v, jvp(grad, (point,), (v,))[1])
        return torch.sum(torch.vmap(jvp_f)(vectors))/num_vectors

    def laplace_hutchinson(points):
        return torch.vmap(laplace_hutchinson_point, randomness="same")(points)

    # N x 1
    return laplace_hutchinson(X).unsqueeze(dim=1)

def compute_derivatives_hte(model, X):
    u, grad_u = compute_u_grad_u(model, X)
    spatial_laplace_u = hutchinson_trace_estimation(model, X)
    return u, grad_u, spatial_laplace_u


def sdgd_loss(model, X, pde_residual, pde_sgsd_single_term_residual, num_dims_to_use: int):
    # sample some indices
    bs,D = X.shape
    d = D-1
    I = torch.randperm(d)[:num_dims_to_use]
    X.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X)
    R = pde_loss(model, X, pde_residual, compute_laplace=True).detach()
    R_stoch = torch.zeros((bs,1))
    for i in I:
        #Ri = 1/d * grad_u[:,-1:] - alpha * spatial_laplace_u[i] + v[i] * grad_u[:,i:i+1] + 1/d * b * u
        Ri = pde_sgsd_single_term_residual(X, u, grad_u, spatial_laplace_u, i)
        R_stoch += Ri
    # total loss
    loss = 2 * R * R_stoch
    loss = torch.mean(loss)
    # scalar
    return loss

def pde_loss(model, X_in, pde_residual, compute_laplace=True):
    """
    X_in: (batch_size, d+1) tensor
    """
    #X_in.requires_grad = True
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
    #X_interior = sample_uniform(n_interior, d+1, device=device)
    X_interior = sample_lhs(n_interior, d+1, device=device)
    
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

        if (si + 1) % 2000 == 0:
        ## Generate training data
            print("New training data arrived!")
            X_interior, X_boundary, X_initial = sample_collocation_points(
                d, n_points_pde, n_points_bc, n_points_ic, device
            )
            #X_int1 = residual_based_adaptive_sampling(d, pde_residual, n_new=2*n_points_pde//3, n_candidates=5*n_points_pde, sampling_strategy="latin", picking_criterion="multinomial")
            #X_int2 = residual_based_adaptive_sampling(d, pde_residual, n_new=n_points_pde//3, n_candidates=5*n_points_pde, sampling_strategy="latin", picking_criterion="top_k")
            #X_interior = torch.cat([X_int1, X_int2], dim=0)
            #_, X_boundary, X_initial = sample_collocation_points(
            #    d, 10, n_points_bc, n_points_ic, device
            #)
    

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
            #loss_pde = sdgd_loss(model, X_interior, pde_residual, None, 3)
            loss_pde = pde_loss(model, X_interior.detach().requires_grad_(True), pde_residual, compute_laplace=compute_laplace)
            loss_bc = boundary_condition_loss(model, X_boundary.detach(), bc_residual)
            loss_ic = initial_condition_loss(model, X_initial.detach(), ic_residual)

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
            l1_err = torch.mean((u_pred - u_true).abs()).item()
            rel_err = torch.max( ((u_pred - u_true-1e-7)/(u_true+1e-7)).abs() ).item()
            l2_errs.append(l2_err)
            print(f'Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde.item():.6f}, '
                  f'BC: {loss_bc.item():.6f}, '
                  f'IC: {loss_ic.item():.6f}, '
                  f'lr: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'L2: {l2_err:.6f}, '
                  f'L1: {l1_err:.6f}, '
                  f'rel_max: {rel_err:.6f}'
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
    dir_name = args.output_dir_name
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    os.makedirs(dir_name, exist_ok=True)    
    
    # Initialize model
    if args.starting_model:
        model = torch.load(args.starting_model, weights_only=False)
    else:
        #model = PINN(D, layers).to(device)
        #model = PINN_SeparableTimes(D, layers).to(device)
        model = PINN_SepTime(D, layers).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Option 1: ExponentialLR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

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
        losses_adam, l2_errs_adam = train_pinn(
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
    with open(f'{dir_name}/args.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    pde_model.dump_pde_params(f'{dir_name}/pde_params.json')

    # Save the results
    torch.save(model, f'{dir_name}/model.pth')
    torch.save(torch.tensor(losses), f'{dir_name}/training_loss.pth')
    torch.save(torch.tensor(l2_errs), f'{dir_name}/training_l2_error.pth')
    print("\nResults saved.")