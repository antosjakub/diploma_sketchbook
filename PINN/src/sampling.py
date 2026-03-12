import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext


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
    if n_interior > 0:
        #X_interior = sample_uniform(n_interior, d+1, device=device)
        X_interior = sample_lhs(n_interior, d+1, device=device)
    else:
        X_interior = None
    
    if n_boundary > 0:
        # Boundary points: spatial coords on boundary, t random in [0,1]
        x_boundary = sample_hypercube_boundary(n_boundary, d, device=device)
        t_boundary = torch.rand(n_boundary, 1, device=device)
        X_boundary = torch.cat([x_boundary, t_boundary], dim=1)
    else:
        X_boundary = None
    
    if n_initial > 0:
        # Initial condition points: spatial coords random in [0,1]^d, t=0
        x_initial = torch.rand(n_initial, d, device=device)
        t_initial = torch.zeros(n_initial, 1, device=device)
        X_initial = torch.cat([x_initial, t_initial], dim=1)
    else:
        X_initial = None
    
    return X_interior, X_boundary, X_initial



from torch.utils.data import TensorDataset, DataLoader
def create_dataloaders(d, num_colloc, bs, u_bc_fun, u_ic_fun):
    # bs = 1024...
    n_cycles = num_colloc // bs
    #num_colloc = bs * n_cycles
    bs_segment_size = bs // 16
    bs_pde = bs_segment_size * 14
    bs_bc  = bs_segment_size
    bs_ic  = bs_segment_size
    num_pde = bs_pde * n_cycles
    num_bc  =  bs_bc * n_cycles
    num_ic  =  bs_ic * n_cycles

    X_pde, X_bc, X_ic = sample_collocation_points(d, num_pde, num_bc, num_ic)
    
    loader_pde =      DataLoader(TensorDataset(X_pde), batch_size=bs_pde, shuffle=True)
    loader_bc =       DataLoader(TensorDataset(X_bc, u_bc_fun(X_bc)), batch_size=bs_bc, shuffle=True)
    loader_ic =       DataLoader(TensorDataset(X_ic, u_ic_fun(X_ic[:,:-1])), batch_size=bs_ic, shuffle=True)
    
    return loader_pde, loader_bc, loader_ic