import torch
import derivatives


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

def sample_domain(n_samples: int, d: int, sampling_strategy="lhs", device="cpu") -> torch.Tensor:
    if sampling_strategy == "lhs":
        return sample_lhs(n_samples, d, device=device)
    else:
        return sample_uniform(n_samples, d, device=device)

def sample_hypercube_boundary(num_samples, d, sampling_strategy="lhs", device='cpu'):
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
    samples = sample_domain(num_samples, d, sampling_strategy=sampling_strategy, device=device)
    
    # Choose which dimension to fix for each sample
    fixed_dims = torch.randint(0, d, (num_samples,), device=device)
    
    # Choose whether to fix to 0 or 1 for each sample
    fixed_values = torch.randint(0, 2, (num_samples,), device=device).float()
    
    # Set the fixed dimension to 0 or 1
    samples[torch.arange(num_samples, device=device), fixed_dims] = fixed_values
    
    return samples


def sample_bc(n_boundary: int, d: int, sampling_strategy="lhs", device="cpu") -> torch.Tensor:
    return torch.cat([
        sample_hypercube_boundary(n_boundary, d, sampling_strategy=sampling_strategy, device=device),
        torch.rand(n_boundary, 1, device=device)
    ], dim=1).float()

def sample_ic(n_initial: int, d: int, sampling_strategy="lhs", device="cpu") -> torch.Tensor:
    return torch.cat([
        sample_domain(n_initial, d, sampling_strategy=sampling_strategy, device=device),
        torch.zeros(n_initial, 1, device=device)
    ], dim=1).float()


def sample_collocation_points(
        d,
        n_interior, n_boundary, n_initial, 
        sampling_strategy="lhs",
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
        X_interior = sample_domain(n_interior, d+1, sampling_strategy=sampling_strategy, device=device)
    else:
        X_interior = None
    
    if n_boundary > 0:
        # Boundary points: spatial coords on boundary, t random in [0,1]
        X_boundary = sample_bc(n_boundary, d, sampling_strategy=sampling_strategy, device=device)
    else:
        X_boundary = None
    
    if n_initial > 0:
        # Initial condition points: spatial coords random in [0,1]^d, t=0
        X_initial = sample_ic(n_initial, d, sampling_strategy=sampling_strategy, device=device)
    else:
        X_initial = None
    
    return X_interior, X_boundary, X_initial


#@torch.no_grad()
#with torch.enable_grad():
def residual_based_adaptive_sampling(d, residual_fn, model, type="pde", n_new=1000, n_candidates=50_000, sampling_strategy="lhs", picking_criterion="multinomial", device="cpu"):
    """
    sampling_strategy: "lhs" or "uniform" 
    picking_criterion: "multinomial" or "top_k" 
    """

    if type == 'pde':
        X_cand = sample_domain(n_candidates, d+1, sampling_strategy=sampling_strategy, device=device)
        X_cand.requires_grad_(True) # needed for grad and laplace computatation
        u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X_cand)
        res = residual_fn(X_cand, u, grad_u, spatial_laplace_u).detach()
    elif type == 'bc':
        X_cand = sample_bc(n_candidates, d, sampling_strategy=sampling_strategy, device=device)
        res = residual_fn(X_cand).detach()
    elif type == 'ic':
        X_cand = sample_ic(n_candidates, d, sampling_strategy=sampling_strategy, device=device)
        res = residual_fn(X_cand).detach()

    abs_res = res.abs().squeeze()
    
    if picking_criterion == "top_k":
        # Pick top-k high-residual points
        _, idx = torch.topk(abs_res, n_new)
        return X_cand[idx].detach()
    else:
        probs = abs_res / abs_res.sum()
        idx = torch.multinomial(probs, n_new, replacement=False)
        return X_cand[idx].detach()


def resample_training_data(d, residual_fn, model, n_interior, n_boundary, n_initial, sampling_strategy="lhs", device="cpu"):
    X_int1 = residual_based_adaptive_sampling(d, residual_fn, model, n_new=2*n_interior//3, n_candidates=4*n_interior, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device)
    X_int2 = residual_based_adaptive_sampling(d, residual_fn, model, n_new=n_interior//3, n_candidates=2*n_interior, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
    X_interior = torch.cat([X_int1, X_int2], dim=0).shuffle(dim=0)
    X_boundary = sample_bc(n_boundary, d, sampling_strategy=sampling_strategy, device=device)
    X_initial = sample_ic(n_initial, d, sampling_strategy=sampling_strategy, device=device)
    return X_interior, X_boundary, X_initial


from torch.utils.data import TensorDataset, DataLoader
def create_dataloaders(d, num_colloc, bs, model, pde_model, use_rbas=False, sampling_strategy="lhs", device="cpu"):
    # bs = 1024...
    n_cycles = num_colloc // bs
    #num_colloc = bs * n_cycles
    bs_segment_size = bs // 16
    bs_pde = bs_segment_size * 14
    bs_bc  = bs_segment_size
    bs_ic  = bs_segment_size
    n_interior = bs_pde * n_cycles
    n_boundary  =  bs_bc * n_cycles
    n_initial  =  bs_ic * n_cycles

    if use_rbas:
        X_interior = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.pde_residual, model, n_new=2*n_interior//3, n_candidates=4*n_interior, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.pde_residual, model, n_new=n_interior//3, n_candidates=2*n_interior, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0).shuffle(dim=0)
        X_bc = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.u_bc_residual, model, n_new=2*n_boundary//3, n_candidates=4*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.u_bc_residual, model, n_new=n_boundary//3, n_candidates=2*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0).shuffle(dim=0)
        X_ic = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.u_ic_residual, model, n_new=2*n_initial//3, n_candidates=4*n_initial, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.u_ic_residual, model, n_new=n_initial//3, n_candidates=2*n_initial, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0).shuffle(dim=0)
    else:
        X_pde, X_bc, X_ic = sample_collocation_points(d, num_pde, num_bc, num_ic, sampling_strategy=sampling_strategy, device=device)
    
    loader_pde =      DataLoader(TensorDataset(X_pde), batch_size=bs_pde, shuffle=True)
    loader_bc =       DataLoader(TensorDataset(X_bc, pde_model.u_bc(X_bc)), batch_size=bs_bc, shuffle=True)
    loader_ic =       DataLoader(TensorDataset(X_ic, pde_model.u_ic(X_ic[:,:-1])), batch_size=bs_ic, shuffle=True)
    
    return loader_pde, loader_bc, loader_ic