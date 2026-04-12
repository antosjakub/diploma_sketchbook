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
    fixed_values = torch.randint(0, 2, (num_samples,), device=device)
    
    # Set the fixed dimension to 0 or 1
    samples[torch.arange(num_samples, device=device), fixed_dims] = fixed_values.float()
    
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
        #sample_collocation_points(d, n_candidates, 0,0, sampling_strategy=sampling_strategy, )
        X_cand = sample_domain(n_candidates, d+1, sampling_strategy=sampling_strategy, device=device)
        X_cand = X_cand.requires_grad_(True)
    elif type == 'bc':
        X_cand = sample_bc(n_candidates, d, sampling_strategy=sampling_strategy, device=device)
    elif type == 'ic':
        X_cand = sample_ic(n_candidates, d, sampling_strategy=sampling_strategy, device=device)

    res = residual_fn(X_cand, model).detach()
    abs_res = res.abs().squeeze()
    
    if picking_criterion == "top_k":
        # Pick top-k high-residual points
        _, idx = torch.topk(abs_res, n_new)
        return X_cand[idx].detach()
    elif picking_criterion == "multinomial":
        probs = abs_res / abs_res.sum()
        idx = torch.multinomial(probs, n_new, replacement=False)
        return X_cand[idx].detach()
    else:
        raise NameError("Provide a correct picking crierion.")



# pde collac:
# - sample points once
# - push it to pde_models to store some funs
# - store those in the dataset
#
# ic collac:
# - same as in pde col
#
# bc collac
# - same as in pde col
# - now also store face / edge indices for n normal


#n_atoms=3, dof_per_atom=2, r_min=0.1
def filter_close_atoms(
    X: torch.Tensor,
    n_atoms: int,
    dof_per_atom: int,
    r_min: float,
) -> torch.Tensor:
    """
    Remove configurations where any pair of atoms is closer than r_min.

    Assumes the first n_atoms*dof_per_atom columns of X encode atom positions:
      [x0_0, ..., x0_{dof_per_atom-1},  x1_0, ...,  x_{n-1}_{dof_per_atom-1},  (optional extra cols, e.g. time)]

    Parameters:
    - X:       (n_samples, >= n_atoms*dof_per_atom)
    - n_atoms: number of atoms
    - dof_per_atom:       spatial dimensions per atom
    - r_min:   minimum allowed distance between any pair of atoms

    Returns:
    - X filtered to rows where all pairwise interatomic distances >= r_min
    """
    d = n_atoms * dof_per_atom
    positions = X[:, :d].view(-1, n_atoms, dof_per_atom)               # (N, n_atoms, dof_per_atom)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)   # (N, n_atoms, n_atoms, dof_per_atom)
    dists = diff.norm(dim=-1)                                # (N, n_atoms, n_atoms)
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=X.device)
    pair_dists = dists[:, i, j]                              # (N, n_pairs)
    mask = (pair_dists >= r_min).all(dim=1)
    return X[mask]


class CollocationDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, precomputed: dict[str, torch.Tensor]) -> None:
        self.X = X
        self.precomputed = precomputed
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #return {"X": self.X[idx]} | {k: v[idx] for k, v in self.precomputed.items()}
        return (self.X[idx], {k: v[idx] for k, v in self.precomputed.items()})



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
    n_boundary =  bs_bc * n_cycles
    n_initial  =  bs_ic * n_cycles

    if use_rbas:
        X_pde = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.pde_residual, model, type='pde', n_new=2*n_interior//3, n_candidates=4*n_interior, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.pde_residual, model, type='pde', n_new=n_interior//3, n_candidates=2*n_interior, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0)
        X_bc = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.bc_residual, model, type='bc', n_new=2*n_boundary//3, n_candidates=4*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.bc_residual, model, type='bc', n_new=n_boundary//3, n_candidates=2*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0)
        X_ic = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=2*n_initial//3, n_candidates=4*n_initial, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=n_initial//3, n_candidates=2*n_initial, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0)
    else:
        X_pde, X_bc, X_ic = sample_collocation_points(d, n_interior, n_boundary, n_initial, sampling_strategy=sampling_strategy, device=device)
    X_pde[:,:-1] = 4.0 * X_pde[:,:-1] - 2.0
    X_bc[:,:-1] = 4.0 * X_bc[:,:-1] - 2.0
    X_ic[:,:-1] = 4.0 * X_ic[:,:-1] - 2.0
    X_pde[:,-1:] *= 1.5
    X_bc[:,-1:] *= 1.5
        
    # dict containing precomputed 
    # precomputed = {"pde": {"V_grad": tensor, "V_laplace": tensor}, "ic": {"analytic": tensor}, "bc": {"V_grad": tensor}}
    precomputed = pde_model.precompute(X_pde, X_bc, X_ic)

    loader_pde = DataLoader(CollocationDataset(X_pde, precomputed["pde"]), batch_size=bs_pde, shuffle=True)
    loader_bc  = DataLoader(CollocationDataset(X_bc, precomputed["bc"]), batch_size=bs_bc, shuffle=True)
    loader_ic  = DataLoader(CollocationDataset(X_ic, precomputed["ic"]), batch_size=bs_ic, shuffle=True)
    
    return loader_pde, loader_bc, loader_ic


from torch.utils.data import DataLoader
def create_dataloader_ic(d, n_calloc, bs, model, pde_model, use_rbas=False, sampling_strategy="lhs", device="cpu"):
    # bs = 1024...
    if use_rbas:
        X_ic = torch.cat([
            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=2*n_calloc//3, n_candidates=4*n_calloc, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=n_calloc//3, n_candidates=2*n_calloc, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
        ], dim=0)
    else:
        _, _, X_ic = sample_collocation_points(d, n_interior=0, n_boundary=0, n_initial=n_calloc, sampling_strategy=sampling_strategy, device=device)
        
    precomputed_ic = {
        "p": pde_model.p_ic(X_ic[:,:-1])
    }

    loader_ic  = DataLoader(CollocationDataset(X_ic, precomputed_ic), batch_size=bs, shuffle=True)
    
    return loader_ic


if __name__ == "__main__":
    dof_per_atom = 3
    n_atoms = 7
    d = n_atoms * dof_per_atom
    r_min = 0.1
    X = torch.rand(10000, d+1)
    X[:, -1] = 0.0
    X_filtered = filter_close_atoms(X, n_atoms, dof_per_atom, r_min)
    print(X.shape)
    print(X_filtered.shape)