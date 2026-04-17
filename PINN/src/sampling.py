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
    - samples: tensor of shape (num_samples, d)
    - normals: outward unit normals, tensor of shape (num_samples, d)
    """
    # Sample all coordinates uniformly from [0,1]
    samples = sample_domain(num_samples, d, sampling_strategy=sampling_strategy, device=device)

    # Choose which dimension to fix for each sample
    fixed_dims = torch.randint(0, d, (num_samples,), device=device)

    # Choose whether to fix to 0 or 1 for each sample
    fixed_values = torch.randint(0, 2, (num_samples,), device=device)

    # Set the fixed dimension to 0 or 1
    samples[torch.arange(num_samples, device=device), fixed_dims] = fixed_values.float()

    # Outward normals: face at 0 has normal -1, face at 1 has normal +1
    normals = torch.zeros(num_samples, d, device=device)
    normals[torch.arange(num_samples, device=device), fixed_dims] = 2.0 * fixed_values.float() - 1.0

    return samples, normals


def sample_bc(n_boundary: int, d: int, sampling_strategy="lhs", device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    spatial, normals = sample_hypercube_boundary(n_boundary, d, sampling_strategy=sampling_strategy, device=device)
    X_bc = torch.cat([spatial, torch.rand(n_boundary, 1, device=device)], dim=1).float()
    return X_bc, normals

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
        X_boundary, normals_bc = sample_bc(n_boundary, d, sampling_strategy=sampling_strategy, device=device)
    else:
        X_boundary = None
        normals_bc = None

    if n_initial > 0:
        # Initial condition points: spatial coords random in [0,1]^d, t=0
        X_initial = sample_ic(n_initial, d, sampling_strategy=sampling_strategy, device=device)
    else:
        X_initial = None

    return X_interior, X_boundary, X_initial, normals_bc


#@torch.no_grad()
#with torch.enable_grad():
def residual_based_adaptive_sampling(d, residual_fn, model, type="pde", n_new=1000, n_candidates=50_000, sampling_strategy="lhs", picking_criterion="multinomial", device="cpu"):
    """
    sampling_strategy: "lhs" or "uniform"
    picking_criterion: "multinomial" or "top_k"
    Returns (X_selected,) for pde/ic, or (X_selected, normals_selected) for bc.
    """
    normals_cand = None
    precomputed = {}
    if type == 'pde':
        X_cand = sample_domain(n_candidates, d+1, sampling_strategy=sampling_strategy, device=device)
        X_cand = X_cand.requires_grad_(True)
    elif type == 'bc':
        X_cand, normals_cand = sample_bc(n_candidates, d, sampling_strategy=sampling_strategy, device=device)
        precomputed["normals"] = normals_cand
    elif type == 'ic':
        X_cand = sample_ic(n_candidates, d, sampling_strategy=sampling_strategy, device=device)

    res = residual_fn(X_cand, model, precomputed).detach()
    abs_res = res.abs().squeeze()

    if picking_criterion == "top_k":
        _, idx = torch.topk(abs_res, n_new)
    elif picking_criterion == "multinomial":
        probs = abs_res / abs_res.sum()
        idx = torch.multinomial(probs, n_new, replacement=False)
    else:
        raise NameError("Provide a correct picking crierion.")

    if normals_cand is not None:
        return X_cand[idx].detach(), normals_cand[idx].detach()
    return X_cand[idx].detach()



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


#from torch.utils.data import DataLoader
#def create_dataloader_ic(d, n_calloc, bs, model, pde_model, use_rbas=False, sampling_strategy="lhs", device="cpu"):
#    # bs = 1024...
#    if use_rbas:
#        X_ic = torch.cat([
#            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=2*n_calloc//3, n_candidates=4*n_calloc, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
#            residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=n_calloc//3, n_candidates=2*n_calloc, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
#        ], dim=0)
#    else:
#        _, _, X_ic, _ = sample_collocation_points(d, n_interior=0, n_boundary=0, n_initial=n_calloc, sampling_strategy=sampling_strategy, device=device)
#        
#    precomputed_ic = {
#        "p": pde_model.p_ic(X_ic[:,:-1])
#    }
#
#    loader_ic  = DataLoader(CollocationDataset(X_ic, precomputed_ic), batch_size=bs, shuffle=True)
#    
#    return loader_ic


import torch
from typing import Callable, Tuple, Optional, Union


def euler_maruyama_trajectory_bank(
    x0: torch.Tensor,
    mu: Union[torch.Tensor, Callable],
    sigma: Union[float, torch.Tensor, Callable],
    T: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate an SDE trajectory bank with Euler-Maruyama:
        dx = mu(x,t) dt + sigma(x,t) dW_t

    Automatically simplifies the computation based on the types of mu and sigma.

    Parameters
    ----------
    x0 : torch.Tensor
        Initial samples, shape (n_traj, d).
    mu : callable or torch.Tensor
        Drift. Either:
          - callable mu(x, t) -> (n_traj, d)
          - constant tensor of shape (d,)
    sigma : callable, torch.Tensor, float
        Diffusion. Either:
          - callable sigma(x, t) -> (n_traj, d, m)
          - constant matrix of shape (d, m)
          - scalar, interpreted as sigma * I
    T : float
        Final time.
    n_steps : int
        Number of time steps.

    Returns
    -------
    times : torch.Tensor
        Time grid, shape (n_steps + 1,).
    traj_bank : torch.Tensor
        Simulated trajectories, shape (n_traj, n_steps + 1, d).
    """
    if x0.ndim != 2:
        raise ValueError("x0 must have shape (n_traj, d).")

    device = x0.device
    dtype = x0.dtype
    n_traj, d = x0.shape
    dt = T / n_steps
    sqrt_dt = dt ** 0.5

    times = torch.linspace(0.0, T, n_steps + 1, device=device, dtype=dtype)
    x = x0.clone()

    traj_bank = torch.empty(n_traj, n_steps + 1, d, device=device, dtype=dtype)
    traj_bank[:, 0, :] = x

    # Classify mu
    mu_is_callable = callable(mu)
    if not mu_is_callable:
        if isinstance(mu, torch.Tensor) and mu.ndim == 1 and mu.shape[0] == d:
            mu_const = torch.as_tensor(mu, device=device, dtype=dtype)
        else:
            raise ValueError("mu has incorrect type.")

    # Classify sigma
    sigma_is_callable = callable(sigma)
    if not sigma_is_callable:
        if isinstance(sigma, (float,)) or (isinstance(sigma, torch.Tensor) and sigma.ndim == 0):
            sigma_mode = "scalar"
            sigma_scalar = torch.as_tensor(sigma, device=device, dtype=dtype)
        elif isinstance(sigma, torch.Tensor) and sigma.ndim == 2 and sigma.shape[0] == d and sigma.shape[1] == d:
            sigma_mode = "matrix"
            sigma_matrix = torch.as_tensor(sigma, device=device, dtype=dtype)  # (d, m)
        else:
            raise ValueError("sigma has incorrect type.")

    for n in range(n_steps):
        t_n = torch.full((n_traj, 1), times[n], device=device, dtype=dtype)

        # Drift
        drift = mu(x, t_n) if mu_is_callable else mu_const

        # Diffusion
        if sigma_is_callable:
            diffusion = sigma(x, t_n)  # (n_traj, d, m)
            m = diffusion.shape[2]
            dW = torch.randn(n_traj, m, device=device, dtype=dtype) * sqrt_dt
            diff_step = torch.einsum("ndm,nm->nd", diffusion, dW)
        elif sigma_mode == "scalar":
            dW = torch.randn(n_traj, d, device=device, dtype=dtype) * sqrt_dt
            diff_step = sigma_scalar * dW
        else:  # sigma_mode == "matrix"
            m = sigma_matrix.shape[1]
            dW = torch.randn(n_traj, m, device=device, dtype=dtype) * sqrt_dt
            diff_step = torch.einsum("dm,nm->nd", sigma_matrix, dW)

        x = x + drift * dt + diff_step
        traj_bank[:, n + 1, :] = x

    return times, traj_bank


def sample_residual_points_from_trajectory_bank(
    traj_bank: torch.Tensor,
    times: torch.Tensor,
    n_points: int = 1000,
    exclude_t0: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample residual collocation points from the trajectory bank.

    Parameters
    ----------
    traj_bank : torch.Tensor
        Shape (n_traj, n_times, d).
    times : torch.Tensor
        Shape (n_times,).
    n_points : int
        Number of residual points to sample.
    exclude_t0 : bool
        If True, do not sample from the initial slice t=0.
    generator : Optional[torch.Generator]
        PyTorch RNG generator for reproducibility.

    Returns
    -------
    x_res : torch.Tensor
        Residual points, shape (n_points, d).
        This is your requested row-wise tensor of size 1000 x d if n_points=1000.
    t_res : torch.Tensor
        Corresponding times, shape (n_points, 1).
    """
    if traj_bank.ndim != 3:
        raise ValueError("traj_bank must have shape (n_traj, n_times, d).")
    if times.ndim != 1:
        raise ValueError("times must have shape (n_times,).")
    if traj_bank.shape[1] != times.shape[0]:
        raise ValueError("traj_bank.shape[1] must equal len(times).")

    n_traj, n_times, d = traj_bank.shape
    time_start_idx = 1 if exclude_t0 else 0

    if time_start_idx >= n_times:
        raise ValueError("No valid time indices available for sampling.")

    traj_idx = torch.randint(
        low=0,
        high=n_traj,
        size=(n_points,),
        device=traj_bank.device,
        generator=generator,
    )
    time_idx = torch.randint(
        low=time_start_idx,
        high=n_times,
        size=(n_points,),
        device=traj_bank.device,
        generator=generator,
    )

    x_res = traj_bank[traj_idx, time_idx, :]          # (n_points, d)
    t_res = times[time_idx].unsqueeze(1)              # (n_points, 1)

    return x_res, t_res



def residual_based_adaptive_sampling(X_cand, residual_fn, model, n_new=1000, picking_criterion="multinomial"):
    """
    sampling_strategy: "lhs" or "uniform" 
    picking_criterion: "multinomial" or "top_k" 
    """

    X_cand = X_cand.requires_grad_(True)

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





def split_res_points(n_res_points, bs=512, f_pde=14, f_bc=1, f_ic=1, f_norm=16):
    n_cycles = n_res_points // bs
    bs_segment_size = bs // (f_pde + f_bc + f_ic + f_norm)
    bs_pde = bs_segment_size * f_pde
    bs_bc  = bs_segment_size * f_bc
    bs_ic  = bs_segment_size * f_ic
    bs_norm  = bs_segment_size * f_norm
    n_interior = bs_pde * n_cycles
    n_boundary =  bs_bc * n_cycles
    n_initial  =  bs_ic * n_cycles
    n_norm  =  bs_norm * n_cycles
    return (bs_pde, bs_bc, bs_ic, bs_norm), (n_interior, n_boundary, n_initial, n_norm)

def contruct_trajs_ic(x0, n_res_points):
    X_ic = torch.cat([
        x0[:n_res_points],
        torch.zeros(n_res_points, 1)],
    dim=1)
    return X_ic


def sample_trajs_res_points(pde_model, x0, T, nt_steps, n_res_points):
    times, traj_bank = euler_maruyama_trajectory_bank(
        x0=x0,
        mu=pde_model.mu,
        sigma=pde_model.sigma,
        T=T,
        n_steps=nt_steps,
    )
    X_pde = torch.cat(
        # Sample n residual points row-wise: shape (n, d)
        sample_residual_points_from_trajectory_bank(
            traj_bank=traj_bank,
            times=times,
            n_points=n_res_points,
            exclude_t0=True,
        ), dim=1
    )
    return X_pde

def scale_samples__spatial(X, lo, hi):
    X[:,:-1] = lo + (hi - lo) * X[:,:-1]
    return X

def scale_samples__temporal(X, T):
    X[:,-1:] *= T
    return X

def scale_samples__spatial_temporal(X, lo, hi, T):
    return scale_samples__temporal(scale_samples__spatial(X, lo, hi), T)


from torch.utils.data import DataLoader



def _precompute_active(pde_model, X_pde, X_bc, X_ic, active_losses, device):
    """Call pde_model.precompute, padding inactive args with a 1-row dummy, then
    strip keys that weren't requested."""
    d = pde_model.d
    dummy = torch.zeros(1, d + 1, device=device)
    precomputed = pde_model.precompute(
        X_pde if X_pde is not None else dummy,
        X_bc  if X_bc  is not None else dummy,
        X_ic  if X_ic  is not None else dummy,
    )
    for k in ("pde", "bc", "ic"):
        if k not in active_losses:
            precomputed.pop(k, None)
    return precomputed


def create_dataloaders__domain(
    model, pde_model, active_losses,
    n_res_points=10_000, bs=1_000, spatial_domain=None, T=1.0,
    use_rbas=False, sampling_strategy="lhs", device="cpu",
):
    (bs_pde, bs_bc, bs_ic, bs_norm), (n_interior, n_boundary, n_initial, n_norm) = split_res_points(n_res_points, bs,
        f_pde  = 14 if "pde"  in active_losses else 0,
        f_bc   =  1 if "bc"   in active_losses else 0,
        f_ic   =  1 if "ic"   in active_losses else 0,
        f_norm = 16 if "norm" in active_losses else 0
    )
    d = pde_model.d

    if "pde" not in active_losses: n_interior = 0
    if "bc"  not in active_losses: n_boundary = 0
    if "ic"  not in active_losses: n_initial  = 0
    if "norm"  not in active_losses: n_norm  = 0

    X_pde = X_bc = X_ic = X_norm = None
    normals_bc = None

    #vanilla = {"pde": {"sampling_type": "domain/trajs","bs":12}, "bc":{}, "ic":{"sampling_type":"p0/domain"}}
    #d = {}

    if use_rbas:
        if "pde" in active_losses:
            X_pde = torch.cat([
                residual_based_adaptive_sampling(d, pde_model.pde_residual, model, type='pde', n_new=2*n_interior//3, n_candidates=4*n_interior, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
                residual_based_adaptive_sampling(d, pde_model.pde_residual, model, type='pde', n_new=n_interior//3, n_candidates=2*n_interior, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device),
            ], dim=0)
        if "bc" in active_losses:
            X_bc_1, normals_bc_1 = residual_based_adaptive_sampling(d, pde_model.bc_residual, model, type='bc', n_new=2*n_boundary//3, n_candidates=4*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device)
            X_bc_2, normals_bc_2 = residual_based_adaptive_sampling(d, pde_model.bc_residual, model, type='bc', n_new=n_boundary//3, n_candidates=2*n_boundary, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device)
            X_bc = torch.cat([X_bc_1, X_bc_2], dim=0)
            normals_bc = torch.cat([normals_bc_1, normals_bc_2], dim=0)
        if "ic" in active_losses:
            X_ic = torch.cat([
                residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=2*n_initial//3, n_candidates=4*n_initial, sampling_strategy=sampling_strategy, picking_criterion="multinomial", device=device),
                residual_based_adaptive_sampling(d, pde_model.ic_residual, model, type='ic', n_new=n_initial//3, n_candidates=2*n_initial, sampling_strategy=sampling_strategy, picking_criterion="top_k", device=device),
            ], dim=0)
    else:
        X_pde, X_bc, X_ic, normals_bc = sample_collocation_points(d, n_interior, n_boundary, n_initial, sampling_strategy=sampling_strategy, device=device)

    if spatial_domain is not None:
        lo = spatial_domain[:, 0]
        hi = spatial_domain[:, 1]
        if X_pde is not None: X_pde = scale_samples__spatial(X_pde, lo, hi)
        if X_bc  is not None: X_bc  = scale_samples__spatial(X_bc,  lo, hi)
        if X_ic  is not None: X_ic  = scale_samples__spatial(X_ic,  lo, hi)
    if T != 1.0:
        if X_pde is not None: X_pde = scale_samples__temporal(X_pde, T)
        if X_bc  is not None: X_bc  = scale_samples__temporal(X_bc,  T)

    if "norm" in active_losses:
        x0 = lo + (hi - lo) * torch.rand(n_norm, pde_model.d, device=device)
        _, traj = euler_maruyama_trajectory_bank(
            x0=x0, mu=pde_model.mu, sigma=pde_model.sigma, T=T, n_steps=1000,
        )
        X_norm = traj[:, -1, :]

    precomputed = _precompute_active(pde_model, X_pde, X_bc, X_ic, active_losses, device)
    if normals_bc is not None and "bc" in active_losses:
        precomputed["bc"]["normals"] = normals_bc
    if "norm" in active_losses:
        precomputed["norm"] = {"p_inf": pde_model.p_inf(X_norm)}

    Xs = {"pde": X_pde, "bc": X_bc, "ic": X_ic, "norm": X_norm}
    bss = {"pde": bs_pde, "bc": bs_bc, "ic": bs_ic, "norm": bs_norm}
    bundle = {}
    for k in ("pde", "bc", "ic", "norm"):
        if k in active_losses:
            bundle[k] = DataLoader(CollocationDataset(Xs[k], precomputed[k]), batch_size=bss[k], shuffle=True)

    print("Constructing dataset:")
    for k, loader in bundle.items():
        print(f"  {k}: X.shape = {loader.dataset.X.shape}, bs = {loader.batch_size}")

    return bundle


def create_dataloaders__trajectories(
    model, pde_model, active_losses,
    n_res_points=10_000, bs=1_000, n_trajs=100, T=1.0, nt_steps=100, spatial_domain=None,
    device="cpu",
):
    (bs_pde, bs_bc, bs_ic, bs_norm), (n_interior, n_boundary, n_initial, n_norm) = split_res_points(n_res_points, bs,
        f_pde  = 14 if "pde"  in active_losses else 0,
        f_bc   =  1 if "bc"   in active_losses else 0,
        f_ic   =  1 if "ic"   in active_losses else 0,
        f_norm = 16 if "norm" in active_losses else 0
    )
    d = pde_model.d

    if "pde" not in active_losses: n_interior = 0
    if "bc"  not in active_losses: n_boundary = 0
    if "ic"  not in active_losses: n_initial  = 0
    if "norm"  not in active_losses: n_norm  = 0

    X_pde = X_bc = X_ic = X_norm = None
    normals_bc = None

    x0 = pde_model.sample_x0(n_trajs)
    if "ic" in active_losses:
        X_ic = contruct_trajs_ic(x0, n_initial)
    if "pde" in active_losses:
        X_pde = sample_trajs_res_points(pde_model, x0, T, nt_steps, n_interior)
    if "bc" in active_losses:
        X_bc, normals_bc = sample_bc(n_boundary, d, sampling_strategy='lhs', device=x0.device)
        if spatial_domain is not None:
            lo = spatial_domain[:, 0]
            hi = spatial_domain[:, 1]
            X_bc = scale_samples__spatial(X_bc, lo, hi)
        if T != 1.0:
            X_bc = scale_samples__temporal(X_bc, T)
    if "norm" in active_losses:
        x0 = lo + (hi - lo) * torch.rand(n_norm, pde_model.d, device=device)
        _, traj = euler_maruyama_trajectory_bank(
            x0=x0, mu=pde_model.mu, sigma=pde_model.sigma, T=T, n_steps=1000,
        )
        X_norm = traj[:, -1, :]

    precomputed = _precompute_active(pde_model, X_pde, X_bc, X_ic, active_losses, device)
    if normals_bc is not None and "bc" in active_losses:
        precomputed["bc"]["normals"] = normals_bc
    if "norm" in active_losses:
        precomputed["norm"]["p_inf"] = pde_model.p_inf(X_norm)

    Xs = {"pde": X_pde, "bc": X_bc, "ic": X_ic, "norm": X_norm}
    bss = {"pde": bs_pde, "bc": bs_bc, "ic": bs_ic, "norm": bs_norm}
    bundle = {}
    for k in ("pde", "bc", "ic", "norm"):
        if k in active_losses:
            bundle[k] = DataLoader(CollocationDataset(Xs[k], precomputed[k]), batch_size=bss[k], shuffle=True)
    return bundle


def create_pde_loader(sampling_type, pde_model, settings, device="cpu"):
    """Build a single DataLoader of PDE collocation points only.

    sampling_type:
      - "domain":       uniform / LHS over spatial_domain x [0, T]
      - "trajectories": Euler-Maruyama SDE trajectory bank, residual points
                        sampled uniformly from (traj_idx, time_idx)

    settings keys:
      - n_res_points: total points in the buffer (default 100_000)
      - bs:           points per gradient step (default 1_000)
      - spatial_domain: (d, 2) tensor of [lo, hi] per spatial dim (optional)
      - T:            final time (default 1.0)
      - sampling_strategy: "lhs" | "uniform" (only used for "domain")
      - n_trajs:      number of trajectories (only used for "trajectories",
                      default n_res_points // 100)
      - nt_steps:     SDE steps per trajectory (default 100)
    """
    n_res = settings.get("n_res_points", 100_000)
    bs = settings.get("bs", 1_000)
    spatial_domain = settings.get("spatial_domain")
    T = settings.get("T", 1.0)
    d = pde_model.d

    if sampling_type == "domain":
        strategy = settings.get("sampling_strategy", "lhs")
        X_pde = sample_domain(n_res, d + 1, sampling_strategy=strategy, device=device)
        if spatial_domain is not None:
            lo = spatial_domain[:, 0]
            hi = spatial_domain[:, 1]
            X_pde = scale_samples__spatial(X_pde, lo, hi)
        if T != 1.0:
            X_pde = scale_samples__temporal(X_pde, T)
    elif sampling_type == "trajectories":
        n_trajs = settings.get("n_trajs", max(1, n_res // 100))
        nt_steps = settings.get("nt_steps", 100)
        x0 = pde_model.sample_x0(n_trajs)
        X_pde = sample_trajs_res_points(pde_model, x0, T, nt_steps, n_res)
    else:
        raise NameError(f"Unknown sampling_type '{sampling_type}' (expected 'domain' or 'trajectories')")

    precomputed = _precompute_active(pde_model, X_pde, None, None, active_losses=("pde",), device=device)
    loader = DataLoader(CollocationDataset(X_pde, precomputed["pde"]), batch_size=bs, shuffle=True)
    print(f"PDE-only loader ({sampling_type}): X.shape = {X_pde.shape}, bs = {bs}")
    return loader


def create_dataloaders(sampling_type, model, pde_model, settings, active_losses, device="cpu"):
    """
    Build a bundle dict {term_name: loader_or_buffer} for all terms in active_losses.
    Loader-typed terms ("pde", "bc", "ic") yield DataLoader; "norm" yields the
    {x, p_inf, Z} buffer dict used by the importance-sampled integral loss.
    """
    if sampling_type == "trajectories":
        bundle = create_dataloaders__trajectories(model, pde_model, active_losses, device=device, **settings)
    elif sampling_type == "domain":
        bundle = create_dataloaders__domain(model, pde_model, active_losses, device=device, **settings)
    else:
        raise NameError(f"Incorrect data loader type specified: '{type}'")

    return bundle




#if __name__ == "__main__":
#    dof_per_atom = 3
#    n_atoms = 7
#    d = n_atoms * dof_per_atom
#    r_min = 0.1
#    X = torch.rand(10000, d+1)
#    X[:, -1] = 0.0
#    X_filtered = filter_close_atoms(X, n_atoms, dof_per_atom, r_min)
#    print(X.shape)
#    print(X_filtered.shape)


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Problem setup
    d = 8
    n_traj = 256
    T = 1.0
    n_steps = 100

    # Example initial distribution p0 = N(0, I)
    x0 = torch.randn(n_traj, d, device=device, dtype=dtype)

    # Example drift and diffusion:
    # Ornstein-Uhlenbeck style: dx = -x dt + sqrt(2) dW
    def mu(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    def sigma(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        eye = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, d, d)
        return (2.0 ** 0.5) * eye

    # Build trajectory bank
    times, traj_bank = euler_maruyama_trajectory_bank(
        x0=x0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
    )

    # Sample exactly 1000 residual points row-wise: shape (1000, d)
    x_residual, t_residual = sample_residual_points_from_trajectory_bank(
        traj_bank=traj_bank,
        times=times,
        n_points=1000,
        exclude_t0=True,
    )

    print("x_residual shape:", x_residual.shape)  # should be (1000, d)
    print("t_residual shape:", t_residual.shape)  # should be (1000, 1)