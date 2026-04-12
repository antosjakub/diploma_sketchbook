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


def sample_residual_points_from_bank(
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


import sampling
from torch.utils.data import DataLoader
def create_dataloaders(model, pde_model, n_res_points=10_000, bs=1_000, n_trajs=100, T=1.0, nt_steps=100):
    n_cycles = n_res_points // bs
    bs_segment_size = bs // 16
    bs_pde = bs_segment_size * 14
    bs_bc  = bs_segment_size
    bs_ic  = bs_segment_size
    n_interior = bs_pde * n_cycles
    n_boundary =  bs_bc * n_cycles
    n_initial  =  bs_ic * n_cycles

    x0 = pde_model.sample_x0(n_trajs)
    X_ic = torch.cat([
        x0[:n_initial],
        torch.zeros(n_initial, 1)],
    dim=1)

    times, traj_bank = euler_maruyama_trajectory_bank(
        x0=x0,
        mu=pde_model.mu,
        sigma=pde_model.sigma,
        T=T,
        n_steps=nt_steps,
    )

    X_pde = torch.cat(
        # Sample n residual points row-wise: shape (n, d)
        sample_residual_points_from_bank(
            traj_bank=traj_bank,
            times=times,
            n_points=n_interior,
            exclude_t0=True,
        ), dim=1
    )

    X_boundary, _ = sampling.sample_bc(n_boundary, d, sampling_strategy='lhs', device=device)
    #X_pde, X_bc, X_ic = sampling.sample_collocation_points(d, n_interior, n_boundary, n_initial, sampling_strategy='lhs', device=device)
    #X_pde[:,:-1] = 4.0 * X_pde[:,:-1] - 2.0
    #X_bc[:,:-1] = 4.0 * X_bc[:,:-1] - 2.0
    #X_ic[:,:-1] = 4.0 * X_ic[:,:-1] - 2.0
    #X_pde[:,-1:] *= 1.5
    #X_bc[:,-1:] *= 1.5


    precomputed = pde_model.precompute(X_pde, X_ic)

    # TD: change bs here to bs_pde, bs_ic:
    loader_pde  = DataLoader(sampling.CollocationDataset(X_pde, precomputed["pde"]), batch_size=bs_pde, shuffle=True)
    loader_ic  = DataLoader(sampling.CollocationDataset(X_ic, precomputed["ic"]), batch_size=bs_ic, shuffle=True)

    return loader_pde, loader_ic


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
    x_residual, t_residual = sample_residual_points_from_bank(
        traj_bank=traj_bank,
        times=times,
        n_points=1000,
        exclude_t0=True,
    )

    print("x_residual shape:", x_residual.shape)  # should be (1000, d)
    print("t_residual shape:", t_residual.shape)  # should be (1000, 1)