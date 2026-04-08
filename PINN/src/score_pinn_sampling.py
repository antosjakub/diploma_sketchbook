import torch
from typing import Callable, Tuple, Optional

import torch
from typing import Callable, Tuple


def euler_maruyama_trajectory_bank_constant_iso_diag(
    x0: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigma: float,
    T: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler-Maruyama trajectory bank for the SDE

        dx = f(x, t) dt + sigma * I dW_t

    where G = sigma * I is a constant diagonal diffusion matrix with the
    same value sigma on every diagonal entry.

    Parameters
    ----------
    x0 : torch.Tensor
        Initial samples of shape (n_traj, d).
    f : callable
        Drift function with signature:
            f(x, t) -> tensor of shape (n_traj, d)
        where x has shape (n_traj, d) and t has shape (n_traj, 1).
    sigma : float
        Constant value on the diagonal of G.
    T : float
        Final time.
    n_steps : int
        Number of Euler-Maruyama steps.

    Returns
    -------
    times : torch.Tensor
        Time grid of shape (n_steps + 1,).
    traj_bank : torch.Tensor
        Simulated trajectories of shape (n_traj, n_steps + 1, d),
        with traj_bank[:, n, :] = x_{t_n}.
    """
    if x0.ndim != 2:
        raise ValueError("x0 must have shape (n_traj, d).")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    device = x0.device
    dtype = x0.dtype
    n_traj, d = x0.shape

    dt = T / n_steps
    sqrt_dt = dt**0.5

    times = torch.linspace(0.0, T, n_steps + 1, device=device, dtype=dtype)

    x = x0.clone()
    traj_bank = torch.empty(n_traj, n_steps + 1, d, device=device, dtype=dtype)
    traj_bank[:, 0, :] = x

    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)

    for n in range(n_steps):
        #t_n = torch.full((n_traj, 1), times[n], device=device, dtype=dtype)
        #drift = f(x, t_n)  # shape: (n_traj, d)

        drift = f(x)  # shape: (n_traj, d)
        if drift.shape != (n_traj, d):
            raise ValueError("f(x, t) must return shape (n_traj, d).")

        dW = torch.randn(n_traj, d, device=device, dtype=dtype) * sqrt_dt

        x = x + drift * dt + sigma_t * dW
        traj_bank[:, n + 1, :] = x

    return times, traj_bank



def euler_maruyama_trajectory_bank(
    x0: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    G: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    T: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate an SDE trajectory bank with Euler-Maruyama:
        dx = f(x,t) dt + G(x,t) dW_t

    Parameters
    ----------
    x0 : torch.Tensor
        Initial samples, shape (n_traj, d).
    f : callable
        Drift function. Signature:
            f(x, t) -> tensor of shape (n_traj, d)
        where x has shape (n_traj, d), t has shape (n_traj, 1) or scalar-like tensor.
    G : callable
        Diffusion function. Signature:
            G(x, t) -> tensor of shape (n_traj, d, m)
        Usually m = d for full Brownian motion, but it can be different.
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
        traj_bank[:, n, :] stores x_{t_n}.
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

    for n in range(n_steps):
        t_n = torch.full((n_traj, 1), times[n], device=device, dtype=dtype)

        drift = f(x, t_n)  # (n_traj, d)
        diffusion = G(x, t_n)  # (n_traj, d, m)

        if diffusion.ndim != 3 or diffusion.shape[0] != n_traj or diffusion.shape[1] != d:
            raise ValueError("G(x,t) must return shape (n_traj, d, m).")

        m = diffusion.shape[2]
        dW = torch.randn(n_traj, m, device=device, dtype=dtype) * sqrt_dt

        # diffusion @ dW
        diffusion_step = torch.einsum("ndm,nm->nd", diffusion, dW)

        x = x + drift * dt + diffusion_step
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
    bs_pde = bs // 10 * 9
    bs_ic =  bs // 10
    n_pde = bs_pde * n_cycles
    n_ic = bs_ic * n_cycles

    x0 = pde_model.sample_x0(n_trajs)
    X_ic = torch.cat([
        x0[:n_ic],
        torch.zeros(n_ic, 1)],
    dim=1)

    if False:
        times, traj_bank = euler_maruyama_trajectory_bank(
            x0=x0,
            f=pde_model.mu,
            G=pde_model.sigma,
            T=T,
            n_steps=nt_steps,
        )
    else:
        times, traj_bank = euler_maruyama_trajectory_bank_constant_iso_diag(
            x0=x0,
            f=pde_model.mu,
            sigma=pde_model.sigma,
            T=T,
            n_steps=nt_steps,
        )

    X_pde = torch.cat(
        # Sample n residual points row-wise: shape (n, d)
        sample_residual_points_from_bank(
            traj_bank=traj_bank,
            times=times,
            n_points=n_pde,
            exclude_t0=True,
        ), dim=1
    )


    precomputed = pde_model.precompute(X_pde, X_ic)

    # change bs here to bs_pde, bs_ic:
    loader_pde  = DataLoader(sampling.CollocationDataset(X_pde, precomputed["pde"]), batch_size=bs, shuffle=True)
    loader_ic  = DataLoader(sampling.CollocationDataset(X_ic, precomputed["ic"]), batch_size=bs, shuffle=True)

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
    def f(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    def G(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        eye = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, d, d)
        return (2.0 ** 0.5) * eye

    # Build trajectory bank
    times, traj_bank = euler_maruyama_trajectory_bank(
        x0=x0,
        f=f,
        G=G,
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