import numpy as np
from scipy.fft import dstn, idstn
from itertools import product

def solve_heat_nd(u0, f, alpha, dt, N, num_steps, d=None):
    """
    Solve u_t = alpha * Lap(u) + f on [0,1]^d, u=0 on boundary.

    u0        : d-dimensional array of shape (N,)*d
    f         : callable f(t) -> array of shape (N,)*d
    alpha     : diffusivity
    dt        : timestep
    N         : interior grid points per dimension
    num_steps : number of timesteps
    d         : dimension (inferred from u0.shape if None)

    Returns u at final time, shape (N,)*d.
    """
    if d is None:
        d = u0.ndim
    assert u0.shape == (N,) * d, "Shape mismatch"

    # --- Eigenvalues λ_k = π² Σ k_i² ---
    # Build d-dimensional array of eigenvalues efficiently
    # using broadcasting instead of meshgrid (avoids materializing
    # d copies of N^d arrays)
    k = np.arange(1, N + 1, dtype=np.float64)
    k2 = (np.pi * k) ** 2                     # shape (N,)

    lam = np.zeros((N,) * d)
    for i in range(d):
        # Add k_i² contribution by broadcasting along axis i
        shape = [1] * d
        shape[i] = N
        lam = lam + k2.reshape(shape)          # broadcasts to (N,)*d

    # --- Exponential integrator factors ---
    D = np.exp(-alpha * lam * dt)              # decay, shape (N,)*d
    G = np.where(lam > 0, (1.0 - D) / (alpha * lam), dt)  # safe divide

    # --- DST axes ---
    axes = tuple(range(d))
    norm = 'ortho'

    # --- Initialize ---
    # s = something smaller... - in reciprocal space
    u_hat = dstn(u0, type=1, norm=norm, axes=axes)

    # --- Time integration ---
    t = 0.0
    for step in range(num_steps):
        # s = N - back to the original
        f_hat = dstn(f(t), type=1, norm=norm, axes=axes)
        u_hat = D * u_hat + G * f_hat
        t += dt

    return idstn(u_hat, type=1, norm=norm, axes=axes)


# ── Helper: build coordinate grid without meshgrid ────────────────────────────
def interior_grid(N, d):
    """Returns list of d coordinate arrays, broadcast-compatible."""
    j = np.linspace(1/(N+1), N/(N+1), N)
    coords = []
    for i in range(d):
        shape = [1]*d
        shape[i] = N
        coords.append(j.reshape(shape))    # each is broadcastable to (N,)*d
    return coords                          # use: x = coords[i] * np.ones((N,)*d)


# ── Validation: known exact solution ──────────────────────────────────────────
def run_test(d, N=12, alpha=0.01, dt=0.01, T=0.5):
    """
    Exact solution: u(x,t) = exp(-d π² α t) * Π sin(π x_i)
    (lowest mode, k=(1,1,...,1))
    """
    coords = interior_grid(N, d)
    lam_exact = d * np.pi**2              # eigenvalue of k=(1,...,1)

    # Build u0 via broadcasting (never materialize meshgrid explicitly)
    u0 = np.ones((N,)*d)
    for xi in coords:
        u0 = u0 * np.sin(np.pi * xi)     # sequential multiply, O(d * N^d)

    f  = lambda t: np.zeros((N,)*d)
    num_steps = int(T / dt)

    u_final = solve_heat_nd(u0, f, alpha, dt, N, num_steps, d=d)

    exact = np.exp(-lam_exact * alpha * T) * u0
    err = np.max(np.abs(u_final - exact))
    mem_gb = u0.nbytes / 1e9
    print(f"d={d}, N={N}: max_error={err:.2e}, grid_mem={mem_gb:.3f} GB")
    return err


if __name__ == "__main__":
    run_test(d=4, N=32)
