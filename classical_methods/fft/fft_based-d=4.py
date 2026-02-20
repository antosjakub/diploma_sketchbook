import numpy as np
from scipy.fft import dstn, idstn

def solve_heat_4d(u0, f, alpha, dt, N, num_steps):
    """
    Solve u_t = alpha * Lap(u) + f on [0,1]^4 with u=0 on boundary.
    
    u0  : (N,N,N,N) array of initial condition on interior grid
    f   : callable f(t) -> (N,N,N,N) array of source term
    alpha: diffusivity
    dt  : time step
    N   : interior grid points per dimension
    """
    d = 4
    
    # Grid: x_j = j/(N+1), j=1..N
    j = np.arange(1, N + 1)
    
    # Eigenvalues λ_k = π²(k1²+k2²+k3²+k4²)
    # Build 4D array of eigenvalues
    k1, k2, k3, k4 = np.meshgrid(j, j, j, j, indexing='ij')
    lam = np.pi**2 * (k1**2 + k2**2 + k3**2 + k4**2)
    
    # Precompute exponential integrator factors
    D = np.exp(-alpha * lam * dt)          # decay factor
    G = (1.0 - D) / (alpha * lam)          # integral factor
    
    # DST-I via scipy (type=1 corresponds to Dirichlet sine transform)
    # Normalize: scipy's dstn has norm options; DST-I is self-inverse up to 2^d*(N+1)^d
    norm = 'ortho'   # use orthonormal so IDST = inverse of DST
    
    # Initialize modal coefficients
    u_hat = dstn(u0, type=1, norm=norm)
    
    t = 0.0
    for step in range(num_steps):
        f_hat = dstn(f(t), type=1, norm=norm)
        u_hat = D * u_hat + G * f_hat
        t += dt
    
    # Reconstruct
    u = idstn(u_hat, type=1, norm=norm)
    return u

# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 16          # 16^4 = 65536 interior points
    alpha = 0.01
    dt = 0.01
    T = 0.5
    num_steps = int(T / dt)

    # Grid coordinates for initial condition
    j = np.linspace(1/(N+1), N/(N+1), N)
    x1, x2, x3, x4 = np.meshgrid(j, j, j, j, indexing='ij')

    # Initial condition: product of sines (exact eigenfunction)
    u0 = np.sin(np.pi*x1) * np.sin(np.pi*x2) * np.sin(np.pi*x3) * np.sin(np.pi*x4)

    # Source term (zero here for pure diffusion test)
    f = lambda t: np.zeros((N, N, N, N))

    u_final = solve_heat_4d(u0, f, alpha, dt, N, num_steps)

    # Exact solution: u(x,t) = exp(-4π²αt) * product of sines
    exact = np.exp(-4 * np.pi**2 * alpha * T) * u0
    print(f"Max error vs exact: {np.max(np.abs(u_final - exact)):.2e}")


    np.save("u_4d.npy", u_final)