import numpy as np

# 1d equation \partial_t u + \partial_x u + \partial_xx u = 0 
# with terminal condition e^(-x^2) at t=T

def fk_advection(t_eval, x_eval, T=1.0, N_paths=100000, dt=0.01):
    tau = T - t_eval
    if tau <= 0: return np.exp(-x_eval**2)  # terminal
    
    N_steps = int(tau / dt)
    np.random.seed(42)
    dW = np.sqrt(dt) * np.random.randn(N_paths, N_steps)
    X = np.zeros((N_paths, N_steps + 1))
    X[:, 0] = x_eval
    for i in range(N_steps):
        X[:, i+1] = X[:, i] - dt + dW[:, i]  # drift=-1
    
    payoffs = np.exp(-X[:, -1]**2)
    u_MC = np.mean(payoffs)
    std_err = np.std(payoffs) / np.sqrt(N_paths)
    return u_MC, std_err


# \partial_t u + a \nabla u \cdot v + b \Delta u = 0
# where a,b are constant and v is a constant vector
# and
# u(t=T,x) = (1/(2pi))^(d/2) e^(-0.5|x|^2)
def fk_multi_d_advection(t_eval, x_eval, T=1.0, d=3, a=1.0, b=1.0, v=None, N_paths=10^5, dt=0.01):
    if v is None: v = np.ones(d)
    tau = T - t_eval
    if tau <= 0: return (2*np.pi)**(-d/2) * np.exp(-0.5 * np.sum(x_eval**2))
    
    N_steps = int(tau / dt)
    np.random.seed(42)
    dW = np.sqrt(dt) * np.random.randn(N_paths, N_steps, d)
    X = np.zeros((N_paths, N_steps + 1, d))
    X[:, 0] = x_eval
    drift_vec = a * v * dt
    sigma = np.sqrt(2 * b)
    for i in range(N_steps):
        X[:, i+1] = X[:, i] + drift_vec + sigma * dW[:, i]
    
    norm2 = np.sum(X[:, -1]**2, axis=1)
    payoffs = (2*np.pi)**(-d/2) * np.exp(-0.5 * norm2)
    u_MC = np.mean(payoffs)
    std_err = np.std(payoffs) / np.sqrt(N_paths)
    return u_MC, std_err

if __name__ == "__main__":
    # Parameters
    T = 1.0
    dt = 0.01
    N_paths = 10^5

    # Coordinates of where we want to get the solution
    x0 = 0.3
    t0 = 0.0
    u_MC, std_err = fk_advection(t0, x0, T, N_paths, dt)
    print(f"u(t={t0},x={x0}) = {u_MC:.6f} ± {std_err:.6f}")

    d = 2
    x0 = 0.0 * np.ones(d)
    t0 = 0.0
    u_MC, std_err = fk_multi_d_advection(t0, x0, d=d, T=T, dt=dt, N_paths=N_paths)
    print(f"u(t={t0},x={x0}) = {u_MC:.6f} ± {std_err:.6f}")



