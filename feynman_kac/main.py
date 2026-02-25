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
