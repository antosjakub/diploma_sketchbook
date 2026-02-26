import numpy as np
import main


# Parameters
T = 1.0
dt = 0.01
N_paths = 10^5
d = 2

# u(t=T,x) = (1/(2pi))^(d/2) e^(-0.5|x|^2)
def u_analytic(t,X):
    tau = T-t
    arg = np.sum((X + tau)**2, axis=1) / (1+2*tau)
    return 1/(1+2*tau)**(d/2) * np.exp(-0.5*arg)


t0 = 0.3

n = 100
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X_grid, Y_grid = np.meshgrid(x, y)
x_flat = X_grid.reshape(-1, 1)
y_flat = Y_grid.reshape(-1, 1)
X = np.concatenate([x_flat, y_flat],axis=1)
U_MC = np.zeros(len(X))
for i in range(len(X)):
    x0 = X[i]
    u_MC, std_err = main.fk_multi_d_advection(t0, x0, d=d, T=T, dt=dt, N_paths=N_paths)
    U_MC[i] = u_MC
U_MC = U_MC.reshape(n,n)
U_analytic = u_analytic(t0, X)
U_analytic = U_analytic.reshape(n,n)



from matplotlib import pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Predicted solution
im1 = axes[0].contourf(X_grid, Y_grid, U_MC, levels=50, cmap='jet')
axes[0].set_title(f'Feyman-Kac Solution at t={t0:.3f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# True solution
im2 = axes[1].contourf(X_grid, Y_grid, U_analytic, levels=50, cmap='jet')
axes[1].set_title(f'Analytical Solution at t={t0:.3f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

# Error
error = np.abs(U_MC - U_analytic)
im3 = axes[2].contourf(X_grid, Y_grid, error, levels=50, cmap='hot')
axes[2].set_title(f'Absolute Error')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(f'plot_2d.png', dpi=150)