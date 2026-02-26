import numpy as np
import main


# Parameters
T = 1.0
dt = 0.001
power = 7
N_paths = 10^power
d = 2

# u(t=T,x) = (1/(2pi))^(d/2) e^(-0.5|x|^2)
def u_analytic(t,X):
    tau = T-t
    arg = np.sum((X + tau)**2, axis=1) / (1+2*tau)
    return 1/(1+2*tau)**(d/2) * np.exp(-0.5*arg)

n = 100
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X_grid, Y_grid = np.meshgrid(x, y)
x_flat = X_grid.reshape(-1, 1)
y_flat = Y_grid.reshape(-1, 1)
X = np.concatenate([x_flat, y_flat], axis=1)

def eval_model_on_grid(t_val):
    U_MC = np.zeros(len(X))
    for i in range(len(X)):
        x0 = X[i]
        u_MC, std_err = main.fk_multi_d_advection(t_val, x0, d=d, T=T, dt=dt, N_paths=N_paths)
        U_MC[i] = u_MC
    U_MC = U_MC.reshape(n,n)
    return U_MC

def eval_analytic_on_grid(t_val):
    U_analytic = u_analytic(t_val, X)
    U_analytic = U_analytic.reshape(n,n)
    return U_analytic

def update_grid(t_val):
    U_pred = eval_model_on_grid(t_val)
    U_true = eval_analytic_on_grid(t_val)
    return U_pred, U_true

t_val = T
U_pred, U_true = update_grid(t_val)

from matplotlib import pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))



# Predicted sol
im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
axes[0].set_title(f'Feyman-Kac Solution at t={t_val:.3f}')
plt.colorbar(im1, ax=axes[0])
#
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Analytic sol
im2 = axes[1].contourf(X_grid, Y_grid, U_true, levels=50, cmap='jet')
axes[1].set_title(f'Analytical Solution at t={t_val}')
plt.colorbar(im2, ax=axes[1])
#
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

# Error
error = np.abs(U_pred - U_true)
im3 = axes[2].contourf(X_grid, Y_grid, error, levels=50, cmap='hot')
axes[2].set_title(f'Absolute Error at t={t_val}')
plt.colorbar(im3, ax=axes[2])
#
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')


num_frames = 10
FPS = 5

def update(frame_indx):
    # update pred, true, sol
    t_val = T - frame_indx/num_frames
    U_pred, U_true = update_grid(t_val)

    im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
    axes[0].set_title(f'PINN Solution at t={t_val}')
    #plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(X_grid, Y_grid, U_true, levels=50, cmap='jet')
    axes[1].set_title(f'Analytical Solution at t={t_val}')
    #plt.colorbar(im1, ax=axes[1])

    error = np.abs(U_pred - U_true)
    im3 = axes[2].contourf(X_grid, Y_grid, error, levels=50, cmap='hot')
    axes[2].set_title(f'Absolute Error at t={t_val}')
    #plt.colorbar(im1, ax=axes[2])

    return [im1, im2, im3]

#plt.tight_layout()
#update(num_frames-1)
#plt.savefig('ps.png', dpi=150)

from matplotlib.animation import FuncAnimation
print("Saving animation...")
ani = FuncAnimation(fig, update, frames=range(num_frames), interval=int(1000/FPS), blit=False)
#plt.tight_layout()
ani.save(f"anim_2d_power={power},dt={dt}.gif", writer="ffmpeg", fps=FPS, dpi=100)
print("Animation saved.")
