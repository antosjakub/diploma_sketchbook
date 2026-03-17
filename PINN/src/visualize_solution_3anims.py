"""
visualize a specific time interval
save as gif
"""

import sys
if len(sys.argv) > 1:
    dir_name = sys.argv[1]
else:
    dir_name = 'run_latest'
import utility
import torch
model, u_analytic, pde_metadata, model_metadata = utility.header(dir_name)
d = model_metadata["args"]["d"]


# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# seed
torch.manual_seed(41)

# config
plot_dims = [0,1]
N = 100
# fixed_dims_vals = torch.rand(d)
fixed_dims_vals = 0.2*torch.ones(d)

def eval_model_on_grid(X):
    # Predicted sol
    with torch.no_grad():
        u_pred = model(X)
        U_pred = u_pred.reshape(N, N)
    return U_pred

def eval_analytic_on_grid(X):
    # Analytic sol
    u_true = u_analytic(X)
    U_true = u_true.reshape(N, N)
    return U_true


# Create domain mesh
x = torch.linspace(0, 1, N, device=device)
y = torch.linspace(0, 1, N, device=device)
X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
x_flat = X_grid.reshape(-1, 1)
y_flat = Y_grid.reshape(-1, 1)

X_flat_list = []
for di in range(d):
    if di == plot_dims[0]:
        X_flat_list.append(x_flat)
    elif di == plot_dims[1]:
        X_flat_list.append(y_flat)
    else:
        fixed_flat = torch.ones_like(x_flat) * fixed_dims_vals[di]
        X_flat_list.append(fixed_flat)

t_val = 0.0
t_flat = torch.ones_like(x_flat) * t_val
X = torch.cat([*X_flat_list, t_flat], dim=1)

def update_grid(t_val):
    #t_flat = torch.ones_like(x_flat) * t_val
    X[:,-1] = t_val
    U_pred = eval_model_on_grid(X)
    U_true = eval_analytic_on_grid(X)
    return U_pred, U_true



import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))


U_pred, U_true = update_grid(t_val)

# Predicted sol
im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
axes[0].set_title(f'PINN Solution at t={t_val}')
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
error = torch.abs(U_pred - U_true)
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
    t_val = frame_indx/num_frames
    U_pred, U_true = update_grid(t_val)

    im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
    axes[0].set_title(f'PINN Solution at t={t_val}')
    #plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(X_grid, Y_grid, U_true, levels=50, cmap='jet')
    axes[1].set_title(f'Analytical Solution at t={t_val}')
    #plt.colorbar(im1, ax=axes[1])

    error = torch.abs(U_pred - U_true)
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
ani.save(f"{dir_name}/pinn_solution_3plots_anim.gif", writer="ffmpeg", fps=FPS, dpi=100)
print("Animation saved.")
