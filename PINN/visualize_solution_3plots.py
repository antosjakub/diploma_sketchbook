"""
make 3 plots:
- model
- analytic
- error
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
# provide or will be set randomly
t_val = torch.rand(1).item()
#fixed_dims_vals = torch.rand(d)
fixed_dims_vals = 0.2*torch.ones(d)

# DO ONCE
x = torch.linspace(0, 1, N, device=device)
y = torch.linspace(0, 1, N, device=device)
X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
x_flat = X_grid.reshape(-1, 1)
y_flat = Y_grid.reshape(-1, 1)

# define domain
X_flat_list = []
for di in range(d):
    if di == plot_dims[0]:
        X_flat_list.append(x_flat)
    elif di == plot_dims[1]:
        X_flat_list.append(y_flat)
    else:
        fixed_flat = torch.ones_like(x_flat) * fixed_dims_vals[di]
        X_flat_list.append(fixed_flat)

t_flat = torch.ones_like(x_flat) * t_val
X = torch.cat([*X_flat_list, t_flat], dim=1)


# Model solution 
with torch.no_grad():
    u_pred = model(X)
    U_pred = u_pred.reshape(N, N)

# Analytical solution
u_true = u_analytic(X)
U_true = u_true.reshape(N, N)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Predicted solution
im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
axes[0].set_title(f'PINN Solution at t={t_val:.3f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# True solution
im2 = axes[1].contourf(X_grid, Y_grid, U_true, levels=50, cmap='jet')
axes[1].set_title(f'Analytical Solution at t={t_val:.3f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

# Error
error = torch.abs(U_pred - U_true)
im3 = axes[2].contourf(X_grid, Y_grid, error, levels=50, cmap='hot')
axes[2].set_title(f'Absolute Error')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(f'{dir_name}/pinn_solution_3plots.png', dpi=150)
print(f'Max error: {error.max():.6f}, Mean error: {error.mean():.6f}')


