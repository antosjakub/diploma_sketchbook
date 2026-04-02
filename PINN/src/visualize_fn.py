"""
make 3 plots:
- model
- analytic
- error
"""
import torch


# config
def plot_fn(u_fn, d, dir_name, t_val=0.25, plot_dims=[0,1], N=100, device='cpu'):

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

    U_fn = u_fn(X).reshape(N, N)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))

    # Predicted solution
    im1 = axes.contourf(X_grid, Y_grid, U_fn, levels=50, cmap='jet')
    axes.set_title(f'PINN Solution at t={t_val:.3f}')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.colorbar(im1, ax=axes)

    plt.tight_layout()
    plt.savefig(f'{dir_name}/visualization_fn.png', dpi=150)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dir_name = sys.argv[1]
    else:
        dir_name = 'run_latest'


    d = 2
    import pde_models
    #u_fn = pde_models.TravellingGaussPacket(d, gamma=1).u_analytic
    u_ic = pde_models.TravellingGaussPacket(d, gamma=1).u_ic
    u_fn = lambda X: u_ic(X[:,:-1])
    plot_fn(u_fn, d, dir_name, t_val=0.0)