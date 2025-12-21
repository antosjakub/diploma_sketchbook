import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from test_suite import *

torch.manual_seed(41)

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PINN(nn.Module):
    def __init__(self, D, hidden_size=64):
        """
        D: input dimension (d spatial dims + 1 time dim)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)



def visualize_solution_1d(model, t_val=0.1, device='cpu'):
    """Visualize solution for 1D case"""
    x = torch.linspace(0, 1, 200, device=device).view(-1, 1)
    t = torch.ones_like(x) * t
    X = torch.cat([x, t], dim=1)
    
    with torch.no_grad():
        u_pred = model(X)
    
    # Analytical solution
    u_true = u_analytic(X)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, u_pred, 'b-', label='PINN', linewidth=2)
    plt.plot(x, u_true, 'r--', label='Analytical', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Solution at t={t_val}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, torch.abs(u_pred - u_true), 'k-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title(f'Error at t={t_val}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pinn_solution.png', dpi=150)
    print(f'Max error: {torch.abs(u_pred - u_true).max():.6f}')


def visualize_solution_2d(model, t_val=0.1, device='cpu'):
    """Visualize solution for 2D case"""
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    
    X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
    x_flat = X_grid.reshape(-1, 1)
    y_flat = Y_grid.reshape(-1, 1)
    t_flat = torch.ones_like(x_flat) * t_val
    
    X = torch.cat([x_flat, y_flat, t_flat], dim=1)
    
    with torch.no_grad():
        u_pred = model(X)
        U_pred = u_pred.reshape(100, 100)
    
    # Analytical solution
    u_true = u_analytic(X)
    U_true = u_true.reshape(100,100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Predicted solution
    im1 = axes[0].contourf(X_grid, Y_grid, U_pred, levels=50, cmap='jet')
    axes[0].set_title(f'PINN Solution at t={t_val}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # True solution
    im2 = axes[1].contourf(X_grid, Y_grid, U_true, levels=50, cmap='jet')
    axes[1].set_title(f'Analytical Solution at t={t_val}')
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
    plt.savefig('pinn_solution.png', dpi=150)
    print(f'Max error: {error.max():.6f}, Mean error: {error.mean():.6f}')



if __name__ == "__main__":
    # pde eq, analytic sol, rhs, dim, model, losses

    model = torch.load('model.pth', weights_only=False)
    losses = torch.load('losses.pth')
    l2_errs = torch.load('l2_errs.pth')
    import json
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    d = metadata["d"]
    D = d + 1 # space + time


    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150)

    # Plot l2
    plt.figure(figsize=(10, 5))
    plt.semilogy(l2_errs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('L2 error')
    plt.grid(True)
    plt.savefig('l2_errs.png', dpi=150)
    
    # Visualize solution
    print("\nGenerating visualizations...")
    if d == 1:
        visualize_solution_1d(model, t_val=0.1, device=device)
    elif d == 2:
        visualize_solution_2d(model, t_val=0.1, device=device)
    else:
        print(f"Visualization not implemented for d={d} (only 1D and 2D supported)")
        # For d>2, you could compute and print error metrics
        X_test = torch.rand(1000, D, device=device)
        X_test[:, -1] = 0.1  # t = 0.1
        with torch.no_grad():
            u_pred = model(X_test)
            u_true = torch.ones_like(u_pred)
            for i in range(d):
                u_true *= torch.sin(torch.pi * X_test[:, i:i+1])
            u_true *= torch.exp(-d * torch.pi**2 * 0.01 * 0.1)
            error = torch.abs(u_pred - u_true)
            print(f"Test error at t=0.1: Max={error.max():.6f}, Mean={error.mean():.6f}")
    