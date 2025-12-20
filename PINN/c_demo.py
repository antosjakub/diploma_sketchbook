import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Simple feedforward neural network
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
    def forward(self, x, y, t):
        # Concatenate inputs
        inputs = torch.cat([x, y, t], dim=1)
        
        # Forward pass through network
        for i, layer in enumerate(self.layers[:-1]):
            inputs = torch.tanh(layer(inputs))
        
        # Output layer (no activation)
        u = self.layers[-1](inputs)
        return u

# Physics-informed loss functions
def pde_loss(model, x, y, t, alpha=0.01):
    """
    Compute PDE residual for 2D heat equation:
    u_t = alpha * (u_xx + u_yy)
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, y, t)
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    
    # PDE residual
    residual = u_t - alpha * (u_xx + u_yy)
    return torch.mean(residual**2)

def initial_condition_loss(model, x, y, t):
    """
    Initial condition: u(x, y, 0) = sin(pi*x) * sin(pi*y)
    """
    u_pred = model(x, y, t)
    u_true = torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return torch.mean((u_pred - u_true)**2)

def boundary_condition_loss(model, x_bc, y_bc, t_bc):
    """
    Boundary condition: u = 0 on all boundaries
    """
    u_pred = model(x_bc, y_bc, t_bc)
    return torch.mean(u_pred**2)

# Generate training data
def generate_training_data(n_interior=2000, n_boundary=400, n_initial=400):
    """Generate collocation points for training"""
    
    # Interior points (for PDE)
    x_int = torch.rand(n_interior, 1) * 2 - 1  # [-1, 1]
    y_int = torch.rand(n_interior, 1) * 2 - 1  # [-1, 1]
    t_int = torch.rand(n_interior, 1) * 0.5    # [0, 0.5]
    
    # Boundary points
    # Four boundaries: x=-1, x=1, y=-1, y=1
    n_per_boundary = n_boundary // 4
    
    # x = -1
    x_bc1 = torch.ones(n_per_boundary, 1) * (-1)
    y_bc1 = torch.rand(n_per_boundary, 1) * 2 - 1
    t_bc1 = torch.rand(n_per_boundary, 1) * 0.5
    
    # x = 1
    x_bc2 = torch.ones(n_per_boundary, 1)
    y_bc2 = torch.rand(n_per_boundary, 1) * 2 - 1
    t_bc2 = torch.rand(n_per_boundary, 1) * 0.5
    
    # y = -1
    x_bc3 = torch.rand(n_per_boundary, 1) * 2 - 1
    y_bc3 = torch.ones(n_per_boundary, 1) * (-1)
    t_bc3 = torch.rand(n_per_boundary, 1) * 0.5
    
    # y = 1
    x_bc4 = torch.rand(n_per_boundary, 1) * 2 - 1
    y_bc4 = torch.ones(n_per_boundary, 1)
    t_bc4 = torch.rand(n_per_boundary, 1) * 0.5
    
    x_bc = torch.cat([x_bc1, x_bc2, x_bc3, x_bc4], dim=0)
    y_bc = torch.cat([y_bc1, y_bc2, y_bc3, y_bc4], dim=0)
    t_bc = torch.cat([t_bc1, t_bc2, t_bc3, t_bc4], dim=0)
    
    # Initial condition points (t=0)
    x_ic = torch.rand(n_initial, 1) * 2 - 1
    y_ic = torch.rand(n_initial, 1) * 2 - 1
    t_ic = torch.zeros(n_initial, 1)
    
    return (x_int, y_int, t_int), (x_bc, y_bc, t_bc), (x_ic, y_ic, t_ic)

# Training function
def train_pinn(model, optimizer, scheduler, n_epochs=10000, 
               lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
    """Train the PINN model"""
    
    # Generate training data
    interior, boundary, initial = generate_training_data()
    x_int, y_int, t_int = interior
    x_bc, y_bc, t_bc = boundary
    x_ic, y_ic, t_ic = initial
    
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute individual losses
        loss_pde = pde_loss(model, x_int, y_int, t_int)
        loss_bc = boundary_condition_loss(model, x_bc, y_bc, t_bc)
        loss_ic = initial_condition_loss(model, x_ic, y_ic, t_ic)
        
        # Total loss with weights
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler
        scheduler.step(loss)
        
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, '
                  f'IC: {loss_ic.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return losses

# Visualization function
def visualize_solution(model, t_val=0.1):
    """Visualize the learned solution at a given time"""
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    y = torch.linspace(-1, 1, 100).view(-1, 1)
    
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.ones_like(x_flat) * t_val
    
    with torch.no_grad():
        u_pred = model(x_flat, y_flat, t_flat)
        U = u_pred.reshape(100, 100).numpy()
    
    # Analytical solution for comparison
    U_true = np.exp(-2 * np.pi**2 * 0.01 * t_val) * np.sin(np.pi * X.numpy()) * np.sin(np.pi * Y.numpy())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Predicted solution
    im1 = axes[0].contourf(X.numpy(), Y.numpy(), U, levels=50, cmap='jet')
    axes[0].set_title(f'PINN Solution at t={t_val}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # True solution
    im2 = axes[1].contourf(X.numpy(), Y.numpy(), U_true, levels=50, cmap='jet')
    axes[1].set_title(f'Analytical Solution at t={t_val}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(U - U_true)
    im3 = axes[2].contourf(X.numpy(), Y.numpy(), error, levels=50, cmap='hot')
    axes[2].set_title(f'Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('pinn_solution.png', dpi=150)
    print(f'Max error: {error.max():.6f}, Mean error: {error.mean():.6f}')

# Main execution
if __name__ == "__main__":
    # Initialize model
    layers = [3, 50, 50, 50, 1]  # Input: (x, y, t), Output: u
    model = PINN(layers)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000, 
        threshold=1e-4
    )
    
    # Train the model
    print("Training PINN for 2D Heat Equation...")
    losses = train_pinn(model, optimizer, scheduler, n_epochs=10000)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150)
    
    # Visualize solution
    print("\nGenerating visualizations...")
    visualize_solution(model, t_val=0.1)
    
    print("\nTraining complete! Check 'pinn_solution.png' and 'training_loss.png'")