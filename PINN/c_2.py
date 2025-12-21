import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

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

def compute_derivatives(model, X):
    """
    Compute u, grad u, and laplace u
    X: (batch_size, D) where D = d + 1 (spatial dims + time)
    """
    u = model(X)
    bs, D = X.shape

    # Gradient
    grad_u = torch.autograd.grad(
        inputs=X,
        outputs=u,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Laplacian - stored as [d2u/dx1^2, d2u/dx2^2, ..., d2u/dxd^2, d2u/dt^2]
    laplace_u = torch.zeros_like(X)
    for i in range(D):
        hess_row = torch.autograd.grad(
            inputs=X,
            outputs=grad_u[:,i].sum(),
            grad_outputs=torch.tensor(1.0),
            create_graph=True,
            retain_graph=True
        )[0]
        laplace_u[:,i] = hess_row[:,i]

    # shapes: bs x 1, bs x D, bs x D 
    return u, grad_u, laplace_u

def sample_hypercube_boundary(d, num_samples, device='cpu'):
    """
    Fast boundary sampling for d-dimensional hypercube [0,1]^d.
    
    Parameters:
    - d: num of spatial dimensions
    - num_samples: number of boundary points
    - device: 'cuda' or 'cpu'
    
    Returns:
    - tensor of shape (num_samples, d) with requires_grad=False
    """
    # Sample all coordinates uniformly from [0,1]
    samples = torch.rand(num_samples, d, device=device, requires_grad=False)
    
    # Choose which dimension to fix for each sample
    fixed_dims = torch.randint(0, d, (num_samples,), device=device)
    
    # Choose whether to fix to 0 or 1 for each sample
    fixed_values = torch.randint(0, 2, (num_samples,), device=device).float()
    
    # Set the fixed dimension to 0 or 1
    samples[torch.arange(num_samples, device=device), fixed_dims] = fixed_values
    
    return samples

def pde_loss(model, X, alpha=0.01, d=2):
    """
    Compute PDE residual for d-dimensional heat equation:
    u_t = alpha * Δu
    where Δu = d²u/dx1² + d²u/dx2² + ... + d²u/dxd²
    
    X: (batch_size, d+1) tensor where columns are [x1, x2, ..., xd, t]
    """
    X.requires_grad_(True)
    
    u, grad_u, laplace_u = compute_derivatives(model, X)
    
    # Time derivative (last column)
    u_t = grad_u[:, -1]
    
    # Laplacian (sum of second derivatives in spatial dimensions)
    spatial_laplacian = laplace_u[:, :d].sum(dim=1)
    
    # PDE residual: u_t - alpha * Δu = 0
    residual = u_t - alpha * spatial_laplacian
    return torch.mean(residual**2)

def initial_condition_loss(model, X_ic, d=2):
    """
    Initial condition: u(x1, ..., xd, 0) = prod(sin(pi * xi))
    X_ic: (batch_size, d+1) tensor with t=0
    """
    u_pred = model(X_ic)
    
    # Initial condition: product of sin(pi * xi) for all spatial dimensions
    u_true = torch.ones_like(u_pred)
    for i in range(d):
        u_true = u_true * torch.sin(torch.pi * X_ic[:, i:i+1])
    
    return torch.mean((u_pred - u_true)**2)

def boundary_condition_loss(model, X_bc):
    """
    Boundary condition: u = 0 on all boundaries
    X_bc: (batch_size, d+1) tensor with points on boundary
    """
    u_pred = model(X_bc)
    return torch.mean(u_pred**2)

def generate_training_data(d=2, n_interior=2000, n_boundary=400, n_initial=400, 
                          t_max=1.0, device='cpu'):
    """
    Generate collocation points for training
    
    Parameters:
    - d: spatial dimensions
    - n_interior: number of interior points for PDE
    - n_boundary: number of boundary points
    - n_initial: number of initial condition points
    - t_max: maximum time value
    - device: 'cuda' or 'cpu'
    """
    # Interior points (for PDE): [x1, ..., xd, t]
    X_interior = torch.rand(n_interior, d + 1, device=device)
    X_interior[:, -1] = X_interior[:, -1] * t_max  # Scale time to [0, t_max]
    
    # Boundary points: spatial coords on boundary, t random in [0, t_max]
    X_boundary_spatial = sample_hypercube_boundary(d, n_boundary, device=device)
    t_boundary = torch.rand(n_boundary, 1, device=device) * t_max
    X_boundary = torch.cat([X_boundary_spatial, t_boundary], dim=1)
    
    # Initial condition points: spatial coords random in [0,1]^d, t=0
    X_initial_spatial = torch.rand(n_initial, d, device=device)
    t_initial = torch.zeros(n_initial, 1, device=device)
    X_initial = torch.cat([X_initial_spatial, t_initial], dim=1)
    
    return X_interior, X_boundary, X_initial

def train_pinn(model, optimizer, scheduler, d=2, n_epochs=10_000, 
               lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
               device='cpu'):
    """Train the PINN model"""
    
    # Generate training data
    X_interior, X_boundary, X_initial = generate_training_data(
        d=d, n_interior=2000, n_boundary=400, n_initial=400, device=device
    )
    
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute individual losses
        loss_pde = pde_loss(model, X_interior, alpha=0.01, d=d)
        loss_bc = boundary_condition_loss(model, X_boundary)
        loss_ic = initial_condition_loss(model, X_initial, d=d)
        
        # Total loss with weights
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        #loss = lambda_bc * loss_bc + lambda_ic * loss_ic
        #loss = lambda_pde * loss_pde
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler (for ExponentialLR, no argument needed)
        scheduler.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 500 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, '
                  f'IC: {loss_ic.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return losses


# Main execution
if __name__ == "__main__":
    # Set spatial dimension
    d = 2  # Change this to 1, 2, 3, etc. for different dimensions
    D = d + 1  # Total input dimension (spatial + time)
    
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D Heat Equation")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = PINN(D=D, hidden_size=64).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Option 1: ExponentialLR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    # Option 2: ReduceLROnPlateau (uncomment to use)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=1000, 
    #     threshold=1e-4, verbose=True
    # )

    # Train the model
    losses = train_pinn(model, optimizer, scheduler, d=d, 
                       n_epochs=5_000, device=device)
    print("\nTraining complete!")
    
    # Save the results - pde eq, analytic sol, rhs
    # dim
    # model
    # losses

    torch.save(model, 'model.pth')

    import json
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump({"d": d}, f, ensure_ascii=False, indent=4)

    torch.save(torch.tensor(losses), 'losses.pth')
    print("\nResults saved.")

    