import torch
import torch.nn as nn

from test_suite import *

# seed
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

    # Gradient - spatial & temporatal
    grad_u = torch.autograd.grad(
        inputs=X,
        outputs=u,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Laplacian - spatial only
    spatial_laplace_u = torch.zeros_like(u)
    for i in range(D-1):
        hess_row = torch.autograd.grad(
            inputs=X,
            outputs=grad_u[:,i].sum(),
            grad_outputs=torch.tensor(1.0),
            create_graph=True,
            retain_graph=True
        )[0]
        spatial_laplace_u += hess_row[:,i:i+1]

    # shapes: bs x 1, bs x D, bs x 1
    return u, grad_u, spatial_laplace_u


def sample_hypercube_boundary(num_samples, d, device='cpu'):
    """
    Boundary sampling for d-dimensional hypercube [0,1]^d
    Parameters:
    - num_samples: number of points to sample
    - d: num of spatial dimensions
    - device: 'cuda' or 'cpu'
    Returns:
    - tensor of shape (num_samples, d)
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


def pde_loss(model, X_in):
    """
    X_in: (batch_size, d+1) tensor
    """
    X_in.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X_in)
    residual = pde_residual(X_in, u, grad_u, spatial_laplace_u)
    return torch.mean(residual**2)

def initial_condition_loss(model, X_ic):
    """
    X_ic: (batch_size, d+1) tensor with t = 0
    IC: u(x,0) = u_0(x)
    """
    u_pred = model(X_ic)
    u_true = u_0(X_ic[:,:-1])
    return torch.mean((u_pred - u_true)**2)

def boundary_condition_loss(model, X_bc):
    """
    X_bc: (batch_size, d+1) tensor with points on boundary
    BC: u(x,t) = 0
    """
    u_pred = model(X_bc)
    u_true = u_D(X_bc)
    return torch.mean((u_pred - u_true)**2)


def generate_training_data(
        d,
        n_interior=2_000, n_boundary=400, n_initial=400, 
        device='cpu'
    ):
    """
    Generate collocation points for training
    Parameters:
    - d: spatial dimensions
    - device: 'cuda' or 'cpu'
    """
    # Interior points (for PDE): [x1, ..., xd, t]
    X_interior = torch.rand(n_interior, d+1, device=device)
    
    # Boundary points: spatial coords on boundary, t random in [0, 1]
    x_boundary = sample_hypercube_boundary(n_boundary, d, device=device)
    t_boundary = torch.rand(n_boundary, 1, device=device)
    X_boundary = torch.cat([x_boundary, t_boundary], dim=1)
    
    # Initial condition points: spatial coords random in [0,1]^d, t=0
    x_initial = torch.rand(n_initial, d, device=device)
    t_initial = torch.zeros(n_initial, 1, device=device)
    X_initial = torch.cat([x_initial, t_initial], dim=1)
    
    return X_interior, X_boundary, X_initial


def train_pinn(
        model,
        optimizer, scheduler,
        d,
        n_epochs=10_000, 
        lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
        device='cpu'
    ):
    """Train the PINN model"""
    
    # Generate training data
    X_interior, X_boundary, X_initial = generate_training_data(
        d=d, n_interior=2000, n_boundary=400, n_initial=400, device=device
    )
    
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute individual losses
        loss_pde = pde_loss(model, X_interior)
        loss_bc = boundary_condition_loss(model, X_boundary)
        loss_ic = initial_condition_loss(model, X_initial)
        
        # Total loss with weights
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        
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
                  f'IC: {loss_ic.item():.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return losses


# Main execution
if __name__ == "__main__":
    # space dims
    d = 2
    # space + time dims
    D = d + 1
    
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
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
    

    # Save the results
    torch.save(model, 'model.pth')

    import json
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump({"d": d}, f, ensure_ascii=False, indent=4)

    torch.save(torch.tensor(losses), 'losses.pth')
    print("\nResults saved.")

   