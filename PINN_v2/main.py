import torch
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="64,64,64", type=str, help="")
parser.add_argument("--n_steps", default=10_000, type=int, help="")
parser.add_argument("--n_steps_log", default=500, type=int, help="")
parser.add_argument("--n_points_pde", default=2000, type=int, help="")
parser.add_argument("--n_points_bc", default=400, type=int, help="")
parser.add_argument("--n_points_ic", default=400, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=10.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--gamma", default=0.9995, type=float, help="")
parser.add_argument("--lr", default=0.001, type=float, help="")


class PINN(nn.Module):
    def __init__(self, D, layers=[64]):
        """
        D: input dimension (d spatial dims + 1 time dim)
        """
        super().__init__()

        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(nn.Tanh())

        self.net = nn.Sequential(
            nn.Linear(D, layers[0]), nn.Tanh(),
            *net_layers,
            nn.Linear(layers[-1], 1)
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


def pde_loss(model, X_in, pde_residual):
    """
    X_in: (batch_size, d+1) tensor
    """
    X_in.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X_in)
    residual = pde_residual(X_in, u, grad_u, spatial_laplace_u)
    return torch.mean(residual**2)

def initial_condition_loss(model, X_ic, ic_residual):
    """
    X_ic: (batch_size, d+1) tensor with t = 0
    IC: u(x,0) = u_IC(x)
    """
    u = model(X_ic)
    residual = ic_residual(X_ic, u)
    return torch.mean(residual**2)

def boundary_condition_loss(model, X_bc, bc_residual):
    """
    X_bc: (batch_size, d+1) tensor with points on boundary
    BC: u(x,t) = u_BC(x,t)
    """
    u = model(X_bc)
    residual = bc_residual(X_bc, u)
    return torch.mean(residual**2)


def sample_collocation_points(
        d,
        n_interior, n_boundary, n_initial, 
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
    
    # Boundary points: spatial coords on boundary, t random in [0,1]
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
        pde_residual, bc_residual, ic_residual,
        u_analytic,
        d,
        n_steps=10_000, 
        n_steps_log=500,
        n_points_pde=2000, n_points_bc=400, n_points_ic=400,
        lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
        device='cpu'
    ):
    """Train the PINN model"""
    
    # Generate training data
    X_interior, X_boundary, X_initial = sample_collocation_points(
        d, n_points_pde, n_points_bc, n_points_ic, device
    )
    
    losses = []
    l2_errs = []
    
    for si in range(n_steps):
        optimizer.zero_grad()
        
        # Sample the training points

        # Compute individual losses
        loss_pde = pde_loss(model, X_interior, pde_residual)
        loss_bc = boundary_condition_loss(model, X_boundary, bc_residual)
        loss_ic = initial_condition_loss(model, X_initial, ic_residual)
        
        # Total loss with weights
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler
        scheduler.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (si + 1) % n_steps_log == 0:
            X_interior_test, X_boundary_test, X_initial_test = sample_collocation_points(
                d, n_points_pde, n_points_bc, n_points_ic, device=device
            )
            u_pred = model(X_interior_test)
            u_true = u_analytic(X_interior_test)
            l2_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
            l2_errs.append(l2_err)
            print(f'Step {si+1}/{n_steps}, Loss: {loss.item():.6f}, '
                  f'PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, '
                  f'IC: {loss_ic.item():.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'L2: {l2_err:.6f}'
            )
    
    return losses, l2_errs


# Main execution
if __name__ == "__main__":

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Arguments
    args = parser.parse_args([] if "__file__" not in globals() else None)
    torch.manual_seed(args.seed)
    d = args.d # space dims
    D = d + 1 # space + time dims
    layers = list(map(lambda x: int(x), args.layers.split(",")))
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = PINN(D, layers).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Option 1: ExponentialLR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # PDE equation
    import pde_models
    pde_model = pde_models.HeatEquation(d)
    
    # Train the model
    losses, l2_errs = train_pinn(
        model, optimizer, scheduler,
        pde_model.pde_residual, pde_model.bc_residual, pde_model.ic_residual,
        pde_model.u_analytic,
        d,
        n_steps=args.n_steps,
        n_steps_log=args.n_steps_log,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ic=args.lambda_ic,
        n_points_pde=args.n_points_pde, n_points_bc=args.n_points_bc, n_points_ic=args.n_points_ic,
        device=device
    )
    print("\nTraining complete!")
    

    # Save the results
    torch.save(model, 'model.pth')

    import json
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    torch.save(torch.tensor(losses), 'losses.pth')
    torch.save(torch.tensor(l2_errs), 'l2_errs.pth')
    print("\nResults saved.")

   