import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, D, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def compute_derivatives(model, X):
    """Compute u, grad u, and laplace u"""

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
    
    # Laplacian - stored as [d2u/dx2, d2u/dy2, ...]
    laplace_u = torch.zeros_like(X)
    for i in range(D):
        hess_row = torch.autograd.grad(
            inputs=X,
            outputs=grad_u[:,i].sum(),
            grad_outputs=torch.tensor(1.0),
            create_graph=False,
            retain_graph=(i<D-1)
        )[0]
        laplace_u[:,i] = hess_row[:,i]

    # shapes: bs x 1, bs x D, bs x D 
    return u, grad_u, laplace_u






def sample_hypercube_boundary(d, num_samples, device='cuda'):
    """
    Fast boundary sampling for d-dimensional hypercube [0,1]^d.
    Optimized for PyTorch and GPU acceleration.
    
    Parameters:
    - d: num of dimensions
    - num_samples: number of boundary points
    - device: 'cuda' or 'cpu'
    
    Returns:
    - tensor of shape (num_samples, d) with requires_grad=True
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



D = 3
d = D-1


alpha = 1.0
#def u_0(x):
#    # takes in a matrix, return a vector
#    return torch.prod(torch.sin(torch.pi * x), dim=1)
#def u_D(x):
#    # takes in a matrix, return a vector
#    return torch.prod(torch.sin(torch.pi * x), dim=1)
def u_analytic_fun(X):
    # takes in a matrix, return a vector
    u_space = torch.prod(torch.sin(torch.pi * X[:,1:]), dim=1)
    u_time = torch.exp(- alpha * torch.pi**2 * X[:,0])
    return u_space * u_time

def f_fun(x):
    # takes in a matrix, return a vector
    return -d * torch.pi**2 * torch.prod(torch.sin(torch.pi * x), dim=1)


def train_pinn(model, epochs=5000, lr=1e-3):
    """Train the PINN"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9995
    )
    
    history = {'loss': [], 'l2_error': []}
    N_pde = 1_000
    #N_bc = 500
    #N_ic = 500
    #N_tot = N_pde + N_bc + N_ic
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Loss setup work
        # inside, boundary, intitial
        #X = torch.rand(N_tot, D, requires_grad=False)
        #X[N_pde:(N_pde+N_bc), 1:] = sample_hypercube_boundary(d)
        #X[N_pde+N_bc:, 0] = 0
        #X._requires_grad = True
        # Loss calc based on the PDE in question:
        #u, grad_u, laplace_u = compute_derivatives(model, X)
        #f = f_fun(X)
        #diff = grad_u[:,0] - alpha * laplace_u[:,1:].sum(dim=1) - f
        #loss = torch.mean(diff ** 2)

        X = torch.rand(N_pde, D, requires_grad=True)
        u = model(X)
        u_analytic = u_analytic_fun(X)
        loss = torch.mean( (u - u_analytic)**2 )

        # Update
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Log progress
        if epoch % (epochs//10) == 0:
            with torch.no_grad():
                # Compute L2 error on test points
                X = torch.rand(10*N_pde, D)
                u_pred = model(X)
                u_analytic = u_analytic_fun(X)
                l2_error = torch.sqrt(torch.mean((u_pred - u_analytic) ** 2)).item()
                
                history['loss'].append(loss.item())
                history['l2_error'].append(l2_error)
                
                print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | L2 Error: {l2_error:.6f}")
    
    return history


if __name__ == "__main__":
    model = PINN(D)
    history = train_pinn(model)
    torch.save(model, 'model.pth')

# Example usage for Poisson equation: laplace u = f
#model = PINN(d)
#x = torch.randn(100, d, requires_grad=True)
#u, grad_u, laplace_u = compute_derivatives(model, x)
#f = f_fun(x)
#pde_loss = ((laplace_u.squeeze() - f)**2).mean()