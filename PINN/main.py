import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, d, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

def compute_derivatives(model, x):
    """Compute u, grad u, and laplace u"""
    u = model(x)
    
    # Gradient
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Laplacian
    laplace_u = 0
    for i in range(x.shape[1]):
        grad = torch.autograd.grad(
            outputs=grad_u[:, i].sum(),
            inputs=x,
            create_graph=True,
            allow_unused=True
        )[0][:, i:i+1]
        laplace_u += grad
    
    return u, grad_u, laplace_u


d = 2

def u_fun(x):
    # takes in a matrix, return a vector
    return torch.prod(torch.sin(torch.pi * x), dim=1)
def f_fun(x):
    # takes in a matrix, return a vector
    return -d * torch.pi**2 * torch.prod(torch.sin(torch.pi * x), dim=1)

# Example usage for Poisson equation: laplace u = f
#model = PINN(d)
#x = torch.randn(100, d, requires_grad=True)
#u, grad_u, laplace_u = compute_derivatives(model, x)
#f = f_fun(x)
#pde_loss = ((laplace_u.squeeze() - f)**2).mean()


def train_pinn(model, epochs=5000, lr=1e-3):
    """Train the PINN"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    history = {'loss': [], 'l2_error': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Loss
        x = torch.randn(100, d, requires_grad=True)
        u, grad_u, laplace_u = compute_derivatives(model, x)
        f = f_fun(x)
        loss = torch.mean((laplace_u - f) ** 2)
        # Update
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Log progress
        if epoch % (epochs//10) == 0:
            with torch.no_grad():
                # Compute L2 error on test points
                x = torch.rand(1000, d)
                u_pred = model(x)
                u_exact = u_fun(x)
                l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
                
                history['loss'].append(loss.item())
                history['l2_error'].append(l2_error)
                
                print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | L2 Error: {l2_error:.6f}")
    
    return history

if __name__ == "__main__":
    model = PINN(d)
    x = torch.rand(1000, d)
    compute_derivatives(model, )
    #history = train_pinn(model, epochs=1000)
    #torch.save(model, 'model.pth')