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




#def compute_derivatives(model, x):
def compute_derivatives(u, x):
    """Compute u, grad u, and laplace u"""
    
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
