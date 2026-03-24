import torch
import derivatives


def sdgd_loss(model, X, pde_model, num_dims_to_use: int):
    # sample some indices
    bs,D = X.shape
    d = D-1
    I = torch.randperm(d)[:num_dims_to_use]
    X.requires_grad = True
    u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
    R = pde_model.pde_residual(X, u, grad_u, spatial_laplace_u).detach()
    R_stoch = torch.zeros((bs,1))
    for i in I:
        #Ri = 1/d * grad_u[:,-1:] - alpha * spatial_laplace_u[i] + v[i] * grad_u[:,i:i+1] + 1/d * b * u
        Ri = pde_model.pde_sgsd_single_term_residual(X, u, grad_u, spatial_laplace_u, i)
        R_stoch += Ri
    # total loss
    loss = 2 * R * d/num_dims_to_use * R_stoch
    loss = torch.mean(loss)
    # scalar
    return loss

def pde_loss(model, X_in, pde_residual, compute_laplace=True):
    """
    X_in: (batch_size, d+1) tensor
    """
    #X_in.requires_grad = True
    u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X_in, compute_laplace=compute_laplace)
    residual = pde_residual(X_in, u, grad_u, spatial_laplace_u)
    return torch.mean(residual**2)

def initial_condition_loss(model, X_ic, u_target):
    """
    X_ic: (batch_size, d+1) tensor with t = 0
    IC: u(x,0) = u_IC(x)
    """
    return torch.mean((model(X_ic)-u_target)**2)

def boundary_condition_loss(model, X_bc, u_target):
    """
    X_bc: (batch_size, d+1) tensor with points on boundary
    BC: u(x,t) = u_BC(x,t)
    """
    return torch.mean((model(X_bc)-u_target)**2)



class ConstantWeights:
    def __init__(self, weights):
        self.weights = weights

    def weight_loss(self, losses):
        assert len(losses) == len(self.weights), "Number of losses and weights must match"
        return sum(self.weights[i] * losses[i] for i in range(len(losses)))
    

class AdaptiveWeights(ConstantWeights):
    def __init__(self, weights, momentum=0.9, device='cpu'):
        self.weights = weights
        self.momentum = momentum

    def __compute_grad_norm(self, loss, model):
        """L2 norm of gradients of `loss` w.r.t. model parameters."""
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, allow_unused=True)
        total = sum(g.norm() ** 2 for g in grads if g is not None)
        return total.sqrt()

    def update(self, losses, model):
        """grad_norms: list of grad_\theta(loss) terms"""
        grad_norms = torch.zeros_like(self.weights)
        for i in range(len(losses)):
            grad_norms[i] = self.__compute_grad_norm(losses[i], model)
        weights_new = torch.ones_like(self.weights) * grad_norms.sum() + 1e-8
        weights_new /= grad_norms + 1e-8
        # exponential moving average
        self.weights = self.momentum * self.weights + (1 - self.momentum) * weights_new
        print(self.weights)
