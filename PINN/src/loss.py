import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext


def sdgd_loss(model, X, pde_residual, pde_sgsd_single_term_residual, num_dims_to_use: int):
    # sample some indices
    bs,D = X.shape
    d = D-1
    I = torch.randperm(d)[:num_dims_to_use]
    X.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X)
    R = pde_loss(model, X, pde_residual, compute_laplace=True).detach()
    R_stoch = torch.zeros((bs,1))
    for i in I:
        #Ri = 1/d * grad_u[:,-1:] - alpha * spatial_laplace_u[i] + v[i] * grad_u[:,i:i+1] + 1/d * b * u
        Ri = pde_sgsd_single_term_residual(X, u, grad_u, spatial_laplace_u, i)
        R_stoch += Ri
    # total loss
    loss = 2 * R * R_stoch
    loss = torch.mean(loss)
    # scalar
    return loss

def pde_loss(model, X_in, pde_residual, compute_laplace=True):
    """
    X_in: (batch_size, d+1) tensor
    """
    #X_in.requires_grad = True
    u, grad_u, spatial_laplace_u = compute_derivatives(model, X_in, compute_laplace=compute_laplace)
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

class AdaptiveWeights:
    def __init__(self, n_terms, momentum=0.9, device='cpu'):
        self.w = torch.ones(n_terms, device=device)
        self.momentum = momentum

    def update(self, grad_norms):
        """grad_norms: list of grad_\theta(loss) terms"""
        # w_i = w_i * mean(losses) / val_loss_i
        norms = torch.stack([g.detach() for g in grad_norms])
        mean_norm = norms.mean()
        target = mean_norm / (norms + 1e-8)
        # exponential moving average
        self.w = self.momentum * self.w + (1 - self.momentum) * target
        return self.w


def compute_grad_norm(loss, model):
    """L2 norm of gradients of `loss` w.r.t. model parameters."""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, allow_unused=True).detach()
    total = sum(g.norm() ** 2 for g in grads if g is not None)
    return total.sqrt()