import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext


def compute_derivatives(model, X, compute_laplace=True):
    """
    Compute u, grad u, and laplace u
    X: (batch_size, D) where D = d + 1 (spatial dims + time)
    """
    u = model(X)
    bs, D = X.shape


    with record_function("grad_u"):
        # Gradient - spatial & temporatal
        grad_u = torch.autograd.grad(
            inputs=X,
            outputs=u,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

    if compute_laplace: 
        with record_function("laplace_u"):
            # Laplacian - spatial only
            spatial_laplace_u = []
            for i in range(D-1):
                hess_row = torch.autograd.grad(
                    inputs=X,
                    outputs=grad_u[:,i].sum(),
                    grad_outputs=torch.tensor(1.0),
                    create_graph=True,
                    retain_graph=True
                )[0]
                spatial_laplace_u.append(hess_row[:,i:i+1])
        spatial_laplace_u = torch.cat(spatial_laplace_u, dim=1)
    else:
        spatial_laplace_u = None

    # shapes: bs x 1, bs x D, bs x D-1
    return u, grad_u, spatial_laplace_u


def compute_u_grad_u(model, X):
    u, vjp_fn = torch.func.vjp(model, X)
    grad_u = vjp_fn(torch.ones_like(u))
    return u, grad_u[0]

# entries contain either -1 or 1
sample_Rademacherlambda = lambda n1,n2: 2.0*torch.randint(0,2,(n1,n2),dtype=torch.float)-1.0,

from torch.func import jacrev, jvp
def hutchinson_trace_estimation(model, X):

    def value_point(point):
        return model(point.unsqueeze(dim=0)).squeeze()

    def laplace_hutchinson_point(point):
        num_vectors = 100
        vectors = sample_Rademacherlambda(num_vectors, len(point))
        grad = lambda point: jacrev(value_point)(point)
        jvp_f = lambda v: torch.dot(v, jvp(grad, (point,), (v,))[1])
        return torch.sum(torch.vmap(jvp_f)(vectors))/num_vectors

    def laplace_hutchinson(points):
        return torch.vmap(laplace_hutchinson_point, randomness="same")(points)

    # N x 1
    return laplace_hutchinson(X).unsqueeze(dim=1)

def compute_derivatives_hte(model, X):
    u, grad_u = compute_u_grad_u(model, X)
    spatial_laplace_u = hutchinson_trace_estimation(model, X)
    return u, grad_u, spatial_laplace_u

