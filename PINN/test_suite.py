import torch


alpha = 0.01

def u_analytic(X):
    # X.shape = (batch size, spatial+time dims)
    bs, D = X.shape
    d = D-1
    alpha = 0.01
    u_space = torch.prod(torch.sin(torch.pi * X[:,:-1]), dim=1)
    u_time = torch.exp(- d * alpha * torch.pi**2 * X[:,-1])
    # return shape = (batch size, 1)
    return (u_space * u_time).unsqueeze(dim=1)


def u_0(x):
    # x.shape = (batch size, spatial dims)
    # return shape = (batch size, 1)
    return torch.prod(torch.sin(torch.pi * x), dim=1).unsqueeze(dim=1)

def u_D(X):
    # X.shape = (batch size, spatial+time dims)
    # return shape = (batch size, 1)
    return torch.zeros(X.shape[0]).unsqueeze(dim=1)


def pde_residual(X, u, grad_u, sp_laplace_u):
    """
    X.shape = (bs, D)
    u.shape = (bs, 1)
    grad_u.shape = (bs, D)
    sp_u_laplace.shape = (bs, 1)
    return shape = (bs, 1)
    """
    u_t = grad_u[:,-1].unsqueeze(dim=1)
    residual = u_t - alpha * sp_laplace_u
    return residual

