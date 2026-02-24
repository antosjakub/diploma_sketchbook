import torch
import json

def json_dump(file_path, d):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4)

def json_load(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        d = json.load(f)
    return d

class HeatEquation:
    def __init__(self, d, alpha=0.01, a=None):
        self.d = d
        #self.a = torch.pi * torch.ones(d) if a is None else a
        self.a = a
        self.a_2 = (self.a**2).sum()
        self.alpha = alpha
        self.has_weak_form = True
    def dump_pde_params(self, file_path) -> None:
        pde_params = {"alpha": self.alpha, "a": list(map(lambda x: float(x), self.a))}
        json_dump(file_path, pde_params)
    def load_pde_params(self, file_path) -> None:
        pde_params = json_load(file_path)
        pde_params["a"] = torch.tensor(pde_params["a"])
        self.__init__(self.d, **pde_params)
    def u_spatial(self, x):
        # x.shape = (batch size, spatial dims)
        # return shape = (batch size,)
        return torch.prod(torch.sin(self.a*x), dim=1)
    def u_analytic(self, X):
        # X.shape = (batch size, spatial+time dims)
        bs, D = X.shape
        d = D-1
        x = X[:,:-1]
        t = X[:,-1]
        # u = sin(a1 x1) ... sin(an xn) * e^(-alpha*(a1^2+...+an^2) t)
        u_space = self.u_spatial(x)
        u_time = torch.exp(- self.alpha * self.a_2 * t)
        # return shape = (batch size, 1)
        return (u_space * u_time).unsqueeze(dim=1)
    def u_IC(self, x):
        # x.shape = (batch size, spatial dims)
        # return shape = (batch size, 1)
        return self.u_spatial(x).unsqueeze(dim=1)
    # --- RESIDUALS ---
    # X.shape = (bs, D)
    # u.shape = (bs, 1)
    # grad_u.shape = (bs, D)
    # sp_u_laplace.shape = (bs, 1)
    # return shape = (bs, 1)
    def pde_residual(self, X, u, grad_u, sp_laplace_u):
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        residual = u_t - self.alpha * sp_laplace_u
        return residual
    def pde_residual_weak_form(self, X, u, grad_u, sp_laplace_u):
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        u_grad_2 = torch.sum(grad_u**2, dim=1).unsqueeze(dim=1)
        residual = u_t * u + self.alpha * u_grad_2
        return residual
    def bc_residual(self, X, u):
        return u - self.u_analytic(X)
    def ic_residual(self, X, u):
        return u - self.u_IC(X[:,:-1])

