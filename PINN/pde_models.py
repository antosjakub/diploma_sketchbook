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
        laplace = sp_laplace_u.sum(dim=1).unsqueeze(dim=1)
        residual = u_t - self.alpha * laplace 
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
    def pde_sgsd_single_term_residual(self, X, u, grad_u, spatial_laplace_u, i: int):
        return 1/self.d * grad_u[:,-1:] - self.alpha * spatial_laplace_u[i]

class FokkerPlanck:
    def __init__(self):
        pass


class TravellingGaussPacket:
    def __init__(self, d):
        self.alpha = 1
        self.a = 1.0 * torch.ones(d)
        self.b = 0.25 * 2*(torch.ones(d)-0.5)
        self.c = 0.5 + 0.1 * 2*(torch.ones(d)-0.5)
        self.delta = 1
        self.v = 1.0 * torch.ones(d)
        self.w = 1

    def __u_analytic(self, z):
        return torch.exp(-self.alpha*(z**2).sum(dim=-1))

    def u_analytic(self, X):
        z = self.a * X[:-1] - self.b + self.c * X[-1:]
        return self.__u_analytic(z)

    def __f_inner(self, z):
        coeff = self.v * self.a + self.c
        sum1  = (coeff * z).sum(dim=-1)
        sum2  = (self.a**2 * (2 * self.alpha * z**2 - 1)).sum(dim=-1)
        return (-2 * self.alpha * (sum1 + self.delta * sum2) + self.w).unsqueeze(-1)

    def f(self, X):
        z = self.a * X[:-1] - self.b + self.c * X[-1:]
        return self.__f_inner(z)

class TravellingGaussPacket_v2:
    def __init__(self, d, alpha=None, beta=None, gamma=None, a=None, b=None, c=None):
        self.d = d
        # t1
        self.alpha = alpha if alpha is not None  else 7.4
        self.a =     a     if a     is not None  else 0.9 + 0.2*torch.rand(d)
        self.b =     b     if b     is not None  else 0.4 + 0.2*torch.rand(d)
        self.c =     c     if c     is not  None else -0.3 + 0.6*torch.rand(d)
        # t2
        self.beta =  beta  if beta  is not None else 0.2
        # t3
        self.gamma = gamma if gamma is not None else 1.9*torch.pi
        # pde
        self.delta = 1.0
        self.v = -1.0 * self.c / self.a
        self.w = -2.0*self.delta*self.alpha * torch.sum(self.a**2)

    def u_analytic(self, X):
        z = self.a * X[:,:-1] - self.b + self.c * X[:,-1:]
        return (
            torch.exp(-self.alpha*(z**2).sum(dim=-1) - self.beta*X[:,-1])
            * torch.cos(self.gamma*X[:,-1])
        ).unsqueeze(dim=1)
    
    def u_IC(self, X):
        z = self.a * X - self.b
        return (
            torch.exp(-self.alpha*(z**2).sum(dim=-1))
        ).unsqueeze(dim=1)

    def f(self, X):
        z = self.a * X[:,:-1] - self.b + self.c * X[:,-1:]
        f_sim_inner  = -4.0*self.alpha**2*self.delta*((self.a * z)**2).sum(dim=-1)
        return ((
                (f_sim_inner - self.beta)*torch.cos(self.gamma*X[:,-1])
                - self.gamma * torch.sin(self.gamma*X[:,-1])
            ) * torch.exp(-self.alpha*(z**2).sum(dim=-1) - self.beta*X[:,-1])
        ).unsqueeze(dim=1)
    
    def pde_residual(self, X, u, grad_u, sp_laplace_u):
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        laplace = sp_laplace_u.sum(dim=1).unsqueeze(dim=1)
        residual = u_t - self.delta * laplace + (self.v * grad_u[:,:-1]).sum(dim=1).unsqueeze(dim=1) + self.w * u - self.f(X)
        return residual
    def bc_residual(self, X, u):
        return u - self.u_analytic(X)
    def ic_residual(self, X, u):
        return u - self.u_IC(X[:,:-1])

    def dump_pde_params(self, file_path) -> None:
        pde_params = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "a": list(map(lambda x: float(x), self.a)),
            "b": list(map(lambda x: float(x), self.b)),
            "c": list(map(lambda x: float(x), self.c))
        }
        json_dump(file_path, pde_params)
    def load_pde_params(self, file_path) -> None:
        pde_params = json_load(file_path)
        pde_params["a"] = torch.tensor(pde_params["a"])
        pde_params["b"] = torch.tensor(pde_params["b"])
        pde_params["c"] = torch.tensor(pde_params["c"])
        self.__init__(self.d, **pde_params)