import math
import torch
import utility
import derivatives


class PDEModel:
    def __init__(self):
        raise NotImplementedError
    def get_pde_metadata(self):
        raise NotImplementedError
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})
    def __load_pde_metadata(self, pde_metadata) -> None:
        pde_class = pde_metadata["pde_class"]
        assert pde_class == type(self).__name__, f"ERROR: The given .json file specifies parameters for '{pde_class}', but this class is of type '{type(self).__name__}'."
        return pde_metadata["params"]
    def load_pde_metadata(self, pde_metadata) -> None:
        raise NotImplementedError
    def pde_residual(self, X, model, precomputed):
        raise NotImplementedError
    def bc_residual(self, X, model, precomputed):
        raise NotImplementedError
    def ic_residual(self, X, model, precomputed):
        raise NotImplementedError
    def pde_loss(self, X, model, precomputed):
        return torch.mean(self.pde_residual(X, model, precomputed)**2)
    def bc_loss(self, X, model, precomputed):
        return torch.mean(self.bc_residual(X, model, precomputed)**2)
    def ic_loss(self, X, model, precomputed):
        return torch.mean(self.ic_residual(X, model, precomputed)**2)
    def precompute(self, X_pde, X_bc, X_ic):
        return {
            "pde": {},
            "bc": {},
            "ic": {},
        }


class HeatEquation(PDEModel):
    def __init__(self, d, alpha_bar=None, k=None):
        self.d = d
        #self.a = torch.pi * torch.ones(d) if a is None else a
        self.alpha_bar = alpha_bar if alpha_bar is not None else 0.01
        self.alpha = self.alpha_bar / float(d)
    def get_pde_metadata(self):
        return {
            "alpha": self.alpha,
        }
    def load_pde_metadata(self, pde_metadata) -> None:
        pde_params = self.__load_pde_metadata(pde_metadata)
        self.__init__(self.d, **pde_params)

    #def u_spatial(self, x):
    #    return torch.prod(torch.sin(self.k*x), dim=1)
    def u_analytic(self, X, t0=False):
        # X.shape = (batch size, spatial+time dims)
        # u = sin(k1 x1) ... sin(kn xn) * e^(-alpha*(k1^2+...+kn^2) t)
        #return (
        #    self.u_spatial(X[:,:-1]) * torch.exp(- self.alpha * self.k_2 * X[:,-1])
        #).unsqueeze(dim=1)
        if t0 == True:
            t = torch.zeros_like(X[:,-1:])
            x = torch.pi*X
        else:
            t = X[:,-1:]
            x = torch.pi*X[:,:-1]

        gamma = self.alpha_bar * torch.pi**2
        out = torch.exp(-gamma*t) * torch.prod(torch.sin(x), dim=1, keepdim=True)
        a = 2.0
        b = 0.67 / float(self.d**0.5)
        delta = self.alpha_bar * (1+(a**2-1)/float(self.d)) * torch.pi**2
        out2 = torch.zeros_like(out)
        for i in range(self.d):
            o = torch.sin(x)
            o[:,i:(i+1)] = torch.sin(a*x[:,i:(i+1)])
            out2 += torch.prod(o, dim=1, keepdim=True)
        out2 *= b*torch.exp(-delta*t)
        return out + out2
        #out = c*torch.exp(-0.01*t) * torch.sin(x[:,0:1])*torch.sin(x[:,1:2])
        #out += b*torch.exp(-t) * torch.sin(x[:,0:1])*torch.sin(a*x[:,1:2])
        #out += b*torch.exp(-t) * torch.sin(a*x[:,0:1])*torch.sin(x[:,1:2])
        #return out

    def u_bc(self, X):
        return self.u_analytic(X)
    def u_ic(self, x):
        return self.u_analytic(x, t0=True)

    def precompute(self, X_pde, X_bc, X_ic):
        return {
            "pde": {},
            "bc": {
                "bc": self.u_bc(X_bc) if X_bc is not None else None,
            },
            "ic": {
                "ic": self.u_ic(X_ic[:,:-1]) if X_ic is not None else None,
            },
        }
    # --- RESIDUALS ---
    # X.shape = (bs, D)
    # u.shape = (bs, 1)
    # grad_u.shape = (bs, D)
    # sp_u_laplace.shape = (bs, 1)
    # return shape = (bs, 1)
    def pde_residual_base(self, X, u, grad_u, spatial_laplace_u, precomputed_pde=None):
        return grad_u[:,-1:] - self.alpha * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1)
    def pde_residual(self, X, model, precomputed_pde=None):
        X = X.detach().requires_grad_(True)
        _, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        #_, grad_u, spatial_laplace_u = derivatives.compute_derivatives_fd(model, X)
        return self.pde_residual_base(None, None, grad_u, spatial_laplace_u, precomputed_pde)
    def bc_residual(self, X, model, precomputed_bc):
        return model(X) - precomputed_bc["bc"]
    def ic_residual(self, X, model, precomputed_ic):
        return model(X) - precomputed_ic["ic"]

    def pde_residual_weak_form(self, X, model):
        X = X.detach().requires_grad_(True)
        u, grad_u, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
        u_t = grad_u[:,-1:]
        u_grad_2 = torch.sum(grad_u**2, dim=1).unsqueeze(dim=1)
        residual = u_t * u + self.alpha * u_grad_2
        return residual
    def pde_sgsd_single_term_residual(self, X, u, grad_u, spatial_laplace_u, i: int):
        return 1/self.d * grad_u[:,-1:] - self.alpha * spatial_laplace_u[:,i:i+1]
    def pde_sgsd_single_term_residual_v1(self, X, u, grad_u, spatial_laplace_u, i: int):
        return grad_u[:,-1:]
    def pde_sgsd_single_term_residual_v2(self, X, u, grad_u, spatial_laplace_u, i: int):
        return -1 * self.alpha * spatial_laplace_u[i:i+1]

# u =   sin(k1 x1) ... sin(kn xn) * e^(-alpha*(k1^2+...+kn^2) t) * cos(beta*t)
#   =   sin(k1 x1) ... sin(kn xn) * e^(-alpha*|k|^2 t) * cos(beta*t)
# u_t = sin(k1 x1) ... sin(kn xn) * ( -alpha*|k|^2 e^. cos() - beta e^. sin() )
#     = - alpha*|k|^2 u - beta tan() u
# laplace_u = - |k|^2 u
# u_t - alpha laplace_u = beta tan() u = f
# f(x,t) = beta * sin(k1 x1) ... sin(kn xn) * e^(-alpha*|k|^2 t) * sin(beta*t)
class HeatEquationWithSource(PDEModel):
    def __init__(self, d, alpha=None, k=None, beta=None):
        self.d = d
        self.k     = k     if k     is not None else torch.pi * torch.ones(d)
        self.alpha = alpha if alpha is not None else 0.01
        self.beta  = beta  if beta  is not None else 2.0 * torch.pi
        self.k_2 = (self.k**2).sum()
    def get_pde_metadata(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "k": list(map(lambda x: float(x), self.k)),
        }

    def load_pde_metadata(self, pde_metadata) -> None:
        pde_params = self.__load_pde_metadata(pde_metadata)
        pde_params["k"] = torch.tensor(pde_params["k"])
        self.__init__(self.d, **pde_params)

    def u_spatial(self, x):
        return torch.prod(torch.sin(self.k*x), dim=1)
    def u_analytic(self, X):
        bs, D = X.shape
        d = D-1
        x = X[:,:-1]
        t = X[:,-1]
        u_space = self.u_spatial(x)
        u_time = torch.exp(- self.alpha * self.k_2 * t) * torch.cos(self.beta * t)
        return (u_space * u_time).unsqueeze(dim=1)
    def f(self, X):
        bs, D = X.shape
        d = D-1
        x = X[:,:-1]
        t = X[:,-1]
        u_space = self.u_spatial(x)
        u_time = - 1 * self.beta * torch.exp(- self.alpha * self.k_2 * t) * torch.sin(self.beta * t)
        return (u_space * u_time).unsqueeze(dim=1)
    def u_ic(self, x):
        return self.u_spatial(x).unsqueeze(dim=1)
    def u_bc(self, X):
        return self.u_analytic(X)

    def precompute(self, X_pde, X_bc, X_ic):
        return {
            "pde": {
                "f": self.f(X_pde),
            },
            "bc": {
                "u": self.u_bc(X_bc),
            },
            "ic": {
                "u": self.u_ic(X_ic[:,:-1]),
            },
        }

    # --- RESIDUALS ---
    # X.shape = (bs, D)
    # u.shape = (bs, 1)
    # grad_u.shape = (bs, D)
    # sp_u_laplace.shape = (bs, 1)
    # return shape = (bs, 1)
    def pde_residual_base(self, X, u, grad_u, spatial_laplace_u, precomputed_pde):
        return grad_u[:,-1:] - self.alpha * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1) - precomputed_pde["f"]
    def pde_residual(self, X, model, precomputed_pde):
        X = X.detach().requires_grad_(True)
        _, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        return self.pde_residual_base(None, None, grad_u, spatial_laplace_u, precomputed_pde)
    def bc_residual(self, X, model, precomputed_bc):
        return model(X) - precomputed_bc["u"]
    def ic_residual(self, X, model, precomputed_ic):
        return model(X) - precomputed_ic["u"]
    #def pde_residual_weak_form(self, X, model):
    #    u, grad_u, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
    #    u_t = grad_u[:,-1].unsqueeze(dim=1)
    #    residual = u_t * u + self.alpha * torch.sum(grad_u**2, dim=1).unsqueeze(dim=1)
    #    return residual
    def pde_sgsd_single_term_residual_v1(self, X, u, grad_u, spatial_laplace_u, i: int):
        return grad_u[:,-1:] - self.f(X)
    def pde_sgsd_single_term_residual_v2(self, X, u, grad_u, spatial_laplace_u, i: int):
        return -1 * self.alpha * spatial_laplace_u[i:i+1]
    def pde_sgsd_single_term_residual(self, X, u, grad_u, spatial_laplace_u, i: int):
        return 1/self.d * grad_u[:,-1:] - self.alpha * spatial_laplace_u[:,i:i+1] - 1/self.d * self.f(X)



class TravellingGaussPacket(PDEModel):
    def __init__(self, d, alpha=None, beta=None, gamma=None, a=None, b=None, c=None):
        self.d = d
        # t1
        self.alpha = alpha if alpha is not None  else 7.4
        self.a =     a     if a     is not None  else 0.8 + 0.4*torch.rand(d)
        self.b =     b     if b     is not None  else 0.4 + 0.2*torch.rand(d)
        self.c =     c     if c     is not  None else -0.3 + 0.6*torch.rand(d)
        # t2
        self.beta =  beta  if beta  is not None else 0.2
        # t3
        self.gamma = gamma if gamma is not None else 1.9*torch.pi
        # pde
        self.delta = 1.0
        self.v = -1.0 * self.c / self.a
        self.w = -2.0 * self.delta*self.alpha * torch.sum(self.a**2)

    def u_analytic(self, X):
        z = self.a * X[:,:-1] - self.b + self.c * X[:,-1:]
        return (
            torch.exp(-self.alpha*(z**2).sum(dim=-1) - self.beta*X[:,-1])
            * torch.cos(self.gamma*X[:,-1])
        ).unsqueeze(dim=1)
    def u_bc(self, X):
        return self.u_analytic(X)
    def u_ic(self, x):
        z = self.a * x - self.b
        return (
            torch.exp(-self.alpha*(z**2).sum(dim=-1))
        ).unsqueeze(dim=1)

    def f(self, X):
        z = self.a * X[:,:-1] - self.b + self.c * X[:,-1:]
        f_sim_inner = -4.0*self.alpha**2*self.delta*((self.a * z)**2).sum(dim=-1)
        return ((
                (f_sim_inner - self.beta)*torch.cos(self.gamma*X[:,-1])
                - self.gamma * torch.sin(self.gamma*X[:,-1])
            ) * torch.exp(-self.alpha*(z**2).sum(dim=-1) - self.beta*X[:,-1])
        ).unsqueeze(dim=1)

    def precompute(self, X_pde, X_bc, X_ic):
        return {
            "pde": {
                "f": self.f(X_pde),
            },
            "bc": {
                "u": self.u_bc(X_bc),
            },
            "ic": {
                "u": self.u_ic(X_ic[:,:-1]),
            },
        }
    
    def pde_residual_base(self, X, u, grad_u, spatial_laplace_u, precomputed_pde):
        return grad_u[:,-1:] - self.delta * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1) + (self.v * grad_u[:,:-1]).sum(dim=1).unsqueeze(dim=1) + self.w * u - precomputed_pde["f"]
    def pde_residual(self, X, model, precomputed_pde):
        X = X.detach().requires_grad_(True)
        u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        return self.pde_residual_base(None, u, grad_u, spatial_laplace_u, precomputed_pde)
    def bc_residual(self, X, model, precomputed_bc):
        return model(X) - precomputed_bc["u"]
    def ic_residual(self, X, model, precomputed_ic):
        return model(X) - precomputed_ic["u"]

    def pde_sgsd_single_term_residual(self, X, u, grad_u, spatial_laplace_u, i: int):
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        return 1/self.d * u_t - self.delta * spatial_laplace_u[:,i:i+1] + (self.v[i] * grad_u[:,i:i+1]) + 1/self.d * self.w * u - 1/self.d * self.f(X)

    def get_pde_metadata(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "a": list(map(lambda x: float(x), self.a)),
            "b": list(map(lambda x: float(x), self.b)),
            "c": list(map(lambda x: float(x), self.c))
        }

    def load_pde_metadata(self, pde_metadata) -> None:
        pde_params = self.__load_pde_metadata(pde_metadata)
        pde_params["a"] = torch.tensor(pde_params["a"])
        pde_params["b"] = torch.tensor(pde_params["b"])
        pde_params["c"] = torch.tensor(pde_params["c"])
        self.__init__(self.d, **pde_params)
