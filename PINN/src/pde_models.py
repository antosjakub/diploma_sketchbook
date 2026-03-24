import torch
import utility
import derivatives


class PDEModel:
    def __init__(self):
        pass
    def get_pde_metadata(self):
        pass
    def dump_pde_metadata(self, file_path) -> None:
        pass
    def load_pde_metadata(self, pde_metadata) -> None:
        pass
    def pde_residual(self, X, model):
        pass
    def bc_residual(self, X, model):
        pass
    def ic_residual(self, X, model):
        pass
    def pde_loss(self, X, model):
        return torch.mean(self.pde_residual(X,model)**2)
    def bc_loss(self, X, model):
        return torch.mean(self.bc_residual(X,model)**2)
    def ic_loss(self, X, model):
        return torch.mean(self.ic_residual(X,model)**2)


class HeatEquation(PDEModel):
    def __init__(self, d, alpha=None, k=None):
        self.d = d
        #self.a = torch.pi * torch.ones(d) if a is None else a
        self.k     = k     if k     is not None else torch.pi * torch.ones(d)
        self.alpha = alpha if alpha is not None else 0.01
        self.k_2 = (self.k**2).sum()
    def get_pde_metadata(self):
        return {
            "alpha": self.alpha,
            "k": list(map(lambda x: float(x), self.k)),
        }
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})
    def load_pde_metadata(self, pde_metadata) -> None:
        pde_class = pde_metadata["pde_class"]
        assert pde_class == type(self).__name__, f"ERROR: The given .json file specifies parameters for '{pde_class}', but this class is of type '{type(self).__name__}'."
        pde_params = pde_metadata["params"]
        pde_params["k"] = torch.tensor(pde_params["k"])
        self.__init__(self.d, **pde_params)
    def u_spatial(self, x):
        return torch.prod(torch.sin(self.k*x), dim=1)
    def u_analytic(self, X):
        # X.shape = (batch size, spatial+time dims)
        # u = sin(k1 x1) ... sin(kn xn) * e^(-alpha*(k1^2+...+kn^2) t)
        return (
            self.u_spatial(X[:,:-1]) * torch.exp(- self.alpha * self.k_2 * X[:,-1])
        ).unsqueeze(dim=1)
    def u_bc(self, x):
        return self.u_analytic(x)
    def u_ic(self, x):
        return self.u_spatial(x).unsqueeze(dim=1)
    # --- RESIDUALS ---
    # X.shape = (bs, D)
    # u.shape = (bs, 1)
    # grad_u.shape = (bs, D)
    # sp_u_laplace.shape = (bs, 1)
    # return shape = (bs, 1)
    def pde_residual(self, X, model):
        X = X.detach().requires_grad_(True)
        _, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        return grad_u[:,-1:] - self.alpha * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1)
    def pde_residual_weak_form(self, X, model):
        X = X.detach().requires_grad_(True)
        u, grad_u, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
        u_t = grad_u[:,-1:]
        u_grad_2 = torch.sum(grad_u**2, dim=1).unsqueeze(dim=1)
        residual = u_t * u + self.alpha * u_grad_2
        return residual
    def bc_residual(self, X, model):
        return model(X) - self.u_analytic(X)
    def ic_residual(self, X, model):
        return model(X) - self.u_ic(X[:,:-1])
    def pde_sgsd_single_term_residual(self, X, u, grad_u, spatial_laplace_u, i: int):
        return 1/self.d * grad_u[:,-1:] - self.alpha * spatial_laplace_u[i]


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
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})
    def load_pde_metadata(self, pde_metadata) -> None:
        pde_class = pde_metadata["pde_class"]
        assert pde_class == type(self).__name__, f"ERROR: The given .json file specifies parameters for '{pde_class}', but this class is of type '{type(self).__name__}'."
        pde_params = pde_metadata["params"]
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
    # --- RESIDUALS ---
    # X.shape = (bs, D)
    # u.shape = (bs, 1)
    # grad_u.shape = (bs, D)
    # sp_u_laplace.shape = (bs, 1)
    # return shape = (bs, 1)
    def pde_residual(self, X, model):
        X = X.detach().requires_grad_(True)
        u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        return grad_u[:,-1:] - self.alpha * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1) - self.f(X)
    def bc_residual(self, X, model):
        return model(X) - self.u_bc(X)
    def ic_residual(self, X, model):
        return model(X) - self.u_ic(X[:,:-1])
    #def pde_residual_weak_form(self, X, model):
    #    u, grad_u, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
    #    u_t = grad_u[:,-1].unsqueeze(dim=1)
    #    residual = u_t * u + self.alpha * torch.sum(grad_u**2, dim=1).unsqueeze(dim=1)
    #    return residual
    #def pde_sgsd_single_term_residual(self, X, model, i: int):
    #    X = X.detach().requires_grad_(True)
    #    u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
    #    return 1/self.d * grad_u[:,-1:] - self.alpha * spatial_laplace_u[i]

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


class TravellingGaussPacket_v2(PDEModel):
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
    def u_bc(self, X):
        return self.u_analytic(X)
    def u_ic(self, X):
        z = self.a * X[:,:-1] - self.b
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
    
    def pde_residual(self, X, model):
        X = X.detach().requires_grad_(True)
        u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        laplace = spatial_laplace_u.sum(dim=1).unsqueeze(dim=1)
        return u_t - self.delta * laplace + (self.v * grad_u[:,:-1]).sum(dim=1).unsqueeze(dim=1) + self.w * u - self.f(X)
    def bc_residual(self, X, model):
        return model(X) - self.u_analytic(X)
    def ic_residual(self, X, model):
        return model(X) - self.u_ic(X[:,:-1])

    def get_pde_metadata(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "a": list(map(lambda x: float(x), self.a)),
            "b": list(map(lambda x: float(x), self.b)),
            "c": list(map(lambda x: float(x), self.c))
        }
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})
    def load_pde_metadata(self, pde_metadata) -> None:
        pde_class = pde_metadata["pde_class"]
        assert pde_class == type(self).__name__, f"ERROR: The given .json file specifies parameters for '{pde_class}', but this class is of type '{type(self).__name__}'."
        pde_params = pde_metadata["params"]
        pde_params["a"] = torch.tensor(pde_params["a"])
        pde_params["b"] = torch.tensor(pde_params["b"])
        pde_params["c"] = torch.tensor(pde_params["c"])
        self.__init__(self.d, **pde_params)


class FokkerPlanckLJ:
    def __init__(self, n_atoms, d, r0=None, epsilon=None, xi=None, D=None):
        self.n_atoms = n_atoms
        self.d = d
        self.r0 =           r0 if r0      is not None else 1.0
        self.epsilon = epsilon if epsilon is not None else 1.0
        self.xi =           xi if xi      is not None else 1.0
        self.D =             D if D       is not None else 0.1

    def pde_residual(self, X, u, grad_u, sp_laplace_u, other):
        u_t = grad_u[:,-1].unsqueeze(dim=1)
        laplace = sp_laplace_u.sum(dim=1).unsqueeze(dim=1)
        lj_grad, lj_laplace = other["lj_grad"], other["lj_laplace"]
        residual = u_t - self.D * laplace + (lj_grad * grad_u[:,:-1]).sum(dim=1).unsqueeze(dim=1)/self.xi + u * lj_laplace/self.xi
        return residual
    
    def precompute_grad_and_laplace(self, Y):
        """
        Y: (B, n_atoms * d)
        returns:
            grad:    (B, n_atoms, d)
            laplace: (B, n_atoms, d)
        """
        B = Y.shape[0]
        X = Y.view(B, self.n_atoms, self.d)                      # (B, n_atoms, d)

        # pairwise differences: dX[b, k, j, c] = X[b, k, c] - X[b, j, c]
        dX = X[:, :, None, :] - X[:, None, :, :]      # (B, n_atoms, n_atoms, d)

        # pairwise distances r[b, k, j]
        r = torch.linalg.vector_norm(dX, dim=-1)      # (B, n_atoms, n_atoms)

        # mask self-interactions (k == j)
        eye = torch.eye(self.n_atoms, dtype=torch.bool, device=X.device).expand(B, self.n_atoms, self.n_atoms)
        r = r.masked_fill(eye, float('inf'))          # (B, n_atoms, n_atoms)

        # shared powers of r
        r_inv8  = r**(-8)                             # (B, n_atoms, n_atoms)
        r_inv14 = r**(-14)                            # (B, n_atoms, n_atoms)

        # ---------------------------
        # Gradient of LJ potential
        # ---------------------------
        coeff_grad = -12 * self.r0**12 * r_inv14 + 6 * self.r0**6 * r_inv8     # (B, n_atoms, n_atoms)

        grad_pair = coeff_grad[..., None] * dX                       # (B, n_atoms, n_atoms, d)
        grad = grad_pair.sum(dim=2)                                  # (B, n_atoms, d)
        grad = 4 * self.epsilon * grad                                    # (B, n_atoms, d)

        # ---------------------------
        # Laplacian of LJ potential
        # ---------------------------
        coeff1 =  12 * 13 * self.r0**12 * r_inv14 - 6 * 7 * self.r0**6 * r_inv8   # (B, n_atoms, n_atoms)
        coeff2 = -12      * self.r0**12 * r_inv14 + 6     * self.r0**6 * r_inv8   # (B, n_atoms, n_atoms)

        ratio    = dX / r[..., None]                    # (B, n_atoms, n_atoms, d)
        ratio_sq = ratio**2                             # (B, n_atoms, n_atoms, d)

        term1   = coeff1[..., None] * ratio_sq                      # (B, n_atoms, n_atoms, d)
        term2   = coeff2[..., None] * (1.0 - ratio_sq)              # (B, n_atoms, n_atoms, d)
        lap_pair = term1 + term2                                   # (B, n_atoms, n_atoms, d)

        laplace = lap_pair.sum(dim=2)                               # (B, n_atoms, d)
        laplace = 4 * self.epsilon * laplace                             # (B, n_atoms, d)

        grad_flat    = grad.view(B, self.n_atoms * self.d)
        laplace_flat = laplace.view(B, self.n_atoms * self.d)
        return grad_flat, laplace_flat