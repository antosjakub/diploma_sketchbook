
import torch

import derivatives
import utility

class SmoluchowskiGeneral:
    """
    formulated in SDE and PDE form
    p_inf(x) = 1/Z e**(-beta V(x))
    """
    def __init__(self, d, beta=1.0, Z=1.0):
        self.d = d
        self.beta = beta
        self.dist_initial = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d)
        )
        self.Z = torch.tensor(Z)
        ### sde terms:
        self.sigma = torch.sqrt( torch.tensor(2.0)/beta )
    def V(self, x):
        raise NotImplementedError
    def V_grad(self, x):
        raise NotImplementedError
    def V_laplace(self, x):
        raise NotImplementedError

    def estimate_Z_full(self, p_inf, d, spatial_domain, n_samples=200_000, proposal="uniform", gaussian_sigma=None, device='cpu'):
        """MC estimate of Z = ∫ exp(-beta*V(x)) dx over the box.

        proposal:
          - "uniform":  x ~ Uniform(Ω),          Z ≈ |Ω| · mean(p_inf(x))
          - "gaussian": x ~ N(0, gaussian_sigma² I),
                        Z ≈ mean(p_inf(x) / q(x))
        gaussian_sigma: scalar std of the isotropic Gaussian proposal.
                        Default: half the largest side of the box (covers the wells).
        """
        lo = spatial_domain[:, 0].to(device)
        hi = spatial_domain[:, 1].to(device)
        if proposal == "uniform":
            vol = (hi - lo).prod().item()
            X = lo + (hi - lo) * torch.rand(n_samples, d, device=device)
            return vol * p_inf(X).mean().item()
        elif proposal == "gaussian":
            if gaussian_sigma is None:
                gaussian_sigma = 0.5 * (hi - lo).max()
            sigma = gaussian_sigma
            X = sigma * torch.randn(n_samples, d, device=device)
            # q(x) = (2π σ²)^(-d/2) · exp(-‖x‖² / (2σ²))
            log_q = -0.5 * d * torch.log(2 * torch.pi * sigma * sigma) \
                    - 0.5 * (X * X).sum(dim=1) / (sigma * sigma)
            q = torch.exp(log_q).unsqueeze(1)
            return (p_inf(X) / q).mean().item()
        else:
            raise ValueError(f"Unknown proposal '{proposal}'. Use 'uniform' or 'gaussian'.")

    def sample_x0(self, n_samples):
        return self.dist_initial.rsample((n_samples,))
    def sample_xinf(self, n_samples):
        """Sample approximately from the normalized stationary density p_inf.

        Uses a simple random-walk Metropolis-Hastings kernel targeting p_inf.
        This avoids relying on a box-dependent rejection sampler and works as
        long as p_inf is evaluable up to normalization (here assumed normalized).
        """
        device = self.dist_initial.loc.device
        dtype = self.dist_initial.loc.dtype

        with torch.no_grad():
            x = self.sample_x0(n_samples).to(device=device, dtype=dtype)
            p_curr = self.p_inf(x).squeeze(1)

            proposal_scale = torch.full(
                (1, self.d),
                max(float(self.sigma), 0.25),
                device=device,
                dtype=dtype,
            )
            n_burnin = 200
            n_steps = 200

            for _ in range(n_burnin + n_steps):
                x_prop = x + proposal_scale * torch.randn_like(x)
                p_prop = self.p_inf(x_prop).squeeze(1)
                accept_prob = torch.minimum(
                    torch.ones_like(p_curr),
                    p_prop / (p_curr + 1e-16),
                )
                accepted = torch.rand(n_samples, device=device) < accept_prob
                if accepted.any():
                    x[accepted] = x_prop[accepted]
                    p_curr[accepted] = p_prop[accepted]

        return x

    def p_0(self, x):
        return torch.exp(self.dist_initial.log_prob(x)).unsqueeze(1)
    def q_0(self, x):
        return self.dist_initial.log_prob(x).unsqueeze(1)
    def p_inf(self, x):
        return torch.exp(-1.0*self.beta*self.V(x)) / self.Z
    def q_inf(self, x):
        return - 1.0*self.beta*self.V(x) - torch.log(self.Z) 

    def pde_residual(self, X, model, precomputed=None):
        raise NotImplementedError
    def bc_residual(self, X, model, precomputed=None):
        raise NotImplementedError
    def ic_residual(self, X, model, precomputed=None):
        return model(X) - precomputed["ic"]

    def pde_loss(self, X, model, precomputed):
        res = self.pde_residual(X, model, precomputed)
        return torch.mean(res**2)
    def bc_loss(self, X, model, precomputed):
        res = self.bc_residual(X, model, precomputed)
        return torch.mean(res**2)
    def ic_loss(self, X, model, precomputed):
        res = self.ic_residual(X, model, precomputed)
        return torch.mean(res**2)

    def get_pde_metadata(self):
        raise NotImplementedError
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})


    class ClassPDE:
        def __init__(self, base_pde_model, bc_type='dir'):
            self.base_pde_model = base_pde_model
            if bc_type == 'dir':
                self.bc_residual = self.bc_residual_dirichlet
            elif bc_type == 'neu':
                self.bc_residual = self.bc_residual_neumann
            else:
                NameError
        def __getattr__(self, name):
            return getattr(self.base_pde_model, name)

        def pde_residual_base(self, X, u, grad_u, spatial_laplace_u, precomputed=None):
            return grad_u[:,-1:] - (
                1/self.beta * spatial_laplace_u.sum(dim=1).unsqueeze(dim=1)
                + (precomputed["V_grad"] * grad_u[:,:-1]).sum(dim=1).unsqueeze(1)
                + precomputed["V_laplace"] * u
            )
        def pde_residual(self, X, model, precomputed_pde=None):
            X = X.detach().requires_grad_(True)
            u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
            return self.pde_residual_base(None, u, grad_u, spatial_laplace_u, precomputed_pde)

        def bc_residual_neumann(self, X, model, precomputed):
            n = precomputed["normals"]
            X = X.detach().requires_grad_(True)
            u, grad, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
            return ( ( 1/self.beta * grad[:,:-1] + u * self.V_grad(X[:,:-1]) ) * n ).sum(dim=1).unsqueeze(dim=1)
        def bc_residual_dirichlet(self, X, model, precomputed):
            return model(X)
        def bc_residual(self, X, model, precomputed):
            raise NotImplementedError
        #def bc_residual(self, X, model, precomputed):
        #    return self.bc_residual_dirichlet(self, X, model, precomputed)

        def precompute(self, X_pde, X_bc, X_ic):
            return {
                "pde": {
                    "V_grad": self.V_grad(X_pde[:,:-1]),
                    "V_laplace": self.V_laplace(X_pde[:,:-1]),
                } if X_pde is not None else {},
                "bc": {},
                "ic": {
                    "ic": self.p_0(X_ic[:,:-1]).detach()
                } if X_ic is not None else {},
            }

    class LogPDE:
        def __init__(self, base_pde_model, bc_type='dir'):
            self.base_pde_model = base_pde_model
            self.bc_type = bc_type
            if bc_type == 'dir':
                self.bc_residual = self.bc_residual_dirichlet
            elif bc_type == 'neu':
                self.bc_residual = self.bc_residual_neumann
            else:
                NameError
        def __getattr__(self, name):
            return getattr(self.base_pde_model, name)

        def pde_residual_base(self, X, u, grad_u, spatial_laplace_u, precomputed=None):
            return grad_u[:,-1:] - (
                1/self.beta * (
                    spatial_laplace_u.sum(dim=1).unsqueeze(dim=1)
                    + (grad_u**2).sum(dim=1).unsqueeze(dim=1)
                )
                + (precomputed["V_grad"] * grad_u[:,:-1]).sum(dim=1).unsqueeze(1)
                + precomputed["V_laplace"]
            )
        def pde_residual(self, X, model, precomputed_pde=None):
            X = X.detach().requires_grad_(True)
            u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)
            return self.pde_residual_base(None, u, grad_u, spatial_laplace_u, precomputed_pde)

        def bc_residual_neumann(self, X, model, precomputed):
            n = precomputed["normals"]
            X = X.detach().requires_grad_(True)
            _, grad, _ = derivatives.compute_derivatives(model, X, compute_laplace=False)
            return ( ( 1/self.beta * grad[:,:-1] + self.V_grad(X[:,:-1]) ) * n ).sum(dim=1).unsqueeze(dim=1)
        def bc_residual_dirichlet(self, X, model, precomputed):
            return model(X)
        #def bc_residual(self, X, model, precomputed):
        #    return self.bc_residual_dirichlet(self, X, model, precomputed)
        def bc_residual(self, X, model, precomputed):
            raise NotImplementedError

        def precompute(self, X_pde, X_bc, X_ic):
            return {
                "pde": {
                    "V_grad": self.V_grad(X_pde[:,:-1]),
                    "V_laplace": self.V_laplace(X_pde[:,:-1]),
                } if X_pde is not None else {},
                "bc": {},
                "ic": {
                    "ic": self.q_0(X_ic[:,:-1]).detach()
                } if X_ic is not None else {},
            }


class SmoluchowskiDoubleWell(SmoluchowskiGeneral):
    """
    V(x) = 1/4 sum_{i=1}^d (x_i^2 - a_i^2)^2
    V_grad_i(x) = (x_i^2 - a_i^2) * x_i
    V_laplace_i(x) = 3 x_i^2 - a_i^2
    V_laplace(x) = 3|x|^2 - |a|^2
    """
    def __init__(self, d, beta, Z=1.0, a=None):
        super().__init__(d, beta, Z)
        self.a = a
        self.a_l2 = (a**2).sum().item()
        self.mu = lambda x,t: - 1.0 * self.V_grad(x)
    #def L_functional(self, X, grad_u, spatial_laplace_u, precomputed=None):
    #    # - <..,s> - div(..)
    #    # V_grad.s + V_laplace
    #    return (
    #        1/self.beta * ( spatial_laplace.sum() + (grad_u**2).sum(dim=1).unsqueeze(1) )
    #        + (s * self.V_grad(X[:,:-1])).sum(dim=1).unsqueeze(1)
    #        + self.V_laplace(X[:,:-1])
    #    )
    def V(self, x):
        return 0.25 * ( (x**2 - self.a**2)**2 ).sum(dim=1).unsqueeze(1)
    def V_grad(self, x):
        return (x**2 - self.a**2) * x
    def V_laplace(self, x):
        return 3.0 * (x**2).sum(dim=1).unsqueeze(1) - self.a_l2
    def get_pde_metadata(self):
        return {
            "beta": self.beta,
            "a": list(map(lambda x: float(x), self.a)),
            "Z": self.Z.item()
        }


class SmoluchowskiCoupledDoubleWell(SmoluchowskiGeneral):
    """
    V(x) = 1/4 sum_{i=1}^d (x_i^2 - a_i^2)^2 + sum_{i=1}^d gamma_i (x_{i+1} - x_i)**2
    """
    def __init__(self, d, beta, Z=1.0, a=None, gamma=None):
        super().__init__(d, beta, Z)
        self.a = a
        self.gamma = gamma
    def V(self, x):
        return (
            0.25 * ( (x**2 - self.a**2)**2 ).sum(dim=1).unsqueeze(1)
            + ((x.roll(-1) - x)**2 * self.gamma).sum(dim=1).unsqueeze(1)
        )


class SmoluchowskiRastigin(SmoluchowskiGeneral):
    """
    V(x) = A d - sum_{i=1}^d (x_i^2 - A cos(2pi x_i))
    """
    def __init__(self, d, beta, A=None, gamma=None):
        super().__init__(d, beta)
        self.A = A if A is not None else 0.3
        self.gamma = gamma if gamma is not None else 6*torch.pi*torch.ones(d)
    def V(self, x):
        #return self.A*self.d + (x**2 - self.A * torch.cos(2.0 * torch.pi * x)).sum(dim=1).unsqueeze(1)
        return (x**2 - self.A * torch.cos(x * self.gamma)).sum(dim=1).unsqueeze(1)




#class SmoluchowskiDiffDrift(SmoluchowskiGeneral):
#    "Diffusive drift"
#    def __init__(self, d, beta, c):
#        super().__init__(d, beta)
#        ### other
#        self.c = c if c is not None else torch.rand(d)
#        assert c.ndim == 1
#        self.mu = - self.c
#    def L_functional(self, X, s, s_div, precomputed):
#        return (
#            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
#            + (s * self.c).sum(dim=1).unsqueeze(1)
#        )
#    def V(self, x):
#        return (x * self.c).sum(dim=1).unsqueeze(1)
#    def V_grad(self, x):
#        "cause: = self.c"
#        return torch.ones((x.shape[0], 1)) * self.c
#    def V_laplace(self, x):
#        "cause: = 0"
#        return torch.zeros((x.shape[0], 1))
#
#    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
#        def __init__(self, score_sde_model) -> None:
#            super().__init__(score_sde_model)
#
#    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
#        def __init__(self, score_sde_model, model_s):
#            super().__init__(score_sde_model, model_s)
#
#
#class SmoluchowskiHarmonicPot(SmoluchowskiGeneral):
#    def __init__(self, d, beta, k=None):
#        super().__init__(d, beta)
#        ### other
#        self.k = k if k is not None else 1.0
#        self.mu = lambda x,t: - 1.0 * self.V_grad()
#    def L_functional(self, X, s, s_div, precomputed):
#        return (
#            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
#            + self.k**2 * (s * X[:,:-1]).sum(dim=1).unsqueeze(1)
#            + self.d * self.k**2
#        )
#    def V(self, x):
#        return 0.5*self.k**2 * (x**2).sum(dim=1).unsqueeze(1)
#    def V_grad(self, x):
#        "cause: = k**2 * x"
#        return self.k**2 * x
#    def V_laplace(self, x):
#        "cause: = d k**2"
#        return torch.ones((x.shape[0], 1)) * d * self.k**2
#
#    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
#        def __init__(self, score_sde_model) -> None:
#            super().__init__(score_sde_model)
#
#    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
#        def __init__(self, score_sde_model, model_s):
#            super().__init__(score_sde_model, model_s)
#
#
#class SmoluchowskiCoupledQuadraticPot(SmoluchowskiGeneral):
#    """
#    V(x) = 1/2 x^T A x
#    A - SPD
#    for example:
#    A = [
#            [a1,g1,0,0,g1]
#            [g2,a2,g2,0,0]
#            [0,g3,a3,g3,0]
#            [0,0,g4,a4,g4]
#            [g5,0,0,g5,a5]
#        ]
#    V_grad = A x
#    V_laplace = Tr(A)
#
#    Final distribution p_inf is basically a highdimensional gaussian elipsoid
#    (the p_inf gaussinal is off-center when the the other dimensions are not 0)
#    """
#    def __init__(self, d, beta, A=None):
#        super().__init__(d, beta)
#        self.A = A
#        self.tr_A = torch.trace(A)
#        ### sde:
#        self.mu = lambda x,t: - 1.0 * self.V_grad(x)
#    def L_functional(self, X, s, s_div, precomputed=None):
#        # - <..,s> - div(..)
#        # V_grad.s + V_laplace
#        return (
#            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
#            + (s * self.V_grad(X[:,:-1])).sum(dim=1).unsqueeze(1)
#            + self.tr_A
#        )
#    def V(self, x):
#        "cause: = 1/2 x^T A x"
#        y = x @ self.A.transpose(0,1)
#        return 0.5 * (x * y).sum(dim=1).unsqueeze(1)
#    def V_grad(self, x):
#        "cause: = A x"
#        return x @ self.A.transpose(0,1)
#    def V_laplace(self, x):
#        "cause: = Tr(A)"
#        return torch.ones((x.shape[0], 1)) * self.tr_A
#
#    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
#        def __init__(self, score_sde_model) -> None:
#            super().__init__(score_sde_model)
#
#    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
#        def __init__(self, score_sde_model, model_s):
#            super().__init__(score_sde_model, model_s)
#
#
#class SmoluchowskiDoubleWell(SmoluchowskiGeneral):
#    """
#    V(x) = 1/4 sum_{i=1}^d (x_i^2 - a_i^2)^2
#    V_grad_i(x) = (x_i^2 - a_i^2) * x_i
#    V_laplace_i(x) = 3 x_i^2 - a_i^2
#    V_laplace(x) = 3|x|^2 - |a|^2
#    """
#    def __init__(self, d, beta, a=None):
#        super().__init__(d, beta)
#        self.a = a
#        self.a_l2 = (a**2).sum().item()
#        self.mu = lambda x,t: - 1.0 * self.V_grad(x)
#    def L_functional(self, X, s, s_div, precomputed=None):
#        # - <..,s> - div(..)
#        # V_grad.s + V_laplace
#        return (
#            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
#            + (s * self.V_grad(X[:,:-1])).sum(dim=1).unsqueeze(1)
#            + self.V_laplace(X[:,:-1])
#        )
#    def V(self, x):
#        return 0.25 * ( (x**2 - self.a**2)**2 ).sum(dim=1).unsqueeze(1)
#    def V_grad(self, x):
#        return (x**2 - self.a**2) * x
#    def V_laplace(self, x):
#        return 3.0 * (x**2).sum(dim=1).unsqueeze(1) - self.a_l2
#
#    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
#        def __init__(self, score_sde_model) -> None:
#            super().__init__(score_sde_model)
#
#    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
#        def __init__(self, score_sde_model, model_s):
#            super().__init__(score_sde_model, model_s)

