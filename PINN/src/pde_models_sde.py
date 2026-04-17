import torch
import derivatives



class SmoluchowskiGeneral:
    """
    formulated in SDE and PDE form
    p_inf(x) = 1/Z e**(-beta V(x))
    """
    def __init__(self, d, beta):
        self.d = d
        self.beta = beta
        self.dist_initial = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d)
        )
        ### sde terms:
        self.sigma = torch.sqrt( torch.tensor(2.0)/beta )
    def L_functional(self, X, s, s_div, precomputed):
        return (
            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + (precomputed["V_grad"] * s).sum(dim=1).unsqueeze(1)
            + precomputed["V_laplace"]
        )
    def V(self, x):
        raise NotImplementedError
    def V_grad(self, x):
        raise NotImplementedError
    def V_laplace(self, x):
        raise NotImplementedError
    #def p_analytic(self, X):
    #    return self.gaussian_obj.p(X[:,:-1], X[:,-1:])
    #def q_analytic(self, X):
    #    return self.gaussian_obj.log_p(X[:,:-1], X[:,-1:])
    #def s_analytic(self, X):
    #    return self.gaussian_obj.s(X[:,:-1], X[:,-1:])
    def sample_x0(self, n_samples):
        return self.dist_initial.rsample((n_samples,))
    def p0(self, x):
        return torch.exp(self.dist_initial.log_prob(x))
    def p_inf(self, x):
        Z = 1.0
        return torch.exp(-1.0*self.beta*self.V(x)).unsqueeze(1) / Z
    def _precompute_V_grad_V_laplace(self, X) -> dict[str, torch.tensor]:
        return {}
    
    class Score_PDE:
        def __init__(self, score_sde_model) -> None:
            self.score_sde_model = score_sde_model
        def __getattr__(self, name):
            return getattr(self.score_sde_model, name)
        def s0(self, x):
            x.detach()
            x.requires_grad_(True)
            q = self.dist_initial.log_prob(x).unsqueeze(1)
            s0 = derivatives.compute_grad(x, q, torch.ones_like(q)).detach()
            return s0
        def pde_residual(self, X, model_s, precomputed):
            X.detach()
            X.requires_grad_(True)
            s, s_t, s_div = derivatives.compute_score_dt_div(model_s, X)
            L = self.L_functional(X, s, s_div, precomputed)
            assert L.shape == (X.shape[0], 1)
            residual = s_t - derivatives.compute_grad(X, L, torch.ones_like(L))[:,:-1]
            return residual
        def bc_residual(self, X, model_s, precomputed):
            n = precomputed["normals"]
            return ( ( 1/self.beta * model_s(X) + self.V_grad(X[:,:-1]) ) * n ).sum(dim=1).unsqueeze(dim=1)
        def ic_residual(self, X, model_s, precomputed):
            return model_s(X) - precomputed["s0"]
        def __term_loss(self, d_dim_residual):
            loss = torch.mean(torch.sum(d_dim_residual**2, dim=1))
            return loss
        def pde_loss(self, X, model_s, precomputed):
            res = self.pde_residual(X, model_s, precomputed)
            return self.__term_loss(res)
        def bc_loss(self, X, model_s, precomputed):
            return torch.mean(self.bc_residual(X, model_s, precomputed)**2)
        def ic_loss(self, X, model_s, precomputed):
            res = self.ic_residual(X, model_s, precomputed)
            return self.__term_loss(res)
        def precompute(self, X_pde, X_bc, X_ic):
            return {
                "pde": self._precompute_V_grad_V_laplace(X_pde),
                "bc": {},
                "ic": {
                    "s0": self.s0(X_ic[:,:-1]).detach()
                },
            }

    class LL_ODE:
        def __init__(self, score_sde_model, model_s):
            self.score_sde_model = score_sde_model
            self.model_s = model_s
        def __getattr__(self, name):
            return getattr(self.score_sde_model, name)
        def q0(self, x):
            return self.dist_initial.log_prob(x).unsqueeze(1)
        def pde_residual(self, X, model_q, precomputed):
            X.detach()
            X.requires_grad_(True)
            q = model_q(X)
            q_t = derivatives.compute_grad(X, q, torch.ones_like(q))[:,-1:]
            return q_t - precomputed["L"]
        def bc_residual(self, X, model_q, precomputed):
            n = precomputed["normals"]
            X = X.detach().requires_grad_(True)
            _, grad_q, _ = derivatives.compute_derivatives(model_q, X, compute_laplace=False)
            return ( ( 1/self.beta * grad_q[:,:-1] + self.V_grad(X[:,:-1]) ) * n ).sum(dim=1).unsqueeze(dim=1)
        def ic_residual(self, X, model_q, precomputed):
            return model_q(X) - precomputed["q0"]
        def pde_loss(self, X, model_q, precomputed):
            res = self.pde_residual(X, model_q, precomputed)
            loss = torch.mean(res**2)
            return loss
        def bc_loss(self, X, model_q, precomputed):
            res = self.bc_residual(X, model_q, precomputed)
            loss = torch.mean(res**2)
            return loss
        def ic_loss(self, X, model_q, precomputed):
            res = self.ic_residual(X, model_q, precomputed)
            loss = torch.mean(res**2)
            return loss
        def precompute(self, X_pde, X_bc, X_ic):
            X_pde.detach()
            X_pde.requires_grad_(True)
            s, _, s_div = derivatives.compute_score_dt_div(self.model_s, X_pde)
            L = self.L_functional(X_pde, s, s_div, self._precompute_V_grad_V_laplace(X_pde))
            return {
                "pde": {
                    "L": L.detach()
                },
                "bc": {},
                "ic": {
                    "q0": self.q0(X_ic[:,:-1]).detach()
                },
            }



class SmoluchowskiDiffDrift(SmoluchowskiGeneral):
    "Diffusive drift"
    def __init__(self, d, beta, c):
        super().__init__(d, beta)
        ### other
        self.c = c if c is not None else torch.rand(d)
        assert c.ndim == 1
        self.mu = - self.c
    def L_functional(self, X, s, s_div, precomputed):
        return (
            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + (s * self.c).sum(dim=1).unsqueeze(1)
        )
    def V(self, x):
        return (x * self.c).sum(dim=1).unsqueeze(1)
    def V_grad(self, x):
        "cause: = self.c"
        return torch.ones((x.shape[0], 1)) * self.c
    def V_laplace(self, x):
        "cause: = 0"
        return torch.zeros((x.shape[0], 1))

    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
        def __init__(self, score_sde_model) -> None:
            super().__init__(score_sde_model)

    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
        def __init__(self, score_sde_model, model_s):
            super().__init__(score_sde_model, model_s)


class SmoluchowskiHarmonicPot(SmoluchowskiGeneral):
    def __init__(self, d, beta, k=None):
        super().__init__(d, beta)
        ### other
        self.k = k if k is not None else 1.0
        self.mu = lambda x,t: - 1.0 * self.V_grad()
    def L_functional(self, X, s, s_div, precomputed):
        return (
            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + self.k**2 * (s * X[:,:-1]).sum(dim=1).unsqueeze(1)
            + self.d * self.k**2
        )
    def V(self, x):
        return 0.5*self.k**2 * (x**2).sum(dim=1).unsqueeze(1)
    def V_grad(self, x):
        "cause: = k**2 * x"
        return self.k**2 * x
    def V_laplace(self, x):
        "cause: = d k**2"
        return torch.ones((x.shape[0], 1)) * d * self.k**2

    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
        def __init__(self, score_sde_model) -> None:
            super().__init__(score_sde_model)

    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
        def __init__(self, score_sde_model, model_s):
            super().__init__(score_sde_model, model_s)


class SmoluchowskiCoupledQuadraticPot(SmoluchowskiGeneral):
    """
    V(x) = 1/2 x^T A x
    A - SPD
    for example:
    A = [
            [a1,g1,0,0,g1]
            [g2,a2,g2,0,0]
            [0,g3,a3,g3,0]
            [0,0,g4,a4,g4]
            [g5,0,0,g5,a5]
        ]
    V_grad = A x
    V_laplace = Tr(A)

    Final distribution p_inf is basically a highdimensional gaussian elipsoid
    (the p_inf gaussinal is off-center when the the other dimensions are not 0)
    """
    def __init__(self, d, beta, A=None):
        super().__init__(d, beta)
        self.A = A
        self.tr_A = torch.trace(A)
        ### sde:
        self.mu = lambda x,t: - 1.0 * self.V_grad(x)
    def L_functional(self, X, s, s_div, precomputed=None):
        # - <..,s> - div(..)
        # V_grad.s + V_laplace
        return (
            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + (s * self.V_grad(X[:,:-1])).sum(dim=1).unsqueeze(1)
            + self.tr_A
        )
    def V(self, x):
        "cause: = 1/2 x^T A x"
        y = x @ self.A.transpose(0,1)
        return 0.5 * (x * y).sum(dim=1).unsqueeze(1)
    def V_grad(self, x):
        "cause: = A x"
        return x @ self.A.transpose(0,1)
    def V_laplace(self, x):
        "cause: = Tr(A)"
        return torch.ones((x.shape[0], 1)) * self.tr_A

    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
        def __init__(self, score_sde_model) -> None:
            super().__init__(score_sde_model)

    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
        def __init__(self, score_sde_model, model_s):
            super().__init__(score_sde_model, model_s)


class SmoluchowskiDoubleWell(SmoluchowskiGeneral):
    """
    V(x) = 1/4 sum_{i=1}^d (x_i^2 - a_i^2)^2
    V_grad_i(x) = (x_i^2 - a_i^2) * x_i
    V_laplace_i(x) = 3 x_i^2 - a_i^2
    V_laplace(x) = 3|x|^2 - |a|^2
    """
    def __init__(self, d, beta, a=None):
        super().__init__(d, beta)
        self.a = a
        self.a_l2 = (a**2).sum().item()
        self.mu = lambda x,t: - 1.0 * self.V_grad(x)
    def L_functional(self, X, s, s_div, precomputed=None):
        # - <..,s> - div(..)
        # V_grad.s + V_laplace
        return (
            1/self.beta * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + (s * self.V_grad(X[:,:-1])).sum(dim=1).unsqueeze(1)
            + self.V_laplace(X[:,:-1])
        )
    def V(self, x):
        return 0.25 * ( (x**2 - self.a**2)**2 ).sum(dim=1).unsqueeze(1)
    def V_grad(self, x):
        return (x**2 - self.a**2) * x
    def V_laplace(self, x):
        return 3.0 * (x**2).sum(dim=1).unsqueeze(1) - self.a_l2

    class Score_PDE(SmoluchowskiGeneral.Score_PDE):
        def __init__(self, score_sde_model) -> None:
            super().__init__(score_sde_model)

    class LL_ODE(SmoluchowskiGeneral.LL_ODE):
        def __init__(self, score_sde_model, model_s):
            super().__init__(score_sde_model, model_s)


class SmoluchowskiCoupledDoubleWell(SmoluchowskiGeneral):
    """
    V(x) = 1/4 sum_{i=1}^d (x_i^2 - a_i^2)^2 + sum_{i=1}^d gamma_i (x_{i+1} - x_i)**2
    """
    def __init__(self, d, beta, a=None, gamma=None):
        super().__init__(d, beta)
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


