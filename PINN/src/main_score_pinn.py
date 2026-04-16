import torch
import argparse
from torch.profiler import record_function
import sampling, loss, architecture, utility

parser = argparse.ArgumentParser()
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="148,148,148", type=str, help="")
parser.add_argument("--n_steps", default=450, type=int, help="")
parser.add_argument("--n_steps_decay", default=2000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--bs", default=512, type=int, help="")

parser.add_argument("--n_res_points", default=10_000, type=int, help="")
parser.add_argument("--n_trajs", default=1_000, type=int, help="")
parser.add_argument("--nt_steps", default=100, type=int, help="")
parser.add_argument("--T", default=1.5, type=float, help="")

parser.add_argument("--n_test_points", default=10_000, type=int, help="Number of test points for the testing suite.")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")

parser.add_argument("--resampling_frequency", default=1000, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_bc", default=0.01, type=float, help="")
parser.add_argument("--lambda_ic", default=100.0, type=float, help="")
parser.add_argument("--lambda_norm", default=0.1, type=float, help="Weight of the ∫p dx = 1 normalization loss.")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--active_losses", default="pde,bc,ic", type=str, help="Comma-separated subset of {pde,bc,ic,norm}. 'pde' is required.")

parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# smart Defaults
parser.add_argument("--output_dir", default="run_score_pinn_latest/", type=str, help="")
parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")

parser.add_argument("--mode", default="score_pde", type=str, help="score_pde or ll_ode")
#
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default="run_sp_latest/model.pth", type=str, help="")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")

import derivatives

#class isotropic_SDE:
#    def __init__(self, sigma, mu):
#        self.sigma = 0.1
#        self.mu = 0
#        # detect whether mu is a constant or a function of x
#        self.loc = torch.zeros(d)
#        self.cov = torch.eye(d)
#        self.dist = torch.distributions.MultivariateNormal(
#            loc=self.loc,
#            covariance_matrix=self.cov
#        )
#    def L_functional(self, s, s_div, precomputed):
#        return (
#            self.sigma**2/2 * (s_div + (s**2).sum(dim=1).unsqueeze(1))
#            - (precomputed["mu"] * s).sum(dim=1).unsqueeze(1)
#            - precomputed["mu_grad"]
#        )
#    def ll_ode_redisual(self, model_q, model_s, X, precomputed):
#        s, _, s_div = derivatives.compute_score_dt_div(model_s, X)
#        q = model_q(X)
#        q_t = derivatives.compute_grad(X, q, torch.ones(q))[:,-1:]
#        return q_t - self.L_functional(s, s_div, precomputed)
#    def score_pde_residual(self, model, X, precomputed):
#        s, s_t, s_div = derivatives.compute_score_dt_div(model, X)
#        L = self.L_functional(s, s_div, precomputed)
#        assert L.shape == (X.shape[0], 1)
#        return s_t - derivatives.compute_grad(X, L, torch.ones(L))[:,:-1]
#    def score_ic_residual(self, model_s, X, precomputed):
#        return model_s(X) - precomputed["s0"]
#    def q0(self, x):
#        return self.dist.log_prob(x)
#    def s0(self, x):
#        assert x.requires_grad == True
#        return derivatives.compute_grad(self.dist.log_prob(x))[:,:-1]
#    def p0(self, x):
#        return torch.exp(self.dist.log_prob(x))
#    def sample_x0(self, n_samples):
#        return self.dist.rsample((n_samples,))
#    def precomputed(self, X):
#        return {
#            "pde": {
#                "": 0.0
#            },
#            "ic": {
#                "s0": self.s0(X[:,-1:])
#            },
#        }


from typing import Optional

#class DiagonalGaussian(GeneralGaussian):
#    def __init__(
#        self,
#        d: int,
#        gammas: None,
#        x0 = None,
#        device: Optional[torch.device] = None,
#        dtype: torch.dtype = torch.float32,
#        seed: int = 76,
#    ):
#        super().__init__()
#
#        if gamma_min < 0 or gamma_max <= 0 or gamma_min > gamma_max:
#            raise ValueError("Need 0 <= gamma_min <= gamma_max and gamma_max > 0")
#        self.d = d
#        self.x0 = x0 if (x0 is not None) else torch.zeros(d)
#        self.device = device
#        self.dtype = dtype
#        self.Q = torch.ones((d,d), dtype=dtype)
#
#        QT = Q.transpose(0, 1)
#        self.Sigma = (QT * gamma.unsqueeze(1)) @ Q
#        self.Sigma_sqrt = (QT * torch.sqrt(gamma).unsqueeze(1)) @ Q


import math
class GeneralGaussian:
    """
    Builds a covariance Sigma = Q^T Gamma Q, then provides:
      - s(x, t) = Sigma_t^{-1} x
      - log p(x, t) for N(0, Sigma_t)
      - p(x, t) = exp(log p(x, t))
    Shapes:
      x: (bs, d)
      t: (bs, 1)
    
    Sigma = Q^T Gamma Q
    """

    def __init__(
        self,
        d: int,
        gamma_min: float = 0.1,
        gamma_max: float = 2.0,
        x0 = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 76,
    ):
        super().__init__()

        if gamma_min < 0 or gamma_max <= 0 or gamma_min > gamma_max:
            raise ValueError("Need 0 <= gamma_min <= gamma_max and gamma_max > 0")
        self.d = d
    
        self.x0 = x0 if (x0 is not None) else torch.zeros(d)

        self.device = device
        self.dtype = dtype
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        # 1) Construct Gamma diagonal
        gamma = gamma_min + (gamma_max - gamma_min) * torch.rand(
            d, generator=g, device=device, dtype=dtype
        )
        self.gamma = gamma
        self.gamma = 10*torch.ones(d)

        # Random orthogonal matrix Q from QR of a Gaussian matrix.
        A = torch.randn(d, d, generator=g, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(A, mode="reduced")

        # Fix signs for a more uniform-looking orthogonal draw.
        # Makes diag(R) positive.
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
        self.Q = torch.ones((d,d), dtype=dtype)
        self.Q = Q

        # Precompute dense Sigma and Sigma^(1/2) once
        # Using Sigma = Q^T diag(gamma) Q
        # Implemented as (Q^T * gamma) @ Q to avoid materializing diag(gamma)
        QT = Q.transpose(0, 1)
        self.Sigma = (QT * gamma.unsqueeze(1)) @ Q
        self.Sigma_sqrt = (QT * torch.sqrt(gamma).unsqueeze(1)) @ Q


    def _check_inputs(self, x: torch.Tensor, t: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must have shape (bs, {self.d}), got {tuple(x.shape)}")
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.ndim != 2 or t.shape[1] != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must have shape (bs, 1) or (bs,), got {tuple(t.shape)} with bs={x.shape[0]}"
            )

    def Sigma_t_evals(self, t: torch.Tensor) -> torch.Tensor:
        """
        Shape: (bs, d)
        """
        et = torch.exp(-t)  # (bs, 1)
        lam = et + (1.0 - et) * self.gamma.unsqueeze(0)  # (bs, d)
        return lam

    def log_p(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns: log(p(x,t)); p(x,t) = N(0, Sigma_t)(x)
        Input:
          x: (bs, d)
          t: (bs, 1) or (bs,)
        Output:
          log_p: (bs,1)
        """
        self._check_inputs(x, t)

        evals = self.Sigma_t_evals(t)  # (bs, d)
        y = (x-self.x0) @ self.Q.transpose(0, 1)   # (bs, d)
        maha = (y * y / evals).sum(dim=1)  # (bs,)

        logdet = torch.log(evals).sum(dim=1)

        return -0.5 * (self.d*math.log(2.0*math.pi) + logdet + maha).unsqueeze(1)

    def p(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Density p(x,t) = N(0, Sigma_t)
        Output shape: (bs,1)
        """
        return torch.exp(self.log_p(x, t))

    def s(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        s(x,t) = - Sigma_t^{-1} x
               = - Q^T 1/diag(gamma) Q x
        Input:
          x: (bs, d)
          t: (bs, 1)
        Output:
          (bs, d)
        """
        self._check_inputs(x, t)
        # apply transpose: (now x^T is a row vector - like in torch)
        # s^T = - x^T Q^T 1/diag(gamma) Q
        y = (x-self.x0) @ self.Q.transpose(0, 1)
        y = y / self.Sigma_t_evals(t)
        s = - y @ self.Q
        return s

    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from N(0, Sigma_t).
        """

        bs = t.shape[0]
        evals_sqrt = torch.sqrt(self.Sigma_t_evals(t))  # (bs, d)
        z = torch.randn(bs, self.d, device=t.device, dtype=self.dtype)

        # In eigenbasis: y ~ N(0, diag(evals))
        y = z * evals_sqrt
        # Back to original coords: x = y Q
        x = y @ self.Q
        return x + self.x0




class Isotropic_OU:
    def __init__(self, d, Sigma=torch.tensor(10.0)):
        self.d = d
        self.Sigma = Sigma
        # sde terms:
        self.mu = lambda x: -0.5*x
        self.sigma = torch.sqrt(Sigma)
        # detect whether mu is a constant or a function of x
        self.dist_initial = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d)
        )
        # the final distribution
        self.dist_inf = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d) * self.Sigma
        )
        self.gaussian_obj = GeneralGaussian(d, gamma_min=0.5, gamma_max=1.5)

    def p0(self, x):
        return torch.exp(self.dist_initial.log_prob(x)).unsqueeze(1)
    def sample_x0(self, n_samples):
        return self.dist_initial.rsample((n_samples,))
    def L_functional(self, X, s, s_div, precomputed=None):
        return 0.5 * (
            self.sigma**2 * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + ( (X[:,:-1] * s).sum(dim=1).unsqueeze(1) - self.d )
        )
    def p_analytic(self, X):
        return self.gaussian_obj.p(X[:,:-1], X[:,-1:])
    def q_analytic(self, X):
        return self.gaussian_obj.log_p(X[:,:-1], X[:,-1:])
    def s_analytic(self, X):
        return self.gaussian_obj.s(X[:,:-1], X[:,-1:])
    def p_inf(self, x):
        return torch.exp(self.dist_inf.log_prob(x)).unsqueeze(1)
    
    class Score_PDE:
        def __init__(self, score_sde_model) -> None:
            self.score_sde_model = score_sde_model
        def __getattr__(self, name):
            return getattr(self.score_sde_model, name)

        def s0(self, x):
            x.detach()
            x.requires_grad_(True)
            q = self.dist_initial.log_prob(x).unsqueeze(1)
            s0 = derivatives.compute_grad(x, q, torch.ones_like(q))
            return s0
        def pde_residual(self, X, model_s, precomputed):
            X.detach()
            X.requires_grad_(True)
            s, s_t, s_div = derivatives.compute_score_dt_div(model_s, X)
            L = self.L_functional(X, s, s_div, precomputed)
            assert L.shape == (X.shape[0], 1)
            residual = s_t - derivatives.compute_grad(X, L, torch.ones_like(L))[:,:-1]
            return residual
        def ic_residual(self, X, model_s, precomputed):
            return model_s(X) - precomputed["s0"]

        def _term_loss(self, d_dim_residual):
            loss = torch.mean(torch.sum(d_dim_residual**2, dim=1))
            return loss
        def pde_loss(self, X, model_s, precomputed):
            res = self.pde_residual(X, model_s, precomputed)
            return self._term_loss(res)
        def ic_loss(self, X, model_s, precomputed):
            res = self.ic_residual(X, model_s, precomputed)
            return self._term_loss(res)
        def precompute(self, X_pde, X_ic):
            return {
                "pde": {},
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
        def pde_loss(self, X, model_q, precomputed):
            res = self.pde_residual(X, model_q, precomputed)
            loss = torch.mean(res**2)
            return loss
        def ic_residual(self, X, model_q, precomputed):
            return model_q(X) - precomputed["q0"]
        def ic_loss(self, X, model_q, precomputed):
            res = self.ic_residual(X, model_q, precomputed)
            loss = torch.mean(res**2)
            return loss
        def precompute(self, X_pde, X_ic):
            X_pde.detach()
            X_pde.requires_grad_(True)
            s, _, s_div = derivatives.compute_score_dt_div(self.model_s, X_pde)
            L = self.L_functional(X_pde, s, s_div)
            return {
                "pde": {
                    "L": L.detach()
                },
                "ic": {
                    "q0": self.q0(X_ic[:,:-1]).detach()
                },
            }



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



"""
1. sampling to obtain X
    - sample x_0 ~ p_0
    - collect into X_ic
    - use x_0 to evolve the trajectories via the SDE
    - select random points from the trajectories
    - collect into X_pde
    - sample once, split into batches, resample every once in a while
2. loss
    - just normal loss with two lambda weights
3. residual
    -
"""




# Main execution
if __name__ == "__main__":
    import run_utils

    args = parser.parse_args([] if "__file__" not in globals() else None)

    d = args.d  # space dims
    D = d + 1   # space + time dims
    layers = utility.layers_from_string(args.layers)
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")

    type_sp = args.mode
    print(f"Training Score-PINN, type: '{type_sp}'\n")

    sde_model_label = 'DW'
    if type_sp == "score_pde":
        args.output_dir = f"run_SM-{sde_model_label}_{type_sp}"
        args.clear_dir = True
        args.enable_testing = False
    elif type_sp == "ll_ode":
        args.output_dir = f"run_SM-{sde_model_label}_{type_sp}"
        args.clear_dir = True
        args.enable_testing = False
        args.starting_model = f"run_SM-{sde_model_label}_score_pde/model.pth"
    else:
        raise NameError("Incorrect mode specified.")

    dir_name, device = run_utils.setup_run(args)

    a = 0.7 + 0.5*torch.rand(d)
    print(a)
    score_sde_model = SmoluchowskiDoubleWell(d=d, beta=1.0, a=a)



    ### PREP PDE MODEL
    #score_sde_model = Isotropic_OU(d=d)
    #score_sde_model = SmoluchowskiDiffDrift(d=d, beta=1.0, c=torch.ones(d))

    # Score PDE
    if type_sp == "score_pde":
        pde_model = score_sde_model.Score_PDE(score_sde_model)
    ## LL ODE
    elif type_sp == "ll_ode":
        model_s = architecture.PINN(D, layers, d).to(device)
        print(f"Loading in a score pde model: '{args.starting_model}'")
        model_s.load_state_dict(torch.load(args.starting_model, weights_only=True))
        model_s.eval()
        pde_model = score_sde_model.LL_ODE(score_sde_model, model_s)

    print(type(pde_model))
    print()
    #print(pde_model.get_pde_metadata())


    # Select the model architecture
    if type_sp == "score_pde":
        # NN t - x
        head_fn = lambda nn_out, X: nn_out * X[:,-1:] - X[:,:-1]
        model = architecture.PINN(D, layers, d, head_fn=head_fn).to(device)
    elif type_sp == "ll_ode":
        model = architecture.PINN(D, layers, 1).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)


    active_losses = tuple(k.strip() for k in args.active_losses.split(",") if k.strip())
    print(f"Active losses: {active_losses}")

    # Preparation time
    losses = run_utils.init_losses(("total",) + active_losses)
    l2_errs = []

    optimizer, scheduler = run_utils.make_optim(model, args)
    loss_weighting = run_utils.make_loss_weighting(args, active_losses)
    profiler = run_utils.make_profiler(dir_name, args)

    sdgd_num_dims = args.sdgd_num_dims if args.sdgd_num_dims is not None else d
    if args.use_sdgd:
        print(f"Using SDGD with {sdgd_num_dims} dimensions (d={d})")
    else:
        print(f"Using regular Adam training.")

    import time
    t1 = time.time()
    if args.enable_testing:
        if type_sp == "score_pde":
            analytic_fn = score_sde_model.s_analytic
        elif type_sp == "ll_ode":
            analytic_fn = score_sde_model.q_analytic
        testing_suite = utility.ScorePINNTestingSuite(d, analytic_fn)
        testing_suite.make_test_data(score_sde_model, args.n_test_points)
        print(f"Testing suite ready ({args.n_test_points} points, mode='{type_sp}').")
    else:
        testing_suite = None

    L = 3.0
    #sampling_settings = {
    #    "n_trajs": args.n_trajs,
    #    "nt_steps": args.nt_steps,
    #    "n_res_points": args.n_res_points,
    #    "bs": args.bs,
    #    "spatial_domain": torch.stack([torch.full((d,), -L), torch.full((d,), L)], dim=1),
    #    "T": args.T,
    #}
    T = args.T
    T = 1.0
    sampling_settings = {
        "n_res_points": args.n_res_points,
        "bs": args.bs,
        "spatial_domain": torch.stack([torch.full((d,), -L), torch.full((d,), L)], dim=1),
        "T": T,
        "use_rbas": args.use_rbas,
    }

    from main import PINN_Trainer
    trainer = PINN_Trainer(
        model, optimizer, scheduler, pde_model,
        #sampling_type="score_pinn", sampling_settings=sampling_settings,
        sampling_type="vanilla_pinn", sampling_settings=sampling_settings,
        loss_weighting=loss_weighting, testing_suite=testing_suite,
        active_losses=active_losses, profiler=profiler, device=device,
    )
    losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
        n_steps=args.n_steps,
        n_steps_decay=args.n_steps_decay,
        resampling_frequency=args.resampling_frequency,
        testing_frequency=args.testing_frequency,
        use_sdgd=args.use_sdgd,
        sdgd_num_dims=sdgd_num_dims,
    )
    run_utils.merge_losses(losses, losses_adam)
    l2_errs += l2_errs_adam
    print("\nAdam training complete!")
    run_utils.print_train_duration(t1, time.time())

    print("\nTraining complete!")

    loss_name, l2_name = run_utils.save_run(dir_name, model, losses, l2_errs, args)


    # Plot results
    import visualize_training_metrics
    visualize_training_metrics.plot_loss(losses, loss_name)
    if args.enable_testing:
        n_steps_log = args.testing_frequency
        n_logged_pnts = len(l2_errs)
        steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)
        visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)


    import viz
    p_ic = lambda X: pde_model.p0(X[:,:-1])
    p_inf = lambda X: pde_model.p_inf(X[:,:-1])

    options = {
        "d": d,
        "plot_dims": [0,1],
        "fixed_dims_vals": 0.5*torch.ones(d),
        "device": device,
        "x_start": -L,
        "x_end": L,
    }


    import os
    os.makedirs(f"{dir_name}/viz/", exist_ok=True)
    if type_sp == "score_pde":
        model_fn_s = viz.wrapp_model(model)
        s_ic = lambda X: pde_model.s0(X[:,:-1])

        if args.enable_testing:
            plotter = viz.FunctionPlotter(**options)
            plotter.add_panel('model_s', title="model_s(x,t)").quiver(model_fn_s)
            plotter.add_panel('s_analytic', title="s_analytic(x,t)").quiver(score_sde_model.s_analytic)
            plotter.add_panel('err', title="err").quiver(lambda X: model_fn_s(X) - score_sde_model.s_analytic(X))
            plotter.save_animation(f'{dir_name}/viz/anim_model_s_vs_s_analytic.gif', num_frames=30, fps=5, t_end=T)

        plotter_ic = viz.FunctionPlotter(**options)
        plotter_ic.add_panel('nn', rf"s_\theta(x,0)").quiver(model_fn_s)
        plotter_ic.add_panel('ic', "s_0(x)").quiver(s_ic)
        plotter_ic.add_panel('err', "err(x)").quiver(lambda X: model_fn_s(X) - s_ic(X))
        plotter_ic.save_plot(f'{dir_name}/viz/plot_s_nn_vs_s0.png', t_val=0.0, cbar={"nn": "linked:ic", "err": "linked:ic"})

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('nn', "s_nn(x,t)").quiver(model_fn_s)
        plotter.save_animation(f'{dir_name}/viz/anim_s_nn_fixed.gif', cbar='fixed', num_frames=30, fps=5, t_end=T)
        plotter.save_animation(f'{dir_name}/viz/anim_s_nn_dynamic.gif', cbar='dynamic', num_frames=30, fps=5, t_end=T)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('ic', title="p_0(x)").heatmap(p_ic)
        plotter.add_panel('final', title="p_inf(x)").heatmap(p_inf)
        plotter.save_plot(f'{dir_name}/viz/plot_p0_vs_p_inf.png', t_val=0.0)

    elif type_sp == "ll_ode":
        model_fn_q = viz.wrapp_model(model)
        model_fn_p = lambda X: torch.exp(model_fn_q(X))
        model_fn_s = viz.wrapp_model(model_s)
        q_ic = lambda X: pde_model.q0(X[:,:-1])

        if args.enable_testing:

            plotter = viz.FunctionPlotter(**options)
            plotter.add_panel('model_q', title="model_q(x,t)").heatmap(model_fn_q)
            plotter.add_panel('q_analytic', title="q_analytic(x,t)").heatmap(score_sde_model.q_analytic)
            plotter.add_panel('err', title="err").heatmap(lambda X: model_fn_q(X) - score_sde_model.q_analytic(X))
            plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q_analytic.png', t_val=0.234)
            plotter.save_animation(f'{dir_name}/viz/anim_model_q_vs_q_analytic.gif', num_frames=30, fps=5)

            plotter = viz.FunctionPlotter(**options)
            plotter.add_panel('model_p', title="model_p(x,t)").heatmap(model_fn_p)
            plotter.add_panel('p_analytic', title="p_analytic(x,t)").heatmap(score_sde_model.p_analytic)
            plotter.add_panel('err', title="err").heatmap(lambda X: model_fn_p(X) - score_sde_model.p_analytic(X))
            plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p_analytic.png', t_val=0.234)
            plotter.save_animation(f'{dir_name}/viz/anim_model_p_vs_p_analytic.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_q', title="model_q(x,0)").heatmap(model_fn_q)
        plotter.add_panel('q_ic', title="q_0(x)").heatmap(q_ic)
        plotter.save_plot(f'{dir_name}/viz/plot_model_q_vs_q0.png', t_val=0.0)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_p', title="model_p(x,0) = exp(model_q(x,0))").heatmap(model_fn_p)
        plotter.add_panel('p_ic', title="p_0(x)").heatmap(p_ic)
        plotter.save_plot(f'{dir_name}/viz/plot_model_p_vs_p0.png', t_val=0.0)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_q', title="model_q(x,t)").heatmap(model_fn_q)
        plotter.save_animation(f'{dir_name}/viz/anim_model_q.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(**options)
        plotter.add_panel('model_p', title="model_p(x,t) = exp(model_q(x,t))").heatmap(model_fn_p)
        plotter.save_animation(f'{dir_name}/viz/anim_model_p.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(**options)
        p = plotter.add_panel('sq', title="model_s & model_q")
        p.heatmap(model_fn_q)
        p.quiver(model_fn_s, color='k')
        p = plotter.add_panel('sp', title="model_s & model_p")
        p.heatmap(model_fn_p)
        p.quiver(model_fn_s, color='k')
        plotter.save_animation(f'{dir_name}/viz/anim_model_sq_sp.gif', num_frames=30, fps=5)
