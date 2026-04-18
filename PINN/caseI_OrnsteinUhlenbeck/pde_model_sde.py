import torch

import os, sys
src_dir = os.path.join(os.path.dirname(__file__), '../src/')
sys.path.append(src_dir)

import derivatives
import utility

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
        x0 = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d = d
        self.x0 = x0 if (x0 is not None) else torch.zeros(d, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device


    def random_init_gamma_Q(
        self,
        gamma_strategy="min_max",
        seed: int = 76,
    ):
        device = self.device
        dtype = self.dtype
        d = self.d
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        # Construct Gamma diagonal
        if gamma_strategy == "min_max":
            gamma_min = 1.5
            gamma_max = 3.5
            gamma = gamma_min + (gamma_max - gamma_min) * torch.rand(
                d, generator=g, device=device, dtype=dtype
            )
            self.gamma = gamma
        elif gamma_strategy == "constant":
            gamma = 10*torch.ones(d)
            self.gamma = gamma
        elif gamma_strategy == "paper":
            # Eigenvalues: pairs lambda_{2i} ~ U([1, 1.1]), lambda_{2i+1} = 1/lambda_{2i}.
            gamma = torch.empty(d, dtype=dtype, device=device)
            n_pairs = d // 2
            if n_pairs > 0:
                pair_vals = 1.0 + 0.1 * torch.rand(n_pairs, generator=g, dtype=dtype, device=device)
                gamma[0:2*n_pairs:2] = pair_vals
                gamma[1:2*n_pairs:2] = 1.0 / pair_vals
            if d % 2 == 1:
                gamma[-1] = 1.0 + 0.1 * torch.rand((), generator=g, dtype=dtype, device=device)
            self.gamma = gamma

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

        self.set_gamma_Q(gamma, Q)


    def set_gamma_Q(self, gamma, Q):
        # Sigma = Q^T Gamma Q  and  Sigma^(1/2) = Q^T Gamma^(1/2) Q.
        # (gamma.unsqueeze(1) * Q) applies diag(gamma) on the left before Q.
        self.gamma = gamma
        self.Q = Q
        self.Sigma = Q.transpose(0, 1) @ (gamma.unsqueeze(1) * Q)
        self.Sigma_sqrt = Q.transpose(0, 1) @ (torch.sqrt(gamma).unsqueeze(1) * Q)

        # Precompute dense Sigma and Sigma^(1/2) once
        # Using Sigma = Q^T diag(gamma) Q
        # Implemented as (Q^T * gamma) @ Q to avoid materializing diag(gamma)
        #QT = Q.transpose(0, 1)
        #self.Sigma = (QT * gamma.unsqueeze(1)) @ Q
        #self.Sigma_sqrt = (QT * torch.sqrt(gamma).unsqueeze(1)) @ Q


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




class InitialDistribution:
    """
    Interface for initial distributions p_0 on R^d.
    All methods take x of shape (bs, d).
    p0 / log_p0 return (bs, 1); s0 returns (bs, d); sample(n) returns (n, d).
    """
    def __init__(self, d, dtype=torch.float32, device=None):
        self.d = d
        self.dtype = dtype
        self.device = device
    def p0(self, x): raise NotImplementedError
    def log_p0(self, x): raise NotImplementedError
    def s0(self, x): raise NotImplementedError
    def sample(self, n): raise NotImplementedError


class GaussianIC(InitialDistribution):
    """p_0(x) = (2*pi)^(-d/2) exp(-||x||^2 / 2)"""
    def __init__(self, d, dtype=torch.float32, device=None):
        super().__init__(d, dtype, device)
        self._log_alpha = -0.5 * d * math.log(2.0 * math.pi)
    def log_p0(self, x):
        return self._log_alpha - 0.5 * (x**2).sum(dim=1, keepdim=True)
    def p0(self, x):
        return torch.exp(self.log_p0(x))
    def s0(self, x):
        return -x
    def sample(self, n):
        return torch.randn(n, self.d, dtype=self.dtype, device=self.device)


class CauchyIC(InitialDistribution):
    """
    Spherical d-dim Cauchy:
        p_0(x) = Gamma((d+1)/2) / (pi^((d+1)/2) (1 + ||x||^2)^((d+1)/2))
    Score:
        s_0(x) = -(d+1) x / (1 + ||x||^2)
    Sampling via N/|N|: x = z / |y|, z ~ N(0, I_d), y ~ N(0, 1).
    """
    def __init__(self, d, dtype=torch.float32, device=None):
        super().__init__(d, dtype, device)
        self._log_alpha = (
            math.lgamma(0.5 * (d + 1)) - 0.5 * (d + 1) * math.log(math.pi)
        )
    def log_p0(self, x):
        r2 = (x**2).sum(dim=1, keepdim=True)
        return self._log_alpha - 0.5 * (self.d + 1) * torch.log1p(r2)
    def p0(self, x):
        return torch.exp(self.log_p0(x))
    def s0(self, x):
        r2 = (x**2).sum(dim=1, keepdim=True)
        return -(self.d + 1) * x / (1.0 + r2)
    def sample(self, n):
        z = torch.randn(n, self.d, dtype=self.dtype, device=self.device)
        y = torch.randn(n, 1, dtype=self.dtype, device=self.device)
        return z / y.abs().clamp(min=1e-20)


class LaplaceIC(InitialDistribution):
    """
    Product of d independent Laplace(0, 1):
        p_0(x) = (1/2)^d prod_i exp(-|x_i|)
    Score (subgradient at 0):
        s_0(x) = -sign(x)
    """
    def __init__(self, d, dtype=torch.float32, device=None):
        super().__init__(d, dtype, device)
        self._log_alpha = -d * math.log(2.0)
    def log_p0(self, x):
        return self._log_alpha - x.abs().sum(dim=1, keepdim=True)
    def p0(self, x):
        return torch.exp(self.log_p0(x))
    def s0(self, x):
        return -torch.sign(x)
    def sample(self, n):
        # Inverse-CDF: U ~ Uniform(-1/2, 1/2), X = -sign(U) log(1 - 2|U|)
        u = torch.rand(n, self.d, dtype=self.dtype, device=self.device) - 0.5
        return -torch.sign(u) * torch.log1p(-2.0 * u.abs())


class Anisotropic_OU:
    """
    Anisotropic, correlated-noise OU (chapter_results.md):

        dX_t = -1/2 * x dt + Sigma^(1/2) dW_t,        p_0 chosen by subclass

    Sigma = Q^T Gamma Q with Q random orthogonal (QR of a Gaussian),
    Gamma = diag(lambda_1,...,lambda_d), pairs lambda_{2i} ~ U([1, 1.1]),
    lambda_{2i+1} = 1/lambda_{2i}.

    Only the Gaussian variant has a closed-form solution.
    """
    def __init__(self, d, initial_dist, gamma=None, Q=None, seed=76, dtype=torch.float32, device=None):
        self.d = d
        self.dtype = dtype
        self.device = device

        self.gaussian_obj = GeneralGaussian(d, dtype=dtype, device=device)
        if gamma is not None and Q is not None:
            gamma_t = torch.as_tensor(gamma, dtype=dtype, device=device)
            Q_t = torch.as_tensor(Q, dtype=dtype, device=device)
            self.gaussian_obj.set_gamma_Q(gamma_t, Q_t)
        else:
            self.gaussian_obj.random_init_gamma_Q(gamma_strategy="min_max", seed=seed)
        self._refresh_sde_cache()

        # SDE drift (diffusion is cached in _refresh_sde_cache).
        self.mu = lambda x, t: -0.5 * x

        self.initial_dist = initial_dist

    def _refresh_sde_cache(self):
        """(Re)build Sigma, sigma, and dist_inf from the current gaussian_obj state."""
        self.Sigma = self.gaussian_obj.Sigma
        # Matrix-valued diffusion coefficient consumed by sampling.py.
        self.sigma = self.gaussian_obj.Sigma_sqrt
        self.dist_inf = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.d, dtype=self.dtype, device=self.device),
            covariance_matrix=self.Sigma,
        )

    def get_pde_metadata(self):
        return {
            "d": self.d,
            "initial_dist": type(self.initial_dist).__name__,
            "gamma": self.gaussian_obj.gamma.detach().cpu().tolist(),
            "Q": self.gaussian_obj.Q.detach().cpu().tolist(),
            "Sigma": self.gaussian_obj.Sigma.detach().cpu().tolist(),
        }
    def dump_pde_metadata(self, file_path) -> None:
        pde_params = self.get_pde_metadata()
        utility.json_dump(file_path, {"pde_class": type(self).__name__, "params": pde_params})
    def __load_pde_metadata(self, pde_metadata) -> None:
        pde_class = pde_metadata["pde_class"]
        assert pde_class == type(self).__name__, f"ERROR: The given .json file specifies parameters for '{pde_class}', but this class is of type '{type(self).__name__}'."
        return pde_metadata["params"]
    def load_pde_metadata(self, pde_metadata) -> None:
        pde_params = self.__load_pde_metadata(pde_metadata)
        assert pde_params["d"] == self.d, (
            f"d mismatch: saved d={pde_params['d']}, current d={self.d}"
        )
        ic_saved = pde_params["initial_dist"]
        ic_current = type(self.initial_dist).__name__
        assert ic_saved == ic_current, (
            f"initial_dist mismatch: saved '{ic_saved}', current '{ic_current}'"
        )
        gamma = torch.tensor(pde_params["gamma"], dtype=self.dtype, device=self.device)
        Q = torch.tensor(pde_params["Q"], dtype=self.dtype, device=self.device)
        self.gaussian_obj.set_gamma_Q(gamma, Q)
        self._refresh_sde_cache()

    def p0(self, x):
        return self.initial_dist.p0(x)
    def q0(self, x):
        return self.initial_dist.log_p0(x)
    def s0(self, x):
        return self.initial_dist.s0(x)
    def sample_x0(self, n_samples):
        return self.initial_dist.sample(n_samples)
    def p_inf(self, x):
        return torch.exp(self.dist_inf.log_prob(x)).unsqueeze(1)

    def L_functional(self, X, s, s_jac_spatial, precomputed=None):
        # L = d_t q = 1/2 ( tr(Sigma grad_x s) + s^T Sigma s + x . s + d ).
        # Reduces to the isotropic form when Sigma = sigma^2 I.
        tr_Sigma_grad_s = torch.einsum('ij,bij->b', self.Sigma, s_jac_spatial).unsqueeze(1)
        quad = torch.einsum('bi,ij,bj->b', s, self.Sigma, s).unsqueeze(1)
        drift = (X[:, :-1] * s).sum(dim=1).unsqueeze(1)
        return 0.5 * (tr_Sigma_grad_s + quad + drift + self.d)

    class Score_PDE:
        def __init__(self, score_sde_model) -> None:
            self.score_sde_model = score_sde_model
        def __getattr__(self, name):
            return getattr(self.score_sde_model, name)

        def s0(self, x):
            return self.initial_dist.s0(x)
        def pde_residual(self, X, model_s, precomputed):
            X.detach()
            X.requires_grad_(True)
            s, jac = derivatives.compute_score_and_jacobian(model_s, X)
            s_t = jac[:, :, -1]                   # (bs, d)
            s_jac_spatial = jac[:, :, :-1]        # (bs, d, d)
            L = self.L_functional(X, s, s_jac_spatial, precomputed)
            assert L.shape == (X.shape[0], 1)
            residual = s_t - derivatives.compute_grad(X, L, torch.ones_like(L))[:,:-1]
            return residual
        def bc_residual(self, X, model_s, precomputed):
            # (Sigma s + x).n = 0
            n = precomputed["normals"]
            Sigma_s = torch.einsum('ij,bj->bi', self.Sigma, model_s(X))
            return ( ( Sigma_s + X[:,:-1] ) * n ).sum(dim=1).unsqueeze(1)
        def ic_residual(self, X, model_s, precomputed):
            return model_s(X) - precomputed["s0"]

        def _term_loss(self, d_dim_residual):
            loss = torch.mean(torch.sum(d_dim_residual**2, dim=1))
            return loss
        def pde_loss(self, X, model_s, precomputed):
            res = self.pde_residual(X, model_s, precomputed)
            return self._term_loss(res)
        def bc_loss(self, X, model_s, precomputed):
            res = self.bc_residual(X, model_s, precomputed)
            return torch.mean(res**2)
        def ic_loss(self, X, model_s, precomputed):
            res = self.ic_residual(X, model_s, precomputed)
            return self._term_loss(res)
        def precompute(self, X_pde, X_bc, X_ic):
            return {
                "pde": {},
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
            return self.initial_dist.log_p0(x)
        def pde_residual(self, X, model_q, precomputed):
            X.detach()
            X.requires_grad_(True)
            q = model_q(X)
            q_t = derivatives.compute_grad(X, q, torch.ones_like(q))[:,-1:]
            return q_t - precomputed["L"]
        def bc_residual(self, X, model_q, precomputed):
            # (Sigma grad q + x).n = 0
            X = X.detach().requires_grad_(True)
            q = model_q(X)
            grad_q = derivatives.compute_grad(X, q, torch.ones_like(q))[:,-1:]
            n = precomputed["normals"]
            Sigma_grad_q = torch.einsum('ij,bj->ib', self.Sigma, grad_q).unsqueeze(1)
            return ( ( Sigma_grad_q + X[:,:-1] ) * n ).sum(dim=1).unsqueeze(1)
        def ic_residual(self, X, model_q, precomputed):
            return model_q(X) - precomputed["q0"]

        def pde_loss(self, X, model_q, precomputed):
            res = self.pde_residual(X, model_q, precomputed)
            return torch.mean(res**2)
        def bc_loss(self, X, model_q, precomputed):
            res = self.bc_residual(X, model_q, precomputed)
            return torch.mean(res**2)
        def ic_loss(self, X, model_q, precomputed):
            res = self.ic_residual(X, model_q, precomputed)
            return torch.mean(res**2)
        def precompute(self, X_pde, X_bc, X_ic):
            X_pde.detach()
            X_pde.requires_grad_(True)
            s, jac = derivatives.compute_score_and_jacobian(self.model_s, X_pde)
            s_jac_spatial = jac[:, :, :-1]
            L = self.L_functional(X_pde, s, s_jac_spatial)
            return {
                "pde": {
                    "L": L.detach()
                },
                "bc": {},
                "ic": {
                    "q0": self.q0(X_ic[:,:-1]).detach()
                },
            }


class Gaussian_OU(Anisotropic_OU):
    """p_0 = N(0, I). Closed-form: p_t = N(0, Sigma_t), Sigma_t = e^{-t}I + (1-e^{-t}) Sigma."""
    def __init__(self, d, gamma=None, Q=None, seed=76, dtype=torch.float32, device=None):
        super().__init__(
            d, GaussianIC(d, dtype=dtype, device=device),
            gamma=gamma, Q=Q,
            seed=seed, dtype=dtype, device=device,
        )

    def p_analytic(self, X):
        return self.gaussian_obj.p(X[:,:-1], X[:,-1:])
    def q_analytic(self, X):
        return self.gaussian_obj.log_p(X[:,:-1], X[:,-1:])
    def s_analytic(self, X):
        return self.gaussian_obj.s(X[:,:-1], X[:,-1:])


class Cauchy_OU(Anisotropic_OU):
    """p_0 = spherical Cauchy. No analytic solution to the Fokker-Planck PDE."""
    def __init__(self, d, gamma=None, Q=None, seed=76, dtype=torch.float32, device=None):
        super().__init__(
            d, CauchyIC(d, dtype=dtype, device=device),
            gamma=gamma, Q=Q,
            seed=seed, dtype=dtype, device=device,
        )


class Laplace_OU(Anisotropic_OU):
    """p_0 = product of independent Laplace(0, 1). No analytic solution."""
    def __init__(self, d, gamma=None, Q=None, seed=76, dtype=torch.float32, device=None):
        super().__init__(
            d, LaplaceIC(d, dtype=dtype, device=device),
            gamma=gamma, Q=Q,
            seed=seed, dtype=dtype, device=device,
        )


