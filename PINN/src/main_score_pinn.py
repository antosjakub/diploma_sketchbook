import torch
import argparse
from torch.profiler import record_function
import sampling, loss, architecture, utility

parser = argparse.ArgumentParser()
parser.add_argument("--description", default="", type=str, help="Smthg to help identify it in grid search.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--d", default=2, type=int, help="Number of spatial dimensions.")
parser.add_argument("--layers", default="128,128,128", type=str, help="")
parser.add_argument("--n_steps", default=1000, type=int, help="")
parser.add_argument("--n_steps_decay", default=2000, type=int, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--gamma", default=0.9, type=float, help="Decay by 0.9 every 2000 steps.")
parser.add_argument("--lr", default=1e-3, type=float, help="")
parser.add_argument("--n_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--n_test_calloc_points", default=5_000, type=int, help="")
parser.add_argument("--testing_frequency", default=100, type=int, help="")
parser.add_argument("--resampling_frequency", default=2000, type=int, help="")
parser.add_argument("--bs", default=512, type=int, help="")
parser.add_argument("--lambda_pde", default=1.0, type=float, help="")
parser.add_argument("--lambda_ic", default=10.0, type=float, help="")
parser.add_argument("--use_adaptive_weights", action="store_true", help="Loss weighting.")
parser.add_argument("--use_rbas", action="store_true", help="Residual-based adaptive sampling")
parser.add_argument("--use_sdgd", action="store_true", help="Stochastic dimension gradient-descend (for loss in high dims)")
parser.add_argument("--sdgd_num_dims", default=None, type=int, help="Number of dimensions to use for SDGD. If None, use all dimensions.")
# smart Defaults
parser.add_argument("--output_dir", default="run_score_pinn_latest/", type=str, help="")
parser.add_argument("--clear_dir", action="store_true", help="Erase contents of the output_dir before the training starts.")

parser.add_argument("--mode", default="score_pde", type=str, help="")
#
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default="run_sp_latest/model.pth", type=str, help="")
parser.add_argument("--enable_testing", action="store_true", help="Compute L2/L1/rel errors during training (requires analytic solution).")
parser.add_argument("--n_test_points", default=5_000, type=int, help="Number of test points for the testing suite.")
# load the pde mode with default parameters, optionally use the .json file to init the class
#parser.add_argument("--pde_model_name", default=None, type=str, help="HeatEquation")
#parser.add_argument("--pde_model_args", default=None, type=str, help="pde_model_args.json")

import score_pinn_derivatives as sp_derivatives
import score_pinn_sampling as sp_sampling
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
#        s, _, s_div = sp_derivatives.compute_score_dt_div(model_s, X)
#        q = model_q(X)
#        q_t = derivatives.compute_grad(X, q, torch.ones(q))[:,-1:]
#        return q_t - self.L_functional(s, s_div, precomputed)
#    def score_pde_residual(self, model, X, precomputed):
#        s, s_t, s_div = sp_derivatives.compute_score_dt_div(model, X)
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
        self.gamma = torch.ones(d)

        # Random orthogonal matrix Q from QR of a Gaussian matrix.
        A = torch.randn(d, d, generator=g, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(A, mode="reduced")

        # Fix signs for a more uniform-looking orthogonal draw.
        # Makes diag(R) positive.
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
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
        self.mu = lambda x: -1/2*x
        self.sigma = torch.sqrt(Sigma)
        # detect whether mu is a constant or a function of x
        self.dist_initial = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d)
        )
        # the final distribution
        self.dist_final = torch.distributions.MultivariateNormal(
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



    def p_final(self, x):
        return torch.exp(self.dist_final.log_prob(x)).unsqueeze(1)
    
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
            s, s_t, s_div = sp_derivatives.compute_score_dt_div(model_s, X)
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
        #    self.d = score_sde_model.d
        #    self.L_functional = score_sde_model.L_functional
        #    self.mu = score_sde_model.mu
        #    self.sigma = score_sde_model.sigma
        #    self.sample_x0 = score_sde_model.sample_x0
        #    ##
        #    self.model_s = model_s
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
            s, _, s_div = sp_derivatives.compute_score_dt_div(self.model_s, X_pde)
            L = self.L_functional(X_pde, s, s_div)
            return {
                "pde": {
                    "L": L.detach()
                },
                "ic": {
                    "q0": self.q0(X_ic[:,:-1]).detach()
                },
            }




class ScorePINN_Trainer:
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
    def __init__(self, model, optimizer, scheduler, pde_model, sampling, loss_weighting, testing_suite, profiler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
        self.sampling = sampling
        self.loss_weighting = loss_weighting
        self.profiler = profiler
        self.testing_suite = testing_suite
        self.device = device
        self.d = self.pde_model.d

    def train_adam_step(self, batch_pde, batch_ic, use_sdgd=False, sdgd_num_dims=None):
        """
        Train a single step of the model.
        """
        self.optimizer.zero_grad()

        # Compute individual losses
        with record_function("loss"):
            if use_sdgd:
                #loss_pde = loss.sdgd_loss(batch_pde[0], self.model, self.pde_model, batch_pde[1], sdgd_num_dims)
                pass
            else:
                loss_pde = self.pde_model.pde_loss(batch_pde[0], self.model, batch_pde[1])
            loss_ic = self.pde_model.ic_loss(batch_ic[0], self.model, batch_ic[1])

        # Weighted loss
        loss_value = self.loss_weighting.weight_loss([loss_pde, loss_ic])

        # Backward pass
        with record_function("backward"):
            loss_value.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with record_function("optimizer_step"):
            self.optimizer.step()

        return loss_value, (loss_pde.item(), loss_ic.item())
    


    def train_adam_minibatch(self, n_steps, n_steps_decay, resampling_frequency=2000, testing_frequency=100, use_sdgd=False, sdgd_num_dims=None):
        """
        Train the model using Adam optimizer.
        """
        losses = []
        l2_errs = []

        if self.profiler: self.profiler.make()

        # batches
        #loader_interior, loader_ic = sp_sampling.create_dataloaders(self.d, self.model_s, n_trajs, nt_steps, n_res_points, bs, self.model, self.pde_model)
        loader_interior, loader_ic = sp_sampling.create_dataloaders(self.model, self.pde_model, **self.sampling)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                print("New training data arrived!")
                #loader_interior, loader_ic = sp_sampling.create_dataloaders(self.d, self.model_s, n_trajs, nt_steps, n_res_points, bs, self.model, self.pde_model)
                loader_interior, loader_ic = sp_sampling.create_dataloaders(self.model, self.pde_model, **self.sampling)
    
            # Start profiler context
            if self.profiler: self.profiler.start(si)
        
            # 1. all barches per step
            batch_iterator = zip(loader_interior, loader_ic)
            for batch_pde, batch_ic in batch_iterator:
                loss_value, (loss_pde, loss_ic) = self.train_adam_step(batch_pde, batch_ic, use_sdgd=use_sdgd, sdgd_num_dims=sdgd_num_dims)
            losses.append(loss_value.item())

            ## 1. variant: single batch per step
            #try:                                                                                          
            #    batch_pde, batch_ic = next(batch_iterator)                                                
            #except StopIteration:                                                                         
            #    batch_iterator = iter(zip(loader_interior, loader_ic))                                    
            #    batch_pde, batch_ic = next(batch_iterator)  


            # Step scheduler
            if (si + 1) % n_steps_decay == 0:
                self.scheduler.step()

            # Exit profiler context after the last profiled step
            if self.profiler: self.profiler.exit(si)

            # Print progress
            if (si + 1) % testing_frequency == 0:
                log = (f'Step {si+1}/{n_steps}, Loss: {loss_value.item():.6f}, '
                       f'PDE: {loss_pde:.6f}, '
                       f'IC: {loss_ic:.6f}, '
                       f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                if self.testing_suite is not None:
                    l2_err, l1_err, rel_err = self.testing_suite.test_model(self.model)
                    l2_errs.append(l2_err)
                    log += (f', L2: {l2_err:.6f}'
                            f', L1: {l1_err:.6f}'
                            f', rel_max: {rel_err:.6f}')
                print(log)

        return losses, l2_errs




# Main execution
if __name__ == "__main__":

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Arguments
    args = parser.parse_args([] if "__file__" not in globals() else None)
    torch.manual_seed(args.seed)
    d = args.d # space dims
    D = d + 1 # space + time dims
    layers = utility.layers_from_string(args.layers)
    print(f"\n{'='*60}")
    print(f"Training PINN for {d}D PDE")
    print(f"Domain: [0,1]^{d} x [0,1]")
    print(f"{'='*60}\n")

    # Prepare storage
    dir_name = args.output_dir
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]

    import os
    import shutil
    if os.path.isdir(dir_name):
        print(f"Directory already exists: '{dir_name}'")
        if args.clear_dir:
            print(f"To the trashbin with you lot...")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
    else:
        print(f"Creating new directory: '{dir_name}'")
        os.makedirs(dir_name)    
        if args.clear_dir:
            print("Why clear the new thing me asky??")
    print()
    

    type_sp = args.mode
    print(f"Training Score-PINN, type: '{type_sp}'")

    ### PREP PDE MODEL
    score_sde_model = Isotropic_OU(d=d)

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
        model = architecture.PINN(D, layers, d).to(device)
    elif type_sp == "ll_ode":
        model = architecture.PINN(D, layers, 1).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model)


    # Preparation time
    losses = [] 
    l2_errs = [] 

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Option 1: ExponentialLR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # Initialize loss weighting and profiler
    if args.use_adaptive_weights:
        loss_weighting = loss.AdaptiveWeights(weights=torch.tensor([args.lambda_pde, args.lambda_ic]))
    else:
        loss_weighting = loss.ConstantWeights(weights=[args.lambda_pde, args.lambda_ic])

    profiler = utility.Profiler(report_filename=f"{dir_name}/{args.profiler_report_filename}.txt", start_step=100, end_step=110) if args.enable_profiler else None

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

    sampling = {
        "n_trajs": 1_000,
        "nt_steps": 100,
        "T": 1.0,
        "n_res_points": args.n_calloc_points,
        "bs": args.bs,
    }
    # use_rbas??

    trainer = ScorePINN_Trainer(model, optimizer, scheduler, pde_model, sampling, loss_weighting, testing_suite, profiler, device)
    losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
        n_steps=args.n_steps,
        n_steps_decay=args.n_steps_decay,
        resampling_frequency=args.resampling_frequency,
        testing_frequency=args.testing_frequency,
        use_sdgd=args.use_sdgd,
        sdgd_num_dims=sdgd_num_dims,
    )
    losses += losses_adam
    l2_errs += l2_errs_adam
    print("\nAdam training complete!")
    t2 = time.time()
    h,m,s = utility.get_duration(t2-t1)
    train_time_str = f"Adam training completed in: "
    train_time_str += f"{h} hours " if h > 0 else ""
    train_time_str += f"{m} minutes " if m > 0 else ""
    train_time_str += f"{s} seconds"
    print(train_time_str)

    # Dan
    print("\nTraining complete!")
    
    import json
    with open(f'{dir_name}/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({"model_class": type(model).__name__, "args": args.__dict__}, f, ensure_ascii=False, indent=4)

    #pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')

    loss_name = f'{dir_name}/training_loss'
    l2_name = f'{dir_name}/training_l2_error'
    # Save the results
    torch.save(model.state_dict(), f'{dir_name}/model.pth')
    torch.save(torch.tensor(losses), f'{loss_name}.pth')
    torch.save(torch.tensor(l2_errs), f'{l2_name}.pth')
    print("\nResults saved.")

    print(l2_errs)

    # Plot results
    if False:
        pass
    else:
        n_steps_log = args.testing_frequency
        n_logged_pnts = len(l2_errs)
        steps = n_steps_log*torch.linspace(1,n_logged_pnts,n_logged_pnts, dtype=torch.int)

    import visualize_training_metrics
    visualize_training_metrics.plot_loss(losses, loss_name)

    import viz
    p_ic = lambda X: pde_model.p0(X[:,:-1])
    p_final = lambda X: pde_model.p_final(X[:,:-1])

    if type_sp == "score_pde":
        model_fn_s = viz.wrapp_model(model)
        s_ic = lambda X: pde_model.s0(X[:,:-1])

        plotter_ic = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter_ic.add_vector_fn(model_fn_s, "model_s(x,0)")
        plotter_ic.add_vector_fn(model_fn_s, "s_0(x)")
        plotter_ic.save_plot(f'{dir_name}/plot_model_s_vs_s0.png', t_val = 0.0)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_vector_fn(model_fn_s, "model_s(x,t)")
        plotter.save_animation(f'{dir_name}/anim_model_s.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(p_ic, "p_0(x)")
        plotter.add_scalar_fn(p_final, "p_T(x)")
        plotter.save_plot(f'{dir_name}/plot_p0_vs_pT.png', t_val = 0.0)

    elif type_sp == "ll_ode":
        model_fn_q = viz.wrapp_model(model)
        model_fn_p = lambda X: torch.exp(model_fn_q(X))
        model_fn_s = viz.wrapp_model(model_s)
        q_ic = lambda X: pde_model.q0(X[:,:-1])

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(model_fn_q, "model_q(x,0)")
        plotter.add_scalar_fn(q_ic, "q_0(x)")
        plotter.save_plot(f'{dir_name}/plot_model_q_vs_q0.png', t_val = 0.0)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(model_fn_p, "model_p(x,0) = exp(model_q(x,0))")
        plotter.add_scalar_fn(p_ic, "p_0(x)")
        plotter.save_plot(f'{dir_name}/plot_model_p_vs_p0.png', t_val = 0.0)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(model_fn_q, "model_q(x,t)")
        plotter.save_animation(f'{dir_name}/anim_model_q.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(model_fn_p, "model_p(x,t) = exp(model_q(x,t))")
        plotter.save_animation(f'{dir_name}/anim_model_p.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_vector_fn(model_fn_s, "model_s & model_q", scalar_fn=model_fn_q)
        plotter.add_vector_fn(model_fn_s, "model_s & model_p", scalar_fn=model_fn_p)
        plotter.save_animation(f'{dir_name}/anim_model_sq_sp.gif', num_frames=30, fps=5)
