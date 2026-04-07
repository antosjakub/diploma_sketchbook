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
parser.add_argument("--output_dir_name", default="run_sp_latest/", type=str, help="")
parser.add_argument("--enable_profiler", action="store_true", help="")
parser.add_argument("--profiler_report_filename", default="profiler_report", type=str, help="")
# enable transfer learning / finetuning
parser.add_argument("--starting_model", default=None, type=str, help="")
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


class Isotropic_OU:
    def __init__(self, d, Sigma=torch.tensor(10.0)):
        self.d = d
        self.mu = lambda x: -1/2*x
        self.Sigma = Sigma
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
    def L_functional(self, X, s, s_div, precomputed=None):
        return 0.5 * (
            self.sigma**2 * ( s_div + (s**2).sum(dim=1).unsqueeze(1) )
            + ( (X[:,:-1] * s).sum(dim=1).unsqueeze(1) - self.d )
        )
    def ll_ode_redisual(self, X, model_q, model_s, precomputed):
        s, _, s_div = sp_derivatives.compute_score_dt_div(model_s, X)
        q = model_q(X)
        q_t = derivatives.compute_grad(X, q, torch.ones_like(q))[:,-1:]
        return q_t - self.L_functional(X, s, s_div, precomputed)
    def score_pde_residual(self, X, model_s, precomputed):
        X.detach()
        X.requires_grad_(True)
        s, s_t, s_div = sp_derivatives.compute_score_dt_div(model_s, X)
        L = self.L_functional(X, s, s_div, precomputed)
        assert L.shape == (X.shape[0], 1)
        residual = s_t - derivatives.compute_grad(X, L, torch.ones_like(L))[:,:-1]
        return residual
    def score_ic_residual(self, X, model_s, precomputed):
        return model_s(X) - precomputed["s0"]

    def score_term_loss(self, d_dim_residual):
        loss = torch.mean(torch.sum(d_dim_residual**2, dim=1))
        return loss
    def score_pde_loss(self, X, model_s, precomputed):
        res = self.score_pde_residual(X, model_s, precomputed)
        return self.score_term_loss(res)
    def score_ic_loss(self, X, model_s, precomputed):
        res = self.score_ic_residual(X, model_s, precomputed)
        return self.score_term_loss(res)

    def q0(self, x):
        return self.dist_initial.log_prob(x)
    def s0(self, x):
        x.detach()
        x.requires_grad_(True)
        q = self.dist_initial.log_prob(x).unsqueeze(1)
        s0 = derivatives.compute_grad(x, q, torch.ones_like(q))
        return s0
    def p0(self, x):
        return torch.exp(self.dist_initial.log_prob(x))
    def p_final(self, x):
        return torch.exp(self.dist_final.log_prob(x))
    def sample_x0(self, n_samples):
        return self.dist_initial.rsample((n_samples,))
    def precompute(self, X_pde, X_ic):
        return {
            "pde": {},
            "ic": {
                "s0": self.s0(X_ic[:,:-1]).detach()
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
    def __init__(self, model, optimizer, scheduler, pde_model, loss_weighting, testing_suite, profiler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pde_model = pde_model
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
                loss_pde = self.pde_model.score_pde_loss(batch_pde[0], self.model, batch_pde[1])
            loss_ic = self.pde_model.score_ic_loss(batch_ic[0], self.model, batch_ic[1])

        # Weighted loss
        loss_value = self.loss_weighting.weight_loss([loss_pde, loss_ic])

        # Backward pass
        with record_function("backward"):
            loss_value.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with record_function("optimizer_step"):
            self.optimizer.step()

        return loss_value, (loss_pde.item(), loss_ic.item())
    


    def train_adam_minibatch(self, bs, n_steps, n_steps_decay, n_res_points, resampling_frequency=2000, testing_frequency=100, use_rbas=False, use_sdgd=False, sdgd_num_dims=None):
        """
        Train the model using Adam optimizer.
        """
        losses = []
        l2_errs = []

        if self.profiler: self.profiler.make()

        n_trajs = 1_000
        nt_steps = 100

        # batches
        loader_interior, loader_ic = sp_sampling.create_dataloaders(self.d, n_trajs, nt_steps, n_res_points, bs, self.model, self.pde_model)

        for si in range(n_steps):

            if (si + 1) % resampling_frequency == 0:
                ## Resample training data
                print("New training data arrived!")
                loader_interior, loader_ic = sp_sampling.create_dataloaders(self.d, n_trajs, nt_steps, n_res_points, bs, self.model, self.pde_model)
    
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
                # do not resample - sample at the beggining, then reuse
                #l2_err, l1_err, rel_err = self.testing_suite.test_model(model)
                #l2_errs.append(l2_err)
                print(f'Step {si+1}/{n_steps}, Loss: {loss_value.item():.6f}, '
                      f'PDE: {loss_pde:.6f}, '
                      f'IC: {loss_ic:.6f}, '
                      f'lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                      #
                      #f'L2: {l2_err:.6f}, '
                      #f'L1: {l1_err:.6f}, '
                      #f'rel_max: {rel_err:.6f}'
                )
                print(l2_errs)

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
    import os
    dir_name = args.output_dir_name
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    os.makedirs(dir_name, exist_ok=True)    
    

    pde_model = Isotropic_OU(d=d)

    print(type(pde_model))
    #print(pde_model.get_pde_metadata())

    # Select the model architecture
    if args.starting_model:
        model = torch.load(args.starting_model, weights_only=False)
    else:
        model = architecture.PINN(D, layers, d).to(device)
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
    testing_suite = None
    #testing_suite = utility.TestingSuite(d)
    #testing_suite.make_test_data(pde_model, args.n_test_calloc_points, f"{dir_name}/testing_data.pt")
    #testing_suite.connect_test_data(f"{dir_name}/testing_data.pt")
    trainer = ScorePINN_Trainer(model, optimizer, scheduler, pde_model, loss_weighting, testing_suite, profiler, device)
    losses_adam, l2_errs_adam = trainer.train_adam_minibatch(
        bs=args.bs,
        n_steps=args.n_steps,
        n_steps_decay=args.n_steps_decay,
        n_res_points=args.n_calloc_points,
        resampling_frequency=args.resampling_frequency,
        testing_frequency=args.testing_frequency,
        use_rbas=args.use_rbas,
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
    torch.save(model, f'{dir_name}/model_adam.pth')

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
    model_fn = viz.wrapp_model(model)
    if True:
        p_ic = lambda X: pde_model.p0(X[:,:-1])
        p_final = lambda X: pde_model.p_final(X[:,:-1])

        #plotter_ic = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        #plotter_ic.add_scalar_fn(model_fn, "PINN")
        #plotter_ic.add_scalar_fn(p_ic, "Initial Condition")
        #plotter_ic.add_scalar_fn(lambda X: torch.abs(model_fn(X) - p_ic(X)), "Error", cmap='hot')
        #plotter_ic.save_plot(f'{dir_name}/pinn_plot_ic.png', t_val = 0.0)

        #plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        #plotter.add_scalar_fn(model_fn, "PINN")
        #plotter.save_animation(f'{dir_name}/pinn_anim.gif', num_frames=30, fps=5)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_scalar_fn(p_ic, "Initial Distribution")
        plotter.add_scalar_fn(p_final, "Final Distribution")
        plotter.save_plot(f'{dir_name}/pinn_plot_initial_final.png', t_val = 0.3)

        plotter_ic = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter_ic.add_vector_fn(model_fn, "PINN", scalar_fn=p_ic)
        plotter_ic.save_plot(f'{dir_name}/pinn_plot_model_s_vs_ic.png', t_val = 0.0)

        plotter = viz.FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
        plotter.add_vector_fn(model_fn, "PINN")
        plotter.save_plot(f'{dir_name}/pinn_plot.png', t_val = 0.3)
        plotter.save_animation(f'{dir_name}/pinn_anim.gif', num_frames=30, fps=5)
    else:
        plotter = viz.FunctionPlotter(d=d, device=device)
        plotter.add_scalar_fn(model_fn, "PINN Solution")
        plotter.add_scalar_fn(pde_model.u_analytic, "Analytic Solution")
        plotter.add_scalar_fn(lambda X: torch.abs(model_fn(X) - pde_model.u_analytic(X)), "Error", cmap='hot')
        plotter.save_plot(f'{dir_name}/pinn_fig.png', t_val = 0.325)
        plotter.save_plot(f'{dir_name}/pinn_fig_ic.png', t_val = 0.0)
        plotter.save_animation(f'{dir_name}/pinn_anim.gif', num_frames=30, fps=5)

        #visualize_training_metrics.plot_l2(steps, l2_errs, l2_name)
        #import visualize_solution_3plots
        #visualize_solution_3plots.plot_3(model, pde_model.u_analytic, d, dir_name)
        #import visualize_solution_3anims
        #visualize_solution_3anims.anim_3(model, pde_model.u_analytic, d, dir_name)