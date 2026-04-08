"""
Visualize N functions side by side on a 2D slice of a d-dimensional domain.
Supports both static plots and animated GIFs sweeping over time.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FunctionPlotter:
    """
    Evaluates and plots any number of functions on a 2D grid slice.

    Usage (static):
        plotter = FunctionPlotter(d=3, t_val=0.5)
        plotter.add(model, "PINN Solution")
        plotter.add(u_analytic, "Analytic Solution")
        plotter.add(lambda X: torch.abs(model(X) - u_analytic(X)), "Error", cmap='hot')
        plotter.save_plot("run/plots.png")

    Usage (animation):
        plotter = FunctionPlotter(d=3)
        plotter.add(model, "PINN Solution")
        plotter.add(u_analytic, "Analytic Solution")
        plotter.add(lambda X: torch.abs(model(X) - u_analytic(X)), "Error", cmap='hot')
        plotter.save_animation("run/anim.gif", num_frames=30, fps=5)
    """

    def __init__(self, d, plot_dims=[0, 1], N=100, fixed_dims_vals=None, device='cpu', x_start=0.0, x_end=1.0):
        self.d = d
        self.plot_dims = plot_dims
        self.N = N
        self.device = device
        self.fixed_dims_vals = fixed_dims_vals if fixed_dims_vals is not None else 0.2 * torch.ones(d)
        self.x_start = x_start
        self.x_end = x_end
        x = torch.linspace(x_start, x_end, N, device=device)
        y = torch.linspace(x_start, x_end, N, device=device)
        self.X_grid, self.Y_grid = torch.meshgrid(x, y, indexing='ij')
        x_flat = self.X_grid.reshape(-1, 1)
        y_flat = self.Y_grid.reshape(-1, 1)

        X_flat_list = []
        for di in range(d):
            if di == plot_dims[0]:
                X_flat_list.append(x_flat)
            elif di == plot_dims[1]:
                X_flat_list.append(y_flat)
            else:
                fixed_flat = torch.ones_like(x_flat) * self.fixed_dims_vals[di]
                X_flat_list.append(fixed_flat)

        t_flat = torch.zeros_like(x_flat)
        self.X = torch.cat([*X_flat_list, t_flat], dim=1)

        self._fns = []  # list of (fn, label, cmap)

    def add_scalar_fn(self, fn, label='', cmap='jet'):
        """Queue a scalar function. fn: (N*N, d+1) -> (N*N, 1)."""
        self._fns.append(('scalar', fn, label, cmap, None))

    def add_vector_fn(self, fn, label='', cmap='jet', scalar_fn=None):
        """Queue a vector function as a quiver plot. fn: (N*N, d+1) -> (N*N, d).
        If scalar_fn is provided, it is plotted as a contourf heatmap underneath the quiver."""
        self._fns.append(('vector', fn, label, cmap, scalar_fn))

    def _eval_all(self, t_val):
        """Evaluate all registered functions at the given t (modifies X[:,-1] in-place)."""
        self.X[:, -1] = t_val
        results = []
        for kind, fn, label, cmap, scalar_fn in self._fns:
            if kind == 'scalar':
                U = fn(self.X).reshape(self.N, self.N)
                results.append(('scalar', U, label, cmap))
            else:
                V = fn(self.X)  # (N*N, d)
                U_vec = V[:, self.plot_dims[0]].reshape(self.N, self.N)
                V_vec = V[:, self.plot_dims[1]].reshape(self.N, self.N)
                S_bg = None
                if scalar_fn is not None:
                    S_bg = scalar_fn(self.X).reshape(self.N, self.N)
                results.append(('vector', (U_vec, V_vec, S_bg), label, cmap))
        return results

    def _plot_panel(self, ax, entry, t_val, cbar=None):
        """Render one panel. If cbar is passed, update it instead of creating a new one.
        Returns the new/updated colorbar (or None if none was created)."""
        kind, data, label, cmap = entry
        ax.set_title(f'{label}\nt={t_val:.3f}')
        ax.set_xlabel(f'x{self.plot_dims[0]}')
        ax.set_ylabel(f'x{self.plot_dims[1]}')
        if kind == 'scalar':
            im = ax.contourf(self.X_grid.cpu(), self.Y_grid.cpu(), data.cpu(), levels=50, cmap=cmap)
            if cbar is None:
                cbar = plt.colorbar(im, ax=ax)
            else:
                cbar.update_normal(im)
            return cbar
        else:
            U_vec, V_vec, S_bg = data
            if S_bg is not None:
                im = ax.contourf(self.X_grid.cpu(), self.Y_grid.cpu(), S_bg.cpu(), levels=50, cmap=cmap)
                if cbar is None:
                    cbar = plt.colorbar(im, ax=ax)
                else:
                    cbar.update_normal(im)
            step = max(1, self.N // 20)
            U_s, V_s = U_vec[::step, ::step].cpu(), V_vec[::step, ::step].cpu()
            mag = torch.sqrt(U_s**2 + V_s**2)
            if S_bg is not None:
                ax.quiver(
                    self.X_grid[::step, ::step].cpu(), self.Y_grid[::step, ::step].cpu(),
                    U_s, V_s, color='k',
                )
            else:
                qv = ax.quiver(
                    self.X_grid[::step, ::step].cpu(), self.Y_grid[::step, ::step].cpu(),
                    U_s, V_s, mag, cmap=cmap,
                )
                if cbar is None:
                    cbar = plt.colorbar(qv, ax=ax, label='magnitude')
                else:
                    cbar.update_normal(qv)
            return cbar

    def save_plot(self, path, t_val=None):
        """Render all added functions at self.t_val and save to path."""
        t_val = t_val if t_val is not None else torch.rand(1).item()

        if not self._fns:
            raise RuntimeError("No plots added. Call .add() before .save().")

        plots = self._eval_all(t_val)
        n = len(plots)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, entry in zip(axes, plots):
            self._plot_panel(ax, entry, t_val, cbar=None)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved to {path}")

    def save_animation(self, path, num_frames=30, fps=5, t_start=0.0, t_end=1.0):
        """Animate all added functions sweeping t from t_start to t_end and save as GIF."""
        if not self._fns:
            raise RuntimeError("No plots added. Call .add() before .save_animation().")

        t_values = torch.linspace(t_start, t_end, num_frames)
        n = len(self._fns)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        # Create first frame and capture colorbars
        t0 = t_values[0].item()
        plots = self._eval_all(t0)
        cbars = []
        for ax, entry in zip(axes, plots):
            cb = self._plot_panel(ax, entry, t0, cbar=None)
            cbars.append(cb)

        def update(frame_idx):
            t_val = t_values[frame_idx].item()
            plots = self._eval_all(t_val)
            for ax, entry, cb in zip(axes, plots, cbars):
                ax.clear()
                self._plot_panel(ax, entry, t_val, cbar=cb)

        ani = FuncAnimation(fig, update, frames=num_frames, interval=int(1000 / fps), blit=False)
        print("Saving animation...")
        ani.save(path, writer="ffmpeg", fps=fps, dpi=100)
        plt.close(fig)
        print(f"Animation saved to {path}")


def wrapp_model(model):
    def model_wrapped(X):
        with torch.no_grad():
            return model(X)
    return model_wrapped

def main_plot_latest():
    import sys
    if len(sys.argv) > 1:
        dir_name = sys.argv[1]
    else:
        dir_name = 'run_latest'
    import utility
    model, u_analytic, pde_metadata, model_metadata = utility.header(dir_name)
    d = model_metadata["args"]["d"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(41)

    model_fn = wrapp_model(model)

    plotter = FunctionPlotter(d=d, device=device)
    plotter.add_scalar_fn(model_fn, "PINN Solution")
    plotter.add_scalar_fn(u_analytic, "Analytic Solution")
    plotter.add_scalar_fn(lambda X: torch.abs(model_fn(X) - u_analytic(X)), "Error", cmap='hot')

    plotter.save_plot(f'{dir_name}/__pinn_solution_plots.png')
    plotter.save_animation(f'{dir_name}/__pinn_solution_anim.gif', num_frames=30, fps=5)





if __name__ == "__main__":
    #main_plot_latest()

    #fn1 = lambda X: torch.sum(X[:,:-1]**2, dim=1).unsqueeze(dim=1)
    #fn2 = lambda X: fn1(X) + X[:,-1:]
    #plotter = FunctionPlotter(d=2)
    #plotter.add_scalar_fn(fn1, "F1", cmap='hot')
    #plotter.add_scalar_fn(fn2, "F1", cmap='hot')
    #plotter.add_scalar_fn(lambda X: torch.abs(fn1(X) - fn2(X)), "Error", cmap='hot')
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)
    #plotter.save_animation(f'__pinn_solution_anim.gif', num_frames=30, fps=5)


    #fn_scal = lambda X: torch.sum(X[:,:-1]**2, dim=1).unsqueeze(dim=1)
    #fn_vec = lambda X: X[:,:-1]
    #plotter = FunctionPlotter(d=2)
    #plotter.add_scalar_fn(fn_scal, "scal")
    #plotter.add_vector_fn(fn_vec, "vec & scal", scalar_fn=fn_scal)
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)

    from main_score_pinn import GeneralGaussian
    d = 3
    g_obj = GeneralGaussian(d, gamma_min=0.5, gamma_max=1.5, x0=0.5*torch.ones(d))
    p = lambda X: g_obj.p(X[:,:-1], X[:,-1:])
    log_p = lambda X: g_obj.log_p(X[:,:-1], X[:,-1:])
    s = lambda X: g_obj.s(X[:,:-1], X[:,-1:])

    plotter = FunctionPlotter(d=d)
    plotter.add_scalar_fn(p, "p(x,t)")
    plotter.add_scalar_fn(log_p, "q(x,t)")
    plotter.add_vector_fn(s, "s(x,t)")
    plotter.save_plot(f'general_gaussian.png', t_val=0.0)
    #plotter.save_animation(f'general_gaussian.png')



    #import sys, os
    #fp_dir = os.path.join(os.path.dirname(__file__), '../../Fokker-Planck')
    #sys.path.append(fp_dir)
    #import utility_fp
    #mol_coords = torch.tensor(utility_fp.read_mol(os.path.join(fp_dir, 'molecules/mol.1')))

    #n_atoms = 3
    #dof_per_atom = 2
    #d = n_atoms * dof_per_atom
    #mol_coords_ss = mol_coords[:n_atoms,:dof_per_atom]
    #print(mol_coords_ss)

    #L = 10.0
    #mean = torch.mean(mol_coords, dim=0)
    #mol_coords_ss -= mean.unsqueeze(dim=1)
    #mol_coords_ss /= L
    #mol_coords_ss += 0.5
    #print(mol_coords_ss)
    #x0 = mol_coords_ss.reshape(-1)
    #print(x0)

    #import pde_models
    #pde_model = pde_models.FokkerPlanckLJ(
    #    n_atoms=n_atoms, dof_per_atom=dof_per_atom,
    #    x0=x0,
    #    L=L
    #)
    #print(type(pde_model))
    #print(pde_model.get_pde_metadata())

    #print("-------------------------------------")
    #p_ic = lambda X: pde_model.p_ic(X[:,:-1])
    #print(pde_model.x0)
    #print(pde_model.sigma0)
    #print(pde_model.inv_sigma)
    #print(pde_model.prefactor)

    #plotter = FunctionPlotter(d=d, fixed_dims_vals=0.5*torch.ones(d))
    #plotter.add_scalar_fn(p_ic)
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)
