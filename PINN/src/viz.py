"""
Visualize N functions side by side on a 2D slice of a d-dimensional domain.
Supports both static plots and animated GIFs sweeping over time.

Usage (static):
    plotter = FunctionPlotter(d=3)
    plotter.add_panel("pinn", title="PINN Solution").heatmap(model)
    plotter.add_panel("analytic", title="Analytic Solution").heatmap(u_analytic)
    p = plotter.add_panel("error", title="|u - u_exact|")
    p.heatmap(lambda X: torch.abs(model(X) - u_analytic(X)), cmap='hot')
    plotter.save_plot("run/plots.png", t_val=0.5)

Usage (animation with cbar dict keyed by label):
    plotter.save_animation("run/anim.gif", cbar={"error": 'fixed'})

Usage (overlay):
    plotter = FunctionPlotter(d=3)
    p = plotter.add_panel("flow", title="Density + Flow")
    p.heatmap(density_fn)
    p.quiver(velocity_fn, color='k')
    plotter.save_plot("run/overlay.png", t_val=0.5)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Panel:
    """A single axes panel that holds layers (heatmap, quiver, contour)."""

    def __init__(self, label, plotter, title=None):
        self.label = label
        self.title = title if title is not None else label
        self._plotter = plotter
        self._layers = []

    def heatmap(self, fn, cmap='jet'):
        """Add a filled-contour heatmap layer. fn: (N*N, d+1) -> (N*N, 1)."""
        self._layers.append(('heatmap', fn, {'cmap': cmap}))
        return self

    def quiver(self, fn, color=None, cmap='jet'):
        """Add a quiver (vector field) layer. fn: (N*N, d+1) -> (N*N, d).
        color: fixed arrow color (e.g. 'k'). If None, arrows are colored by magnitude."""
        self._layers.append(('quiver', fn, {'color': color, 'cmap': cmap}))
        return self

    def contour(self, fn, colors='white', levels=10):
        """Add contour lines. fn: (N*N, d+1) -> (N*N, 1)."""
        self._layers.append(('contour', fn, {'colors': colors, 'levels': levels}))
        return self

    def scatter(self, points, color='k', s=10, alpha=0.6):
        """Add a scatter layer. points: tensor (M, 2) or callable t -> (M, 2)."""
        self._layers.append(('scatter', points, {'color': color, 's': s, 'alpha': alpha}))
        return self

    def eval_layers(self, X, N, plot_dims, t_val):
        """Evaluate all layers and return list of (kind, data, opts)."""
        results = []
        for kind, fn, opts in self._layers:
            if kind == 'heatmap':
                U = fn(X).reshape(N, N)
                results.append(('heatmap', U, opts))
            elif kind == 'quiver':
                V = fn(X)
                U_vec = V[:, plot_dims[0]].reshape(N, N)
                V_vec = V[:, plot_dims[1]].reshape(N, N)
                results.append(('quiver', (U_vec, V_vec), opts))
            elif kind == 'contour':
                U = fn(X).reshape(N, N)
                results.append(('contour', U, opts))
            elif kind == 'scatter':
                pts = fn(t_val) if callable(fn) else fn
                results.append(('scatter', pts, opts))
        return results

    def get_heatmap_data(self, evaluated_layers):
        """Return the heatmap data from evaluated layers, or None."""
        for kind, data, opts in evaluated_layers:
            if kind == 'heatmap':
                return data
        return None

    def get_colorable_data(self, evaluated_layers):
        """Return scalar data suitable for range computation: heatmap first, else quiver magnitude."""
        for kind, data, opts in evaluated_layers:
            if kind == 'heatmap':
                return data
        for kind, data, opts in evaluated_layers:
            if kind == 'quiver':
                U_vec, V_vec = data
                return torch.sqrt(U_vec**2 + V_vec**2)
        return None


class FunctionPlotter:
    """
    Evaluates and plots any number of function panels on a 2D grid slice.
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

        self._panels = []

    def add_panel(self, label='', title=None):
        """Create and return a new Panel. label is the key for cbar={}, title is displayed."""
        panel = Panel(label, self, title=title)
        self._panels.append(panel)
        return panel

    # -- evaluation --

    def _eval_all(self, t_val):
        """Evaluate all panels at the given t. Returns list of evaluated layer lists."""
        self.X[:, -1] = t_val
        return [panel.eval_layers(self.X, self.N, self.plot_dims, t_val) for panel in self._panels]

    # -- colorbar range helpers --

    def _make_levels(self, data, vmin=None, vmax=None, n=50):
        lo = vmin if vmin is not None else float(data.min())
        hi = vmax if vmax is not None else float(data.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return np.linspace(lo, hi, n + 1)

    def _compute_global_ranges(self, t_values):
        """Scan all frames and return per-panel (vmin, vmax) based on colorable data."""
        ranges = [None] * len(self._panels)
        for t in t_values:
            all_layers = self._eval_all(t.item())
            for i, (panel, layers) in enumerate(zip(self._panels, all_layers)):
                hm = panel.get_colorable_data(layers)
                if hm is None:
                    continue
                lo, hi = hm.min().item(), hm.max().item()
                if ranges[i] is None:
                    ranges[i] = (lo, hi)
                else:
                    ranges[i] = (min(ranges[i][0], lo), max(ranges[i][1], hi))
        return ranges

    def _resolve_cbar_modes(self, cbar, default):
        """Resolve the cbar argument into a per-panel list of specs.

        cbar can be:
          - None           -> use default for all panels
          - str            -> apply that mode to all ('dynamic', 'fixed', 'symmetric')
          - tuple(lo, hi)  -> explicit range for all
          - dict           -> per-panel overrides keyed by label (str) or index (int)
        """
        n = len(self._panels)
        if cbar is None:
            cbar = default

        if isinstance(cbar, (str, tuple)):
            specs = [cbar] * n
        elif isinstance(cbar, dict):
            specs = [default] * n
            label_to_idx = {self._panels[i].label: i for i in range(n)}
            for key, val in cbar.items():
                if isinstance(key, int):
                    specs[key] = val
                else:
                    specs[label_to_idx[key]] = val
        else:
            specs = [default] * n

        return specs

    def _resolve_ranges(self, specs, global_ranges):
        """Convert spec list into per-panel (vmin, vmax) or None."""
        n = len(specs)
        label_to_idx = {self._panels[i].label: i for i in range(n)}
        ranges = [None] * n

        for i, spec in enumerate(specs):
            if isinstance(spec, tuple):
                ranges[i] = spec
            elif spec == 'fixed':
                ranges[i] = global_ranges[i]
            elif spec == 'symmetric':
                if global_ranges[i] is not None:
                    lo, hi = global_ranges[i]
                    M = max(abs(lo), abs(hi))
                    ranges[i] = (-M, M)
            elif isinstance(spec, str) and spec.startswith('linked:'):
                ref_label = spec[len('linked:'):]
                ref_idx = label_to_idx[ref_label]
                ranges[i] = ranges[ref_idx] if ranges[ref_idx] is not None else global_ranges[ref_idx]
            # 'dynamic' -> stays None

        return ranges

    # -- rendering --

    def _render_panel(self, ax, panel, layers, t_val, cbar=None, vmin=None, vmax=None):
        """Render all layers of a panel onto an axes. Returns colorbar."""
        ax.set_title(f'{panel.title}\nt={t_val:.3f}')
        ax.set_xlabel(f'x{self.plot_dims[0]}')
        ax.set_ylabel(f'x{self.plot_dims[1]}')

        Xg = self.X_grid.cpu()
        Yg = self.Y_grid.cpu()

        for kind, data, opts in layers:
            if kind == 'heatmap':
                levels = self._make_levels(data, vmin, vmax)
                im = ax.contourf(Xg, Yg, data.cpu(), levels=levels, cmap=opts['cmap'])
                if cbar is None:
                    cbar = plt.colorbar(im, ax=ax)
                else:
                    cbar.ax.cla()
                    plt.colorbar(im, cax=cbar.ax)

            elif kind == 'quiver':
                U_vec, V_vec = data
                step = max(1, self.N // 20)
                U_s = U_vec[::step, ::step].cpu()
                V_s = V_vec[::step, ::step].cpu()
                Xq = Xg[::step, ::step]
                Yq = Yg[::step, ::step]
                if opts['color'] is not None:
                    ax.quiver(Xq, Yq, U_s, V_s, color=opts['color'])
                else:
                    mag = torch.sqrt(U_s**2 + V_s**2)
                    norm = plt.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else None
                    qv = ax.quiver(Xq, Yq, U_s, V_s, mag, cmap=opts['cmap'], norm=norm)
                    if cbar is None:
                        cbar = plt.colorbar(qv, ax=ax, label='magnitude')
                    else:
                        cbar.ax.cla()
                        plt.colorbar(qv, cax=cbar.ax)

            elif kind == 'contour':
                ax.contour(Xg, Yg, data.cpu(),
                           levels=opts['levels'], colors=opts['colors'])

            elif kind == 'scatter':
                pts = data
                if isinstance(pts, torch.Tensor):
                    pts = pts.detach().cpu().numpy()
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=opts['color'], s=opts['s'], alpha=opts['alpha'],
                           zorder=10)

        return cbar

    # -- saving --

    def _build_figure(self, t_val, cbar):
        """Build a figure with all panels rendered at t_val. Returns (fig, t_val)."""
        t_val = t_val if t_val is not None else torch.rand(1).item()

        if not self._panels:
            raise RuntimeError("No panels added.")

        specs = self._resolve_cbar_modes(cbar, default='dynamic')
        t_values = torch.tensor([t_val])
        global_ranges = self._compute_global_ranges(t_values)
        ranges = self._resolve_ranges(specs, global_ranges)

        all_layers = self._eval_all(t_val)
        n = len(self._panels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, panel, layers, rng in zip(axes, self._panels, all_layers, ranges):
            vmin, vmax = rng if rng is not None else (None, None)
            self._render_panel(ax, panel, layers, t_val, vmin=vmin, vmax=vmax)

        plt.tight_layout()
        return fig, t_val

    def save_plot(self, path, t_val=None, cbar=None):
        """Render all panels at t_val and save. Default cbar: 'dynamic'."""
        fig, _ = self._build_figure(t_val, cbar)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved to {path}")

    def show_plot(self, t_val=None, cbar=None):
        """Render all panels at t_val and display inline (for notebooks)."""
        self._build_figure(t_val, cbar)
        plt.show()

    def _build_animation(self, num_frames, fps, t_start, t_end, cbar):
        """Build a FuncAnimation over all panels sweeping t. Returns (fig, ani)."""
        if not self._panels:
            raise RuntimeError("No panels added.")

        t_values = torch.linspace(t_start, t_end, num_frames)
        n = len(self._panels)

        specs = self._resolve_cbar_modes(cbar, default='dynamic')
        needs_global = any(s in ('fixed', 'symmetric') or
                          (isinstance(s, str) and s.startswith('linked:'))
                          for s in specs)
        if needs_global:
            print("Pre-computing global color ranges...")
            global_ranges = self._compute_global_ranges(t_values)
        else:
            global_ranges = [None] * n
        ranges = self._resolve_ranges(specs, global_ranges)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        # First frame — capture colorbars
        t0 = t_values[0].item()
        all_layers = self._eval_all(t0)
        cbars = []
        for ax, panel, layers, rng in zip(axes, self._panels, all_layers, ranges):
            vmin, vmax = rng if rng is not None else (None, None)
            cb = self._render_panel(ax, panel, layers, t0, vmin=vmin, vmax=vmax)
            cbars.append(cb)

        def update(frame_idx):
            t_val = t_values[frame_idx].item()
            all_layers = self._eval_all(t_val)
            for ax, panel, layers, cb, rng in zip(axes, self._panels, all_layers, cbars, ranges):
                vmin, vmax = rng if rng is not None else (None, None)
                ax.clear()
                self._render_panel(ax, panel, layers, t_val, cbar=cb, vmin=vmin, vmax=vmax)

        ani = FuncAnimation(fig, update, frames=num_frames, interval=int(1000 / fps), blit=False)
        return fig, ani

    def save_animation(self, path, num_frames=30, fps=5, t_start=0.0, t_end=1.0, cbar=None):
        """Animate all panels sweeping t and save as GIF. Default cbar: 'dynamic'."""
        fig, ani = self._build_animation(num_frames, fps, t_start, t_end, cbar)
        print("Saving animation...")
        ani.save(path, writer="ffmpeg", fps=fps, dpi=100)
        plt.close(fig)
        print(f"Animation saved to {path}")

    def show_animation(self, num_frames=30, fps=5, t_start=0.0, t_end=1.0, cbar=None):
        """Animate all panels sweeping t and display inline (for notebooks)."""
        from IPython.display import HTML
        fig, ani = self._build_animation(num_frames, fps, t_start, t_end, cbar)
        plt.close(fig)
        return HTML(ani.to_jshtml())


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
    plotter.add_panel("pinn", title="PINN Solution").heatmap(model_fn)
    plotter.add_panel("analytic", title="Analytic Solution").heatmap(u_analytic)
    plotter.add_panel("error", title="Error").heatmap(lambda X: torch.abs(model_fn(X) - u_analytic(X)), cmap='hot')

    plotter.save_plot(f'{dir_name}/__pinn_solution_plots.png')
    plotter.save_animation(f'{dir_name}/__pinn_solution_anim.gif', num_frames=30, fps=5)


def plot_score_pde():
    from main_score_pinn import SmoluchowskiCoupledQuadraticPot
    from main_score_pinn import SmoluchowskiDoubleWell
    from main_score_pinn import SmoluchowskiCoupledDoubleWell
    from main_score_pinn import SmoluchowskiRastigin
    d = 10; D=d+1
    layers=[128,128,128]
    device='cpu'
    #mode = "score_pde"
    #dir_name = f"run_SMCQP_{mode}"


    #import architecture
    #model = architecture.PINN(D, layers, d).to(device)
    #model.load_state_dict(torch.load(f"{dir_name}/model.pth", weights_only=True))
    #model.eval()

    #A = 2.3*torch.diag(torch.ones(d)) + torch.diag(torch.ones(d-1), 1) + torch.diag(torch.ones(d-1), -1)
    if False:
        import utility
        A = utility.generate_SPD(d)
        print(A)
        score_sde_model = SmoluchowskiCoupledQuadraticPot(d=d, beta=1.0, A=A)

        a = 0.7 + 0.5*torch.rand(d)
        print(a)
        print(a[:2])
        score_sde_model = SmoluchowskiDoubleWell(d=d, beta=1.0, a=a)

        #a = 0.7 + 0.5*torch.rand(d)
        #gamma = 0.1*torch.rand(d) - 0.05
        a = 0.7 + 0.3*torch.ones(d)
        gamma = 0.03 * (2*((torch.rand(d)-0.5)>0.0).int()-1).float() # +- 0.03
        print(a)
        print(gamma)
        print(a[:2])
        print(gamma[:2])
        score_sde_model = SmoluchowskiCoupledDoubleWell(d=d, beta=1.0, a=a, gamma=gamma)
    else:
        d = 2
        gamma = 6*torch.pi*torch.ones(d)
        A = 0.3
        print(gamma)
        print(gamma[:2])
        #score_sde_model = SmoluchowskiRastigin(d=d, beta=1.0, A=0.7)
        #score_sde_model = SmoluchowskiRastigin(d=d, beta=1.0, A=A, gamma=gamma)
        score_sde_model = SmoluchowskiRastigin(d=d, beta=1.0)

    x_max = -2




    #if mode == "score_pde":
    #    pde_model = score_sde_model.Score_PDE(score_sde_model)
    #elif mode == "ll_ode":
    #    pde_model = score_sde_model.LL_ODE(score_sde_model)

    p_0 = lambda X: score_sde_model.p0(X[:,:-1])
    p_inf = lambda X: score_sde_model.p_inf(X[:,:-1])
    plotter = FunctionPlotter(d=d, device=device, fixed_dims_vals=0.0*torch.ones(d), x_start=-x_max, x_end=x_max)
    plotter.add_panel('p_0').heatmap(p_0)
    plotter.add_panel('p_inf').heatmap(p_inf)
    plotter.save_plot(f'__0_inf.png')

    #model_fn_s = wrapp_model(model)
    #s_ic = lambda X: pde_model.s0(X[:,:-1])

    #plotter_ic = FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
    #plotter_ic.add_panel('nn', rf"s_\theta(x,0)").quiver(model_fn_s)
    #plotter_ic.add_panel('ic', "s_0(x)").quiver(s_ic)
    #plotter_ic.save_plot(f'{dir_name}/viz/plot_s_nn_vs_s0.png', t_val=0.0, cbar={"nn": "linked:ic"})

    #if mode == "score_pde":
    #    plotter = FunctionPlotter(d=d, device=device, fixed_dims_vals=0.5*torch.ones(d))
    #    plotter.add_panel('nn', "s_nn(x,t)").quiver(model_fn_s)
    #    plotter.save_animation(f'{dir_name}/viz/anim_s_nn_fixed_NWK.gif', cbar='fixed', num_frames=30, fps=5)





if __name__ == "__main__":


    #plot_score_pde()

    #main_plot_latest()

    #fn1 = lambda X: torch.sum(X[:,:-1]**2, dim=1).unsqueeze(dim=1)
    #fn2 = lambda X: fn1(X) + X[:,-1:]
    #plotter = FunctionPlotter(d=2)
    #plotter.add_panel("F1").heatmap(fn1, cmap='hot')
    #plotter.add_panel("F2").heatmap(fn2, cmap='hot')
    #plotter.add_panel("Error").heatmap(lambda X: torch.abs(fn1(X) - fn2(X)), cmap='hot')
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)
    #plotter.save_animation(f'__pinn_solution_anim.gif', num_frames=30, fps=5)

    #fn_scal = lambda X: torch.sum(X[:,:-1]**2, dim=1).unsqueeze(dim=1)
    #fn_vec = lambda X: X[:,:-1]
    #plotter = FunctionPlotter(d=2)
    #plotter.add_panel("scal").heatmap(fn_scal)
    #p = plotter.add_panel("vec & scal")
    #p.heatmap(fn_scal)
    #p.quiver(fn_vec, color='k')
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)

    #from main_score_pinn import GeneralGaussian
    #d = 3
    #g_obj = GeneralGaussian(d, gamma_min=0.5, gamma_max=1.5, x0=0.5*torch.ones(d))
    #p = lambda X: g_obj.p(X[:,:-1], X[:,-1:])
    #log_p = lambda X: g_obj.log_p(X[:,:-1], X[:,-1:])
    #s = lambda X: g_obj.s(X[:,:-1], X[:,-1:])

    #plotter = FunctionPlotter(d=d)
    #plotter.add_panel("p(x,t)").heatmap(p)
    #plotter.add_panel("q(x,t)").heatmap(log_p)
    #plotter.add_panel("s(x,t)").quiver(s)
    #plotter.save_plot(f'general_gaussian.png', t_val=0.0)
    ##plotter.save_animation(f'general_gaussian.png')

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
    #plotter.add_panel("p_ic").heatmap(p_ic)
    #plotter.save_plot(f'__pinn_solution_plots.png', t_val=0.0)

    # --- cbar mode examples ---
    fn1 = lambda X: torch.sin(3 * X[:, 0:1]) * torch.exp(-X[:, -1:])
    fn2 = lambda X: torch.sin(3 * X[:, 0:1]) * torch.exp(-0.5 * X[:, -1:])
    err = lambda X: torch.abs(fn1(X) - fn2(X))
    fn_vec = lambda X: X[:, :-1]

    plotter = FunctionPlotter(d=2)
    plotter.add_panel("sol", title="Solution").heatmap(fn1)
    plotter.add_panel("ref", title="Reference").heatmap(fn2)
    plotter.add_panel("err", title="Error").heatmap(err, cmap='hot')


    # explicit range on error, symmetric on solution
    plotter.save_plot('__cbar_mixed.png', t_val=0.3, cbar={
        "sol": 'symmetric',
        "err": (0.0, 0.1),
    })
    plotter.save_plot('__cbar_same_cbar.png', t_val=0.3, cbar={
        "sol": 'linked:ref',
        "err": 'linked:ref',
    })

    # dynamic (default) — colorbar rescales each frame
    plotter.save_animation('__cbar_dynamic.gif', num_frames=20, fps=5)

    # fixed — colorbar pinned to global min/max across all frames
    plotter.save_animation('__cbar_fixed.gif', num_frames=20, fps=5, cbar='fixed')

    # per-panel: error linked to Solution's range, rest fixed
    plotter.save_animation('__cbar_linked.gif', num_frames=20, fps=5, cbar={
        "sol": 'fixed',
        "ref": 'linked:sol',
        "err": 'dynamic',
    })


    ## overlay: heatmap + quiver + contour lines
    #plotter2 = FunctionPlotter(d=2)
    #p = plotter2.add_panel("overlay", title="Heatmap + Quiver")
    #p.heatmap(fn1)
    #p.quiver(fn_vec, color='k')
    #p.contour(fn2, colors='white', levels=8)
    #plotter2.save_plot('__overlay.png', t_val=0.3)
