"""
Compare PINN model outputs as 2D heatmaps in a Dash dashboard.

Three operating modes:
    # Two models (shows model_a, model_b, and their difference)
    python launch_dash_webpage.py <dir1> <dir2>

    # Model vs analytic solution (picks u_analytic / q_analytic / p_analytic from the
    # pde class found in pde_models.py or case1_OrnsteinUhlenbeck/pde_model_sde.py)
    python launch_dash_webpage.py <dir> --analytic

    # Model vs the stationary distribution p_inf or its log q_inf = log(p_inf)
    # (2 panels, no diff panel)
    python launch_dash_webpage.py <dir> --p_inf
    python launch_dash_webpage.py <dir> --q_inf

Each directory must contain: model.pth, model_metadata.json, pde_metadata.json.

Colorbar control (viz.py-style dict keyed by panel label):
    --cbar '{"model": "symmetric", "diff": "linked:model"}'
Specs:  "dynamic"  (default — per-frame min/max)
        "fixed"    (global min/max precomputed over t in [0, t_max])
        "symmetric" (symmetric around 0)
        "linked:<label>"   (use the resolved range of another panel)
        [lo, hi]           (explicit range)

Panel labels:
    two_models : "model_a", "model_b", "diff"
    analytic   : "model",   "analytic", "diff"
    p_inf      : "model",   "p_inf"
    q_inf      : "model",   "q_inf"
"""
import argparse
import json
import os
import sys

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "src"))
sys.path.append(os.path.join(_HERE, "case1_OrnsteinUhlenbeck"))

import torch
import architecture
import pde_models
import utility


# --- CLI --------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dir1", type=str,
                    help="First run directory (model.pth + metadata).")
parser.add_argument("dir2", nargs="?", default=None, type=str,
                    help="Second run directory — triggers two-model comparison.")
parser.add_argument("--analytic", action="store_true",
                    help="Compare model vs analytic solution.")
parser.add_argument("--p_inf", action="store_true",
                    help="Compare model vs p_inf (stationary distribution).")
parser.add_argument("--q_inf", action="store_true",
                    help="Compare model vs q_inf = log(p_inf) (log-stationary distribution).")
parser.add_argument("--analytic-fn", default=None, type=str,
                    help="Override analytic-fn selection "
                         "(one of: u_analytic, q_analytic, p_analytic, s_analytic).")
parser.add_argument("--model-transform", default="none",
                    choices=["none", "exp"],
                    help="Transform applied to model output (use 'exp' when model "
                         "outputs log-density and you want to compare against a density).")
parser.add_argument("--cbar", default="{}", type=str,
                    help='JSON dict of colorbar specs, e.g. \'{"diff":"linked:model"}\'.')
parser.add_argument("--nx", default=100, type=int, help="Spatial grid resolution.")
parser.add_argument("--nt", default=200, type=int, help="Number of time frames.")
parser.add_argument("--t-max", default=1.0, type=float, help="Maximum time value.")
parser.add_argument("--x-min", default=0.0, type=float, help="Min of plotted spatial range.")
parser.add_argument("--x-max", default=1.0, type=float, help="Max of plotted spatial range.")
parser.add_argument("--fixed-val", default=0.5, type=float,
                    help="Default value for dims not on the plot axes.")
parser.add_argument("--port", default=8082, type=int, help="Dash server port.")
args = parser.parse_args()

cbar_spec = json.loads(args.cbar) if args.cbar.strip() else {}

if args.p_inf and args.q_inf:
    raise SystemExit("Choose only one of --p_inf / --q_inf.")
if args.dir2 is not None:
    MODE = "two_models"
elif args.p_inf:
    MODE = "p_inf"
elif args.q_inf:
    MODE = "q_inf"
else:
    MODE = "analytic"
print(f"Mode: {MODE}")


# --- loader -----------------------------------------------------
def _infer_output_dim(state_dict):
    """Infer PINN output_dim from a saved state_dict (supports nn.Linear and RWFLinear)."""
    # modified_mlp path
    for key in ("out_layer.weight", "out_layer.V"):
        if key in state_dict:
            return state_dict[key].shape[0]
    # Standard Sequential("net.0.weight", "net.0.V", ...)
    net_keys = [k for k in state_dict
                if k.startswith("net.") and (k.endswith(".weight") or k.endswith(".V"))]
    if net_keys:
        last_idx = max(int(k.split(".")[1]) for k in net_keys)
        for suffix in (".weight", ".V"):
            k = f"net.{last_idx}{suffix}"
            if k in state_dict:
                return state_dict[k].shape[0]
    raise ValueError("Could not infer output_dim from state_dict.")


def _load_pde_model(pde_metadata, d):
    pde_class_name = pde_metadata["pde_class"]
    classes = utility.get_module_classes(pde_models)
    if pde_class_name in classes:
        pde_model = classes[pde_class_name](d)
    else:
        import pde_model_sde
        sde_classes = utility.get_module_classes(pde_model_sde)
        if pde_class_name not in sde_classes:
            raise ValueError(
                f"Unknown pde_class '{pde_class_name}' "
                "(not found in pde_models.py or pde_model_sde.py)."
            )
        pde_model = sde_classes[pde_class_name](d)
    pde_model.load_pde_metadata(pde_metadata)
    return pde_model


def load_dir(dir_name):
    print(f"Loading: {dir_name}")
    model_metadata = utility.json_load(f"{dir_name}/model_metadata.json")
    pde_metadata = utility.json_load(f"{dir_name}/pde_metadata.json")

    d = model_metadata["args"]["d"]
    D = d + 1
    layers = utility.layers_from_string(model_metadata["args"]["layers"])

    state_dict = torch.load(f"{dir_name}/model.pth", map_location="cpu", weights_only=True)
    output_dim = _infer_output_dim(state_dict)

    model_class_name = model_metadata["model_class"]
    model_cls = utility.get_module_classes(architecture)[model_class_name]
    model = model_cls(D, layers, output_dim)
    model.load_state_dict(state_dict)
    model.eval()

    pde_model = _load_pde_model(pde_metadata, d)
    print(f"  d={d}, output_dim={output_dim}, pde_class={pde_metadata['pde_class']}")
    return {"model": model, "pde_model": pde_model, "d": d,
            "output_dim": output_dim, "meta": model_metadata}


# --- load and resolve panels ------------------------------------
run1 = load_dir(args.dir1)
d = run1["d"]
if MODE == "two_models":
    run2 = load_dir(args.dir2)
    assert run2["d"] == d, f"d mismatch: dir1 has d={d}, dir2 has d={run2['d']}"


def wrap_model(model, transform="none"):
    def f(X):
        with torch.no_grad():
            y = model(X)
            if transform == "exp":
                y = torch.exp(y)
            if y.ndim == 2 and y.shape[1] > 1:
                y = y[:, 0:1]
        return y
    return f


def pick_analytic_fn(pde_model, override=None):
    names = [override] if override else ["u_analytic", "q_analytic", "p_analytic", "s_analytic"]
    for name in names:
        if name and hasattr(pde_model, name):
            print(f"Using analytic fn: {name}")
            return getattr(pde_model, name)
    raise ValueError(
        f"No analytic function on pde_model {type(pde_model).__name__}. "
        "Try --p_inf, or pass --analytic-fn=<name>."
    )


fun_1 = wrap_model(run1["model"], args.model_transform)
if MODE == "two_models":
    fun_2 = wrap_model(run2["model"], args.model_transform)
    panels = [
        {"label": "model_a", "title": f"model_a: {os.path.basename(args.dir1.rstrip('/'))}", "fn": fun_1},
        {"label": "model_b", "title": f"model_b: {os.path.basename(args.dir2.rstrip('/'))}", "fn": fun_2},
        {"label": "diff",    "title": "model_a - model_b",                                   "fn": "diff:model_a-model_b"},
    ]
    colorscales = ["Plasma", "Plasma", "RdBu"]
elif MODE == "analytic":
    fun_2 = pick_analytic_fn(run1["pde_model"], args.analytic_fn)
    panels = [
        {"label": "model",    "title": "model",            "fn": fun_1},
        {"label": "analytic", "title": "analytic",         "fn": fun_2},
        {"label": "diff",     "title": "model - analytic", "fn": "diff:model-analytic"},
    ]
    colorscales = ["Plasma", "Plasma", "RdBu"]
else:  # p_inf / q_inf
    p_inf_fn = run1["pde_model"].p_inf

    if MODE == "p_inf":
        def fun_stat(X):
            with torch.no_grad():
                return p_inf_fn(X[:, :-1])
        stat_label = "p_inf"
    else:  # q_inf
        def fun_stat(X):
            with torch.no_grad():
                return torch.log(p_inf_fn(X[:, :-1]))
        stat_label = "q_inf"

    panels = [
        {"label": "model",    "title": "model",    "fn": fun_1},
        {"label": stat_label, "title": stat_label, "fn": fun_stat},
    ]
    colorscales = ["Plasma", "Plasma"]

N_PANELS = len(panels)
PANEL_LABELS = [p["label"] for p in panels]
PANEL_TITLES = [p["title"] for p in panels]
print(f"Panels: {PANEL_LABELS}")
print(f"Cbar specs: {cbar_spec}")


# --- grid setup -------------------------------------------------
nx, nt, t_max = args.nx, args.nt, args.t_max
xi = torch.linspace(args.x_min, args.x_max, nx)
xj = torch.linspace(args.x_min, args.x_max, nx)
Xi_grid, Xj_grid = torch.meshgrid(xi, xj, indexing="ij")
xi_flat = Xi_grid.reshape(-1, 1)
xj_flat = Xj_grid.reshape(-1, 1)


def define_domain(plot_dims, x_vals):
    x_flat_list = []
    for di in range(d):
        if di == plot_dims[0]:
            x_flat_list.append(xi_flat)
        elif di == plot_dims[1]:
            x_flat_list.append(xj_flat)
        else:
            x_flat_list.append(torch.ones_like(xi_flat) * x_vals[di])
    return x_flat_list


t_val_init = 0.0
x_vals_init = args.fixed_val * torch.ones(d)
plot_dims = [0, 1]
x_flat_list = define_domain(plot_dims, x_vals_init)
t_flat = torch.ones_like(xi_flat) * t_val_init
X = torch.cat([*x_flat_list, t_flat], dim=1)


# --- panel evaluation -------------------------------------------
def eval_panels(X):
    """Return a list of (nx,nx) numpy arrays, one per panel."""
    vals = {}
    for p in panels:
        fn = p["fn"]
        if isinstance(fn, str) and fn.startswith("diff:"):
            continue
        y = fn(X).reshape(nx, nx)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        vals[p["label"]] = y

    grids = []
    for p in panels:
        fn = p["fn"]
        if isinstance(fn, str) and fn.startswith("diff:"):
            a, b = fn[len("diff:"):].split("-")
            grids.append(vals[a] - vals[b])
        else:
            grids.append(vals[p["label"]])
    return grids


# --- cbar resolution (viz.py semantics) -------------------------
def _range_from_spec(spec, Y, global_range):
    if spec == "dynamic" or spec is None:
        return (float(Y.min()), float(Y.max()))
    if spec == "symmetric":
        M = max(abs(float(Y.min())), abs(float(Y.max())))
        return (-M, M)
    if spec == "fixed":
        return global_range if global_range is not None else (float(Y.min()), float(Y.max()))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (float(spec[0]), float(spec[1]))
    return (float(Y.min()), float(Y.max()))


def resolve_ranges(Y_grids, cbar_spec, global_ranges):
    label_to_idx = {PANEL_LABELS[i]: i for i in range(N_PANELS)}
    out = []
    for i in range(N_PANELS):
        spec = cbar_spec.get(PANEL_LABELS[i], "dynamic")
        if isinstance(spec, str) and spec.startswith("linked:"):
            ref = spec[len("linked:"):]
            if ref not in label_to_idx:
                raise ValueError(f"linked:{ref} — label not found in {PANEL_LABELS}")
            j = label_to_idx[ref]
            ref_spec = cbar_spec.get(ref, "dynamic")
            out.append(_range_from_spec(ref_spec, Y_grids[j], global_ranges[j]))
        else:
            out.append(_range_from_spec(spec, Y_grids[i], global_ranges[i]))
    return out


def _needs_global_ranges():
    for label in PANEL_LABELS:
        spec = cbar_spec.get(label, "dynamic")
        if spec == "fixed":
            return True
        if isinstance(spec, str) and spec.startswith("linked:"):
            ref = spec[len("linked:"):]
            if cbar_spec.get(ref, "dynamic") == "fixed":
                return True
    return False


def precompute_global_ranges(n_samples=40):
    if not _needs_global_ranges():
        return [None] * N_PANELS
    print(f"Precomputing global color ranges ({n_samples} t-samples)...")
    lo = [float("inf")] * N_PANELS
    hi = [float("-inf")] * N_PANELS
    x_flat_local = define_domain(plot_dims, x_vals_init)
    for ti in range(n_samples):
        t = t_max * ti / max(1, n_samples - 1)
        X_t = torch.cat([*x_flat_local, torch.ones_like(xi_flat) * t], dim=1)
        grids = eval_panels(X_t)
        for i, g in enumerate(grids):
            lo[i] = min(lo[i], float(g.min()))
            hi[i] = max(hi[i], float(g.max()))
    ranges = [(lo[i], hi[i]) for i in range(N_PANELS)]
    for lbl, r in zip(PANEL_LABELS, ranges):
        print(f"  global[{lbl}] = ({r[0]:.4g}, {r[1]:.4g})")
    return ranges


GLOBAL_RANGES = precompute_global_ranges()


# --- figure builders --------------------------------------------
def create_figure(Y_grid, title, colorscale, zmin=None, zmax=None):
    fig = go.Figure(
        data=go.Heatmap(
            z=Y_grid,
            x=xi.numpy(),
            y=xj.numpy(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(len=0.75, thickness=7),
        )
    )
    fig.update_layout(
        title=dict(text=title, y=0.98, yanchor="top"),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="x1",
        yaxis_title="x2",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
    )
    return fig


def get_coord_names(d):
    return [f"x{i+1}" for i in range(d)]


# --- initial evaluation -----------------------------------------
Y_grids_init = eval_panels(X)
ranges_init = resolve_ranges(Y_grids_init, cbar_spec, GLOBAL_RANGES)
figs = [
    create_figure(Y_grids_init[i], PANEL_TITLES[i], colorscales[i],
                  zmin=ranges_init[i][0], zmax=ranges_init[i][1])
    for i in range(N_PANELS)
]

print("session saved")


# --- Dash app ---------------------------------------------------
app = Dash(__name__)

N_SLIDERS_PER_ROW = 4
n_full_slider_rows = d // N_SLIDERS_PER_ROW
n_sliders_last_row = d - N_SLIDERS_PER_ROW * n_full_slider_rows
assert N_SLIDERS_PER_ROW * n_full_slider_rows + n_sliders_last_row == d

spatial_sliders = []
for ri in range(n_full_slider_rows + 1):
    n_sliders = N_SLIDERS_PER_ROW if ri != n_full_slider_rows else n_sliders_last_row
    if n_sliders != 0:
        spatial_sliders.append(
            html.Div(
                [html.Div(
                    [html.Label(f"x{s+1}"),
                     dcc.Slider(id=f"slider_x{s+1}", min=args.x_min, max=args.x_max,
                                value=args.fixed_val)],
                    style={"width": "30%"}
                ) for s in range(N_SLIDERS_PER_ROW * ri,
                                 N_SLIDERS_PER_ROW * ri + n_sliders)],
                style={"display": "flex", "justifyContent": "space-between"}
            )
        )

panel_width_pct = f"{100 / N_PANELS:.2f}%"
app.layout = html.Div([
    dcc.Location(id="url", refresh=True),

    html.Div([
        html.Div([
            html.Div([
                html.Div("", style={"width": "20%"}),
                html.Div("time", style={"width": "10%"}),
                html.Div(dcc.Slider(id="slider_t", min=0, max=t_max, value=0.0),
                         style={"width": "60%"}),
                html.Div(html.Button("Play", id="play-button", n_clicks=0),
                         style={"width": "10%"}),
                dcc.Interval(id="interval", interval=50, n_intervals=0, disabled=True),
                dcc.Store(id="is-playing", data=False),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Div(html.Div([
                    html.Div("", style={"width": "20%"}),
                    html.Div("xi-axis", style={"width": "10%"}),
                    html.Div(dcc.Dropdown(get_coord_names(d), "x1", id="xi-axis"),
                             style={"width": "20%"}),
                ], style={"display": "flex", "alignItems": "center"})),
                dcc.Store(id="xi-axis-prev", data="x1"),
                html.Div(html.Div([
                    html.Div("", style={"width": "20%"}),
                    html.Div("xj-axis", style={"width": "10%"}),
                    html.Div(dcc.Dropdown(get_coord_names(d), "x2", id="xj-axis"),
                             style={"width": "20%"}),
                ], style={"display": "flex", "alignItems": "center"})),
                dcc.Store(id="xj-axis-prev", data="x2"),
            ]),
        ], style={"width": "50%"}),
        html.Div([], style={"width": "50%"}),
    ], style={"display": "flex", "alignItems": "center"}),

    html.Div([
        html.Div([dcc.Graph(figure=figs[i], id=f"fig{i+1}")],
                 style={"width": panel_width_pct, "display": "inline-block"})
        for i in range(N_PANELS)
    ]),

    *spatial_sliders,
])


@app.callback(
    Output("is-playing", "data"),
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, is_playing):
    new_state = not is_playing
    return new_state, (not new_state), ("Pause" if new_state else "Play")


@app.callback(
    Output("slider_t", "value"),
    Input("interval", "n_intervals"),
    State("slider_t", "value"),
)
def advance_slider_t(n_intervals, value):
    value = 0.0 if value is None else value
    value += t_max / nt
    return 0.0 if value > t_max else value


@app.callback(
    [Output(f"fig{i+1}", "figure") for i in range(N_PANELS)] +
    [Output(f"slider_x{i+1}", "disabled") for i in range(d)] +
    [Output("xi-axis-prev", "data"), Output("xj-axis-prev", "data")],
    Input("url", "href"),
    Input("xi-axis", "value"),
    Input("xj-axis", "value"),
    Input("slider_t", "value"),
    *[Input(f"slider_x{i+1}", "value") for i in range(d)],
    State("xi-axis-prev", "data"),
    State("xj-axis-prev", "data"),
    *[State(f"fig{i+1}", "figure") for i in range(N_PANELS)],
)
def update_heatmaps(*cb_args):
    trigger = callback_context.triggered_id
    xi_axis = cb_args[1]
    xj_axis = cb_args[2]
    t_value = cb_args[3]
    x_values = list(cb_args[4:4 + d])
    xi_axis_prev = cb_args[4 + d]
    xj_axis_prev = cb_args[4 + d + 1]
    curr_figs = list(cb_args[4 + d + 2:])

    disabled_list = []
    if xi_axis == xj_axis:
        print("Illegal to set same xi xj")
    for i in range(d):
        disabled_list.append(f"x{i+1}" == xi_axis or f"x{i+1}" == xj_axis)

    global X
    if trigger == "url":
        t_flat_local = torch.ones_like(xi_flat) * t_val_init
        X = torch.cat([*x_flat_list, t_flat_local], dim=1)
    elif trigger in ("xi-axis", "xj-axis"):
        prev_axes = [int(xi_axis_prev[1:]) - 1, int(xj_axis_prev[1:]) - 1]
        X[:, prev_axes[0]] = torch.ones(nx * nx) * x_values[prev_axes[0]]
        X[:, prev_axes[1]] = torch.ones(nx * nx) * x_values[prev_axes[1]]
        new_axes = [int(xi_axis[1:]) - 1, int(xj_axis[1:]) - 1]
        X[:, new_axes[0]] = xi_flat[:, 0]
        X[:, new_axes[1]] = xj_flat[:, 0]
    elif isinstance(trigger, str) and trigger.startswith("slider_"):
        coord = trigger[len("slider_"):]
        if coord == "t":
            X[:, -1:] = t_value * torch.ones_like(xi_flat)
        elif coord.startswith("x"):
            di = int(coord[1:]) - 1
            X[:, di:di+1] = x_values[di] * torch.ones_like(xi_flat)

    Y_grids = eval_panels(X)
    ranges = resolve_ranges(Y_grids, cbar_spec, GLOBAL_RANGES)
    for i in range(N_PANELS):
        curr_figs[i]["data"][0]["z"] = Y_grids[i]
        curr_figs[i]["data"][0]["zmin"] = ranges[i][0]
        curr_figs[i]["data"][0]["zmax"] = ranges[i][1]
        curr_figs[i]["layout"]["xaxis"]["title"]["text"] = xi_axis
        curr_figs[i]["layout"]["yaxis"]["title"]["text"] = xj_axis
    return (*curr_figs, *disabled_list, xi_axis, xj_axis)


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
