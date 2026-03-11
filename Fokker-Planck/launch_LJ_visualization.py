"""
Lennard-Jones Potential Visualizer
===================================
Run:
    pip install dash plotly numpy torch
    python lj_app.py
Then open http://127.0.0.1:8050 in your browser.

Controls:
    - Dropdown  : choose number of atoms (n)
    - Slider    : set colorbar clamp range (±scale)
    - Drag atoms: click-and-drag the labelled markers on the plot
"""

import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ── Potential ────────────────────────────────────────────────────────────────

R0      = 0.1
EPSILON = 1.0
GRID_N  = 100          # heatmap resolution
DOMAIN  = (0.0, 10.0)  # spatial extent

def potential(point: torch.Tensor) -> float:
    """LJ potential for n atoms given as (n x 2) tensor."""
    m = point.shape[0]
    result = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            rij = R0 / torch.linalg.vector_norm(point[i] - point[j])
            result = result + 4 * EPSILON * (rij**12 - rij**6)
    return float(result)


def probe_potential_grid(atoms_xy: np.ndarray) -> np.ndarray:
    """
    Evaluate potential of a probe atom placed at every grid point
    due to all atoms in atoms_xy (n x 2 numpy array).
    Returns a (GRID_N x GRID_N) array.
    """
    lo, hi = DOMAIN
    xs = np.linspace(lo, hi, GRID_N)
    ys = np.linspace(lo, hi, GRID_N)
    grid = np.zeros((GRID_N, GRID_N))
    atoms_t = torch.tensor(atoms_xy, dtype=torch.float64)
    for gi, py in enumerate(ys):
        for gj, px in enumerate(xs):
            probe = torch.tensor([[px, py]], dtype=torch.float64)
            pts   = torch.cat([atoms_t, probe], dim=0)
            # only sum pairs involving the probe (last row)
            v = 0.0
            for k in range(len(atoms_xy)):
                r = torch.linalg.vector_norm(atoms_t[k] - probe[0])
                if r < 1e-3:
                    v = 1e9
                    break
                rij = R0 / r
                v  += float(4 * EPSILON * (rij**12 - rij**6))
            grid[gi, gj] = v
    return grid


def default_positions(n: int) -> np.ndarray:
    """Evenly spread n atoms in the domain."""
    lo, hi = DOMAIN
    mid = (lo + hi) / 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = (hi - lo) * 0.28
    return np.column_stack([mid + r * np.cos(angles),
                             mid + r * np.sin(angles)])

# ── App layout ───────────────────────────────────────────────────────────────

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H3("Lennard-Jones Potential", style={"marginBottom": "4px"}),
    html.Div("V(r) = 4ε [(σ/r)¹² − (σ/r)⁶]   σ=1  ε=1",
             style={"color": "gray", "fontSize": "13px", "marginBottom": "16px"}),

    # ── Controls row ──
    html.Div([
        html.Div([
            html.Label("Number of atoms (n)", style={"fontSize": "13px"}),
            dcc.Dropdown(
                id="n-dropdown",
                options=[{"label": str(i), "value": i} for i in range(2, 9)],
                value=5,
                clearable=False,
                style={"width": "120px"},
            ),
        ], style={"marginRight": "40px"}),

        html.Div([
            html.Label("Colorbar scale  ±", style={"fontSize": "13px"}),
            dcc.Slider(
                id="scale-slider",
                min=1, max=20, step=1, value=5,
                marks={v: str(v) for v in [1, 5, 10, 15, 20]},
                tooltip={"always_visible": False},
                #style={"width": "320px"},
            ),
        ]),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

    # ── Main figure ──
    dcc.Graph(
        id="main-graph",
        config={
            "editable": True,          # enables annotation dragging
            "edits": {"annotationPosition": True},
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
        style={"width": "700px", "height": "620px"},
    ),

    # ── Total energy bar ──
    html.Div([
        html.Span("Total potential energy: ", style={"fontWeight": "bold"}),
        html.Span(id="energy-display", style={"fontSize": "18px", "marginLeft": "8px"}),
        html.Span(" ε", style={"color": "gray", "fontSize": "13px"}),
    ], style={
        "marginTop": "10px",
        "padding": "10px 20px",
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "display": "inline-block",
        "background": "#fafafa",
        "fontSize": "15px",
    }),

    # Hidden store for atom positions
    dcc.Store(id="atom-positions"),

], style={"fontFamily": "monospace", "padding": "24px", "maxWidth": "780px"})


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_figure(atoms: np.ndarray, scale: float) -> go.Figure:
    grid = probe_potential_grid(atoms)
    lo, hi = DOMAIN
    xs = np.linspace(lo, hi, GRID_N)
    ys = np.linspace(lo, hi, GRID_N)

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=grid,
        zmin=-scale, zmax=scale,
        colorscale="RdBu_r",
        colorbar=dict(title="V (ε)", thickness=15, len=0.9),
        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>V: %{z:.3f} ε<extra></extra>",
    ))

    # Atoms as draggable annotations
    n = len(atoms)
    for i, (ax, ay) in enumerate(atoms):
        fig.add_annotation(
            x=ax, y=ay,
            text=str(i + 1),
            showarrow=False,
            font=dict(size=13, color="white", family="monospace"),
            bgcolor="rgba(30,30,30,0.75)",
            bordercolor="white",
            borderwidth=1.5,
            borderpad=4,
            xref="x", yref="y",
            # tag so we can identify order
            name=f"atom_{i}",
        )

    fig.update_layout(
        xaxis=dict(range=[lo, hi], title="x", showgrid=False, zeroline=False),
        yaxis=dict(range=[lo, hi], title="y", showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#111",
        paper_bgcolor="white",
        dragmode="zoom",   # default; editable annotations override for annotation clicks
    )
    return fig


def atoms_from_relayout(relayout: dict, current: np.ndarray) -> np.ndarray:
    """Extract updated atom positions from relayoutData annotation drags."""
    atoms = current.copy()
    for key, val in relayout.items():
        # keys look like  "annotations[2].x"  or  "annotations[2].y"
        if key.startswith("annotations[") and (key.endswith("].x") or key.endswith("].y")):
            bracket = key.index("]")
            idx = int(key[len("annotations["):bracket])
            if idx < len(atoms):
                coord = key.split(".")[-1]   # 'x' or 'y'
                if coord == "x":
                    atoms[idx, 0] = float(val)
                else:
                    atoms[idx, 1] = float(val)
    return atoms


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("atom-positions", "data"),
    Output("main-graph",     "figure"),
    Output("energy-display", "children"),
    Output("energy-display", "style"),
    Input("n-dropdown",   "value"),
    Input("scale-slider", "value"),
    Input("main-graph",   "relayoutData"),
    State("atom-positions", "data"),
    prevent_initial_call=False,
)
def update(n, scale, relayout, stored):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Initialise or reset positions when n changes
    if stored is None or "n-dropdown" in triggered:
        atoms = default_positions(n)
    else:
        atoms = np.array(stored)
        # Handle atom drag
        if relayout and "main-graph.relayoutData" in triggered:
            atoms = atoms_from_relayout(relayout, atoms)
        # If n changed (safety), reset
        if len(atoms) != n:
            atoms = default_positions(n)

    # Clamp positions to domain
    lo, hi = DOMAIN
    atoms = np.clip(atoms, lo + 0.1, hi - 0.1)

    # Total pairwise LJ energy
    pt  = torch.tensor(atoms, dtype=torch.float64)
    E   = potential(pt)
    E_str = f"{E:.4f}" if abs(E) < 1e6 else "∞"
    color = "#1565c0" if E < -0.5 else ("#e53935" if E > 1 else "#333")
    style = {"fontSize": "18px", "marginLeft": "8px",
             "fontWeight": "bold", "color": color}

    fig = build_figure(atoms, scale)
    return atoms.tolist(), fig, E_str, style


if __name__ == "__main__":
    app.run(debug=True)