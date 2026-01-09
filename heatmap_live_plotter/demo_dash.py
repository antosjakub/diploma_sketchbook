import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# read from metadata
t_max = 10.0
d = 6
def u_analytic(X):
    # X.shape = (batch size, spatial+time dims)
    bs, D = X.shape
    d = D-1
    alpha = 0.01
    u_space = np.prod(np.sin(np.pi * X[:,:-1]), axis=1) # .shape = (bs,)
    u_time = np.exp(- d * alpha * np.pi**2 * X[:,-1]) # .shape = (bs,)
    # return shape = (batch size, 1)
    #return (u_space * u_time).unsqueeze(dim=1)
    return np.array([u_space * u_time])


def get_coord_names(d):
    return [f"x{i+1}" for i in range(d)]


# --- create grid data ---------------------------------
nx = 100
nt = 200

xi = np.linspace(0, 1, nx)
xj = np.linspace(0, 1, nx)
Xi_grid, Xj_grid = np.meshgrid(xi, xj, indexing='ij')
xi_flat = Xi_grid.reshape(-1, 1)
xj_flat = Xj_grid.reshape(-1, 1)

# initialize - those are to be updated
t_val = 0.0
x_vals = 0.5*np.ones(d)
plot_dims = [0,1]

# define domain
x_flat_list = []
for di in range(d):
    if di == plot_dims[0]:
        x_flat_list.append(xi_flat)
    elif di == plot_dims[1]:
        x_flat_list.append(xj_flat)
    else:
        fixed_flat = np.ones_like(xi_flat) * x_vals[di]
        x_flat_list.append(fixed_flat)

t_flat = np.ones_like(xi_flat) * t_val
X = np.concatenate([*x_flat_list, t_flat], axis=1)
Y = u_analytic(X)
Y_grid = Y.reshape(nx, nx)


"""
animation
- just keep increating the value of one coordinate

slider
- set val of one coord

selecting
- keep X the same, just redraw
"""


# --- initial figure ----------------------------------------------
fig = go.Figure(
    data=go.Heatmap(
        z=Y_grid,
        x=xi_flat,
        y=xj_flat,
        colorscale="Viridis",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="u(x1,..,xd,t)"),
    )
)
fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_title="x",
    yaxis_title="y",
)

# --- Dash app ----------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div([
                    html.Div("animate", style={"width": "10%"}),
                    html.Div(
                        dcc.Dropdown(["t"] + get_coord_names(d), "t", id="animate_di"),
                        style={"width": "10%"}
                    ),
                    html.Div(dcc.Slider(id="slider_t", min=0, max=t_max, value=0.0), style={"width": "30%"}),
                    html.Div(html.Button("Play", id="play-button", n_clicks=0), style={"width": "10%"}),
                    html.Div("", style={"width": "40%"}),
                ], style={"display": "flex", "alignItems": "center"}),

                dcc.Interval(
                    id="interval",
                    interval=50, # ms between frames (20 FPS)
                    n_intervals=0,
                    disabled=True, # start paused
                ),
                dcc.Store(id="is-playing", data=False),
            ]
        ),

        html.Div([
            html.Div(
                html.Div([
                    html.Div("xi-axis", style={"width": "20%"}),
                    html.Div(
                        dcc.Dropdown(get_coord_names(d), "x1", id="xi-axis"),
                        style={"width": "80%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), style={"width": "50%"}
            ),
            #dcc.Store(id="xi-axis_prev_val", data="x1"),
            html.Div(
                html.Div([
                    html.Div("xj-axis", style={"width": "20%"}),
                    html.Div(
                        dcc.Dropdown(get_coord_names(d), "x2", id="xj-axis"),
                        style={"width": "80%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), style={"width": "50%"}
            ),
            #dcc.Store(id="xj-axis_prev_val", data="x2"),
            ]
            #], style={
            #    "display": "flex",
            #    "justifyContent": "space-between"
            #}
        ),

        dcc.Graph(id="heatmap", figure=fig, style={"height": "80vh"}),

        html.Div([
            html.Div(
                [html.Label("x1"), dcc.Slider(id="slider_x1", min=0, max=1, value=0.5)],
                style={"width": "45%"}
            ),
            html.Div(
                [html.Label("x2"), dcc.Slider(id="slider_x2", min=0, max=1, value=0.5)],
                style={"width": "45%"}
            ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between"
            }
        ),
        html.Div([
            html.Div(
                [html.Label("x3"), dcc.Slider(id="slider_x3", min=0, max=1, value=0.5)],
                style={"width": "45%"}
            ),
            html.Div(
                [html.Label("x4"), dcc.Slider(id="slider_x4", min=0, max=1, value=0.5)],
                style={"width": "45%"}
            ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between"
            }
        ),
    ]
)

@app.callback(
    [Output(f"slider_x{i+1}", "disabled") for i in range(4)],
    Input("xi-axis", "value"),
    Input("xj-axis", "value"),
)
def disable_slider(xi_chosen, xj_chosen):
    disabed_list = []
    if xi_chosen == xj_chosen:
        print("Illegal to set same xi xj")
    for i in range(4):
        if f"x{i+1}" == xi_chosen or f"x{i+1}" == xj_chosen:
            disabed_list.append(True)
        else:
            disabed_list.append(False)
    return disabed_list

# Toggle play / pause
@app.callback(
    Output("is-playing", "data"),
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("is-playing", "data"),
)
def toggle_play(n_clicks, is_playing):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    new_state = not is_playing
    return new_state, (not new_state), ("Pause" if new_state else "Play")

# Increase slider value while playing
@app.callback(
    Output("slider_t", "value"),
    Input("interval", "n_intervals"),
    State("slider_t", "value"),
)
def update_slider(n_intervals, value):
    if value is None:
        value = 0.0
    #return (value + 1) % nt
    value += 10*1/nt
    if value > t_max:
        return 0.0
    else:
        return value

from dash import callback_context
# Update heatmap when time index changes
@app.callback(
    Output("heatmap", "figure"),
    Input("slider_t", "value"),
    Input("slider_x1", "value"),
    State("heatmap", "figure"),
    prevent_initial_call=True,
)
def update_heatmap(t_value, x_value, current_fig):
    trigger = callback_context.triggered_id
    if trigger == "slider_t":
        X[:, -1:] = t_value * np.ones_like(xi_flat)
    elif trigger == "slider_x1":
        X[:, 0:1] = x_value * np.ones_like(xi_flat)
    Y = u_analytic(X)
    Y_grid = Y.reshape(nx, nx)
    current_fig["data"][0]["z"] = Y_grid
    return current_fig


if __name__ == "__main__":
    app.run(debug=True)
